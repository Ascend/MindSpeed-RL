import re
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import threading
import logging

import torch
from mathruler.grader import extract_boxed_content, grade_answer

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.math_eval_toolkit.grader import math_equal
from mindspeed_rl.utils.math_eval_toolkit.parser import extract_answer
from mindspeed_rl.utils.utils import mstx_timer_decorator

logger = Loggers("Rule verify")


class GlobalProcessPool:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_workers=16, reset_threshold=100000):
        self.max_workers = max_workers
        self.reset_threshold = reset_threshold
        self.task_counter = 0
        self.executor = None
        self.logger = logging.getLogger(__name__)
        self._initialize_executor()

    def _initialize_executor(self):
        """Initialize a new ProcessPoolExecutor and reset task counter."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
            gc.collect()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_counter = 0
        self.logger.warning(f"Initialized ProcessPoolExecutor with {self.max_workers} workers")

    @classmethod
    def get_instance(cls, max_workers=16, reset_threshold=100000) -> 'GlobalProcessPool':
        """Get or create the singleton instance of GlobalProcessPool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_workers=max_workers, reset_threshold=reset_threshold)
        return cls._instance

    def submit(self, fn, *args, **kwargs):
        """
        Submit a task to the executor with automatic recovery and periodic reset.

        Args:
            fn: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Future object representing the computation
        """
        try:
            if self.executor is None:
                with self._lock:
                    self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Process pool broken, recreating: {str(e)}")
            with self._lock:
                self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)

global_executor = GlobalProcessPool.get_instance(max_workers=16)


@mstx_timer_decorator
def compute_verifier_score(batch, megatron_config, rl_config, tokenizer, ignore_token=-100):
    start_time = time.time()
    question = batch["prompts"]
    indexes = [i for i in range(0, question.size(0), rl_config.n_samples_per_prompt)]
    question = question[indexes]
    responses = torch.where(batch["responses"] == ignore_token, tokenizer.eos_token_id, batch["responses"])

    str_question = tokenizer.batch_decode(question, skip_special_tokens=True)
    str_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)

    reward_index = batch["response_length"]

    logger.info("=" * 50)
    logger.info(">>>>>>>>>> User:\n")
    logger.info(str_question[0])
    logger.info(">>>>>>>>>> Assistant:\n")
    logger.info(str_responses[0])

    extra_data = {}

    if hasattr(megatron_config, "dataset_additional_keys"):
        for k in megatron_config.dataset_additional_keys:
            extra_data[k] = tokenizer.batch_decode(batch[k], skip_special_tokens=True)
            logger.info(f">>>>>>>>>> {k}")
            logger.info(extra_data[k][0])

    logger.info("=" * 50)

    scores, metrics = verifier(str_responses, extra_data, rl_config)

    scores = torch.tensor(
        scores,
        dtype=torch.float64,
        device=reward_index.device
    )

    scores = scores.reshape(-1, rl_config.n_samples_per_prompt)
    scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)
    scores = scores.reshape(reward_index.shape)

    scores = torch.tensor(
        scores,
        dtype=torch.float32,
        device=reward_index.device
    )

    end_time = time.time()
    metrics["timing/rule_reward"] = [round(end_time, 4), round(start_time, 4)]
    metrics["start_time/rule_reward"] = [round(start_time, 4)]
    metrics["end_time/rule_reward"] = [round(end_time, 4)]


    return scores, metrics


def verifier(responses, data, config, **kwargs):
    """
    User-defined verifier scoring process.

    Parameters:
    ----------
    responses(List[`str`]):
        Actor rollout answers.
    labels(List[`str`]):
        Ground Truth.
    infos(List[`str`], *optional*):
         Additional usable information loaded from the dataset.

    Return:
        scores(List[`float`]): Final scores.
    """
    rule_verifier_function = {
        "acc": accuracy_reward,
        "format": format_reward,
        "step": reasoning_steps_reward,
        "strict_format": strict_format_reward,
        "base_acc": base_model_accuracy_reward,
    }

    labels = data["labels"]
    rewards = [0.0] * len(labels)
    metrics = {}

    verifier_function = config.verifier_function
    verifier_weight = config.verifier_weight

    for idx, fun_verifier in enumerate(verifier_function):
        if fun_verifier not in rule_verifier_function:
            continue
        scores = rule_verifier_function[fun_verifier](sequences=responses, answers=labels)

        metrics[f'grpo/{fun_verifier}_rewards/mean'] = scores
        rewards = [all_score + tmp_score * verifier_weight[idx]
                  for all_score, tmp_score in zip(rewards, scores)]

    return rewards, metrics


def base_acc_worker(sequence, answer, timeout=False):
    format_correct = _validate_response_structure(sequence)
    ext_answer = extract_answer(pred_str=sequence, data_name="math")
    box_match = 0.0
    if math_equal(prediction=ext_answer, reference=answer, timeout=False) and format_correct:
        box_match = 1.0
    return box_match


def base_acc_subprocess(prediction, reference, timeout_seconds=1):
    try:
        future = global_executor.submit(base_acc_worker, prediction, reference)
        result = future.result(timeout=timeout_seconds)
        return result
    except TimeoutError:
        return 0.0
    except Exception as e:
        return 0.0


def new_format_and_acc(sequence, answer):
    pattern = re.compile(r"<think>.*</think>.*<answer>.*\\boxed\{.*\}.*</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, sequence)

    extract_output = extract_boxed_content(sequence)
    if grade_answer(extract_output, answer) and format_match:
        return 1.0
    else:
        return 0.0


def base_model_accuracy_reward(sequences, answers, *args, **kwargs):
    scores = []
    for sequence, answer in zip(sequences, answers):
        box_match = new_format_and_acc(sequence, answer)
        scores.append(box_match)

    return scores


def acc_worker(sequence, answer, timeout=False):
    model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', sequence, flags=re.DOTALL,
                              count=1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    ext_answer = extract_answer(pred_str=model_output, data_name="math")

    if ext_answer:
        if math_equal(prediction=ext_answer, reference=answer, timeout=False):
            box_match = 1.0
        else:
            box_match = -0.5

        if "boxed" not in model_output:
            box_match = -1.0
    else:
        box_match = -1.0
    return box_match


def acc_subprocess(prediction, reference, timeout_seconds=1):
    try:
        future = global_executor.submit(acc_worker, prediction, reference)
        result = future.result(timeout=timeout_seconds)
        return result
    except TimeoutError:
        return 0.0
    except Exception as e:
        return 0.0


def accuracy_reward(sequences, answers, *args, **kwargs):
    scores = []

    for sequence, answer in zip(sequences, answers):
        box_match = acc_subprocess(sequence, answer)
        scores.append(box_match)

    return scores


def _validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    validation_passed = True

    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'boxed_start': (r'\\boxed\{.*?\}', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        if tag_name == 'boxed_start':
            match = re.findall(tag_str, processed_str)
            count = len(match)
            pos = re.search(tag_str, processed_str)
            if pos is not None:
                positions[tag_name] = re.search(tag_str, processed_str).start()
            else:
                positions[tag_name] = -1
        else:
            count = processed_str.count(tag_str)
            positions[tag_name] = processed_str.find(tag_str)

        if count != expected_count:
            validation_passed = False

    misplace_think = positions.get('think_start') > positions.get('think_end') or positions.get('think_end') > positions.get('answer_start')
    misplace_answer = positions.get('answer_start') > positions.get('boxed_start') or positions.get('boxed_start') > positions.get('answer_end')
    missing_format = not processed_str.startswith('<think>') or not processed_str.endswith('</answer>')
    if (misplace_think
            or misplace_answer or missing_format):
        validation_passed = False
    else:
        pass

    return validation_passed


def strict_format_worker(sequence, timeout=False):
    box_match = -0.5
    format_correct = _validate_response_structure(sequence)
    if format_correct:
        box_match = 1.0
    return box_match


def strict_format_subprocess(prediction, timeout_seconds=1):
    try:
        future = global_executor.submit(strict_format_worker, prediction)
        result = future.result(timeout=timeout_seconds)
        return result
    except TimeoutError:
        return 0.0
    except Exception as e:
        return 0.0


def strict_format_reward(sequences, *args, **kwargs):
    """
    Reward function that checks if the completion has a specific format.

    Args:
        sequences: A list of sequences, where each completion is a tuple containing a list of dictionaries.
                     Each dictionary should have a "content" key with the text to be checked.

    Returns:
        A list of floats, where each float is 1.0 if the corresponding completion matches the required format,
        and 0.0 otherwise.

    Raises:
        ValueError: If the input sequences are not in the expected format.
    """

    scores = []
    for completion in sequences:
        reward = strict_format_subprocess(completion)
        scores.append(reward)

    return scores


def format_reward(sequences, *args, **kwargs):
    """
    Reward function that checks if the completion has a specific format.

    Args:
        sequences: A list of sequences, where each completion is a tuple containing a list of dictionaries.
                     Each dictionary should have a "content" key with the text to be checked.

    Returns:
        A list of floats, where each float is 1.0 if the corresponding completion matches the required format,
        and 0.0 otherwise.

    Raises:
        ValueError: If the input sequences are not in the expected format.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    if not isinstance(sequences, list):
        raise ValueError("Input sequences must be a list.")

    scores = []
    for completion in sequences:
        if re.match(pattern, completion, re.DOTALL | re.MULTILINE):
            scores.append(1.0)
        else:
            scores.append(0.0)

    return scores


def reasoning_steps_reward(sequences, *args, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    matches = [len(re.findall(pattern, content)) for content in sequences]
    scores = [min(1.0, count / 3) for count in matches]

    return scores


def math_format_reward(predict_str: str) -> float:
    """
    Reward function that checks if the completion has a specific format for math questions.
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    Reward function that checks if the answer is right by `mathruler`.
    """
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def math_compute_score(predict_str: str, ground_truth: str, acc_ratio=0.9, format_ratio=0.1) -> float:
    """
    Compute score for math questions by format and accuary reward.
    """
    return acc_ratio * math_acc_reward(predict_str, ground_truth) + format_ratio * math_format_reward(predict_str)