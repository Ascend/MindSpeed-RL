import re
from multiprocessing import Process, Queue

import torch
from transformers import AutoTokenizer

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.math_eval_toolkit.grader import math_equal
from mindspeed_rl.utils.math_eval_toolkit.parser import extract_answer

logger = Loggers('rule_verifier')


def _math_worker(q, prediction, reference):
    result = math_equal(prediction=prediction, reference=reference, timeout=False)
    q.put(result)


def _extract_worker(q, model_output):
    result = extract_answer(pred_str=model_output, data_name="math")
    q.put(result)


def compute_verifier_score(batch, config, n_samples_per_prompt):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, trust_remote_code=True)
    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id, skip_special_tokens=True)
    question = batch["prompts"]
    responses = batch["responses"]
    str_question = tokenizer.batch_decode(question, skip_special_tokens=True)
    str_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)

    reward_index = batch["response_length"]

    logger.info("=" * 50)
    logger.info(">>>>>>>>>> User:\n")
    logger.info(str_question[0])
    logger.info(">>>>>>>>>> Assistant:\n")
    logger.info(str_responses[0])

    extra_data = {}

    if hasattr(config, "dataset_additional_keys"):
        for k in config.dataset_additional_keys:
            extra_data[k] = tokenizer.batch_decode(batch[k], skip_special_tokens=True)
            logger.info(f">>>>>>>>>> {k}")
            logger.info(extra_data[k][0])

    logger.info("=" * 50)

    scores = verifier(str_responses, extra_data, config, infos=None)

    scores = torch.tensor(
        scores,
        dtype=torch.float32,
        device=reward_index.device
    )

    scores = scores.reshape(-1, n_samples_per_prompt)
    scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-8)
    scores = scores.reshape(reward_index.shape)

    return scores


def verifier(responses, data, config, infos=None):
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
        "acc": preprocess_box_response_for_prompt,
        "format": format_reward,
        "step": reasoning_steps_reward,
        "strict_format": strict_format_reward,
        "base_acc": base_model_accuracy_reward
    }

    labels = data["labels"]
    scores = [0.0] * len(labels)

    #
    verifier_function = config.verifier_function if hasattr(
        config, "verifier_function") else ["acc"]
    verifier_weight = config.verifier_weight if hasattr(
        config, "verifier_weight") else [1.0]

    for idx, fun_verifier in enumerate(verifier_function):
        if fun_verifier not in rule_verifier_function:
            continue
        score = rule_verifier_function[fun_verifier](sequences=responses, answers=labels)
        scores = [all_score + tmp_score * verifier_weight[idx]
                  for all_score, tmp_score in zip(scores, score)]

    return scores


def math_equal_subprocess(prediction, reference, timeout_seconds=10):


    q = Queue()
    p = Process(target=_math_worker, args=(q, prediction, reference))
    p.start()

    p.join(timeout=timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    try:
        return q.get_nowait()
    except Exception as e:
        return False


def extract_answer_subprocess(model_output, timeout_seconds=10):

    q = Queue()
    p = Process(target=_extract_worker, args=(q, model_output))
    p.start()

    p.join(timeout=timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        return ""

    try:
        return q.get_nowait()
    except Exception as e:
        return ""


def preprocess_box_response_for_prompt(sequences, answers, **kwargs):
    scores = []

    for sequence, answer in zip(sequences, answers):
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', sequence, flags=re.DOTALL,
                              count=1)
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        for stop_word in stop_words:
            if stop_word in model_output:
                model_output = model_output.split(stop_word)[0].strip()
        ext_answer = extract_answer_subprocess(model_output=model_output)

        if math_equal_subprocess(prediction=ext_answer, reference=answer):
            box_match = 1.0
        else:
            box_match = -0.5

        if "boxed" not in model_output:
            box_match = -1.0

        scores.append(box_match)

    return scores


def base_model_accuracy_reward(sequences, answers, **kwargs):
    scores = []

    for sequence, answer in zip(sequences, answers):
        ext_answer = extract_answer_subprocess(model_output=sequence)

        if math_equal_subprocess(prediction=ext_answer, reference=answer):
            box_match = 1.0
        else:
            box_match = 0.0

        scores.append(box_match)

    return scores


def format_reward(sequences, **kwargs):
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

    rewards = []
    for completion in sequences:
        if re.match(pattern, completion, re.DOTALL | re.MULTILINE):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def validate_response_structure(processed_str: str) -> bool:
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


def strict_format_reward(sequences, **kwargs):
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

    rewards = []
    for completion in sequences:
        reward = -1.0
        format_correct = validate_response_structure(completion)
        if format_correct:
            reward = 1.0
        rewards.append(reward)
    return rewards


def reasoning_steps_reward(sequences, **kwargs):
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

    return [min(1.0, count / 3) for count in matches]