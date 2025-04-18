import asyncio
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import math
import random
from .evaluation_utils.code_util import evaluate_code
from .evaluation_utils.math_util import evaluate_math
import time
import ray
from tqdm.asyncio import tqdm


def validate_response_structure(processed_str: str, task: str) -> bool:
    """只检查是否有</think>标签且标签前有内容"""
    
    # print("\n[Structure Validation]")
    
    # 检查</think>标签是否存在
    think_end_pos = processed_str.find('</think>')
    
    # 验证标签存在且前面有内容
    validation_passed = think_end_pos > 0
    
    # print(f"  </think>: position={think_end_pos}")
    # print(f"  Validation: {'PASS' if validation_passed else 'FAIL'}")
    
    return validation_passed


def process_completion(completion, task, reference):
    if task == "code":
        return evaluate_code(completion, reference)
    elif task == "math":
        return evaluate_math(completion, str(reference))
    else:
        print('task')
        raise NotImplementedError


def get_format_score(validation_passed):
    format_reward = 1
    format_reward if validation_passed else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if validation_passed else 'FAIL'}")
    print(f"  Format score: {format_reward}")
    return format_reward
    


async def process_row_with_timeout(completion, reference, task, executor, timeout=300.0):
    """
    Process a single row with a timeout.
    """
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(process_completion, completion, task, reference)  # Ensure synchronous
            ),
            timeout=timeout
        )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows

async def parallel_evaluate_continual_async(completions, references, tasks, num_processes, task_timeout=300.0):
    """
    Evaluate rows in parallel with a process pool and timeout handling.
    """
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            process_row_with_timeout(completion, reference, task, executor, timeout=task_timeout)
            for completion, reference, task in zip(completions, references, tasks)
        ]
        # Use tqdm for progress tracking
        results = await tqdm.gather(*tasks_async, disable=True)

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        validation_passed = validate_response_structure(completion, task)

        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            # scores.append(0.0)
            score = 0
            continue
        print(f"Test for result {result}")
        try:
            # Process result based on task type
            if task == 'code' and not result[0][0]: # if task is code, the reference should be json string
                correct = 0
                error = 0
                total = min(
                    len(json.loads(reference)['inputs'] if not isinstance(reference, dict) else reference['inputs']),
                    10)
                for run in result[0][1]:
                    if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[True]':
                        correct += 1
                    if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[-2]':
                        error += 1
                score = correct / total
                # scores.append(correct / total) # 添加小数奖励
            else:
                score = float(int(result[0][0]))
                # scores.append(float(int(result[0][0])))
        except Exception as e:
            print(f"Error processing result for row: {completion[:10]}, Error: {e}")
            # scores.append(0.0)
            score = 0

        # answer_score = 0
        # # 添加整数奖励
        # if score>0:
        #     if score==1:
        #         answer_score = 2 
        #     else:
        #         answer_score = -1.5
        # else:
        #     answer_score = -2
        # format score
        validation_passed = validate_response_structure(completion, task)
        # format_score = get_format_score(validation_passed)
        format_score = 1
        # 格式优先
        if validation_passed == False:
            format_score = -1
            total_score = format_score
        else:
            total_score = format_score + score
        # print(" completion ".center(80, '=')+"\n\n" + str(completion) + "\n\n"
        #     + " validate_response_structure ".center(80, '=')+"\n\n" + str(validation_passed) + "\n\n"
        #     + " format_score ".center(80, '=')+"\n\n" + str(format_score) + "\n\n"
        #     + " total_score ".center(80, '=')+"\n\n" + str(total_score) + "\n\n"
        #     + " score ".center(80, '=')+"\n\n" + str(score) + "\n\n"
        #     )

        scores.append(total_score)
    return scores

@ray.remote
def process_row(completion, reference, task, timeout=300.0):
    """
    Process a single row synchronously.
    """
    try:
        result = process_completion(completion, task, reference)
        return [result]
    except Exception as e:
        print(f"Error processing completion in Process_Row function: {completion[:10]}, Error: {e}")
        return None

def sequential_evaluate_continual(completions, references, tasks):
    """
    Evaluate rows sequentially without concurrency.
    """
    scores = []
    futures = []
    for completion, reference, task in zip(completions, references, tasks):
        futures.append(process_row.remote(completion, reference, task ))
    results = ray.get(futures)
    for completion, reference, task, result in zip(completions, references, tasks, results):
        validation_passed = validate_response_structure(completion, task)

        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            # scores.append(0.0)
            score = 0
            continue
        else:
            try:
                if task == 'code' and not result[0][0]:
                    correct = 0
                    error = 0
                    total = min(
                        len(json.loads(reference)['inputs'] if not isinstance(reference, dict) else reference['inputs']),
                        10)
                    for run in result[0][1]:
                        if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[True]':
                            correct += 1
                        if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[-2]':
                            error += 1
                    score = correct / total
                else:
                    score = float(int(result[0][0]))
            except Exception as e:
                print(f"Error processing result for row in sequential function: {completion[:10]}, Error: {e}")
                print(result)
                score = 0

        format_score = 0.2 if validation_passed else 0
        total_score = format_score + score
        scores.append(total_score)
    
    return scores




def compute_score(queue, sequences, answers, tasks, *kwargs):
    do_print = True
    # completions = [completions]
    # references = [references]
    # tasks = [tasks]
    if do_print:
        print("completions:", sequences[0])
        print("references:", answers[0])
        print("tasks:", tasks[0])
    # three lists should have identical length
    # TODO: make this one completely asynchronous, which means the main process can do other things(e.g., forwarding reward model) while computing score
    assert len(sequences) == len(answers) == len(tasks)
    try:
        res = sequential_evaluate_continual(sequences, answers, tasks)
        print("res:", res[0])
        # return res
    except asyncio.TimeoutError as e:
        print('Global timeout in reward computing! Setting all as 0.5.')
        res = [0.02 for _ in range(len(sequences))]
    except Exception as e:
        print(f"Unexpected error: {e}")
        res = [0.01 for _ in range(len(sequences))]
    
    if queue is not None:
        queue.put(res)

    return res