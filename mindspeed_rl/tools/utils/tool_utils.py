import threading
import time
import traceback
import uuid
from typing import Any, Optional
import json
import requests
import torch

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.tools.utils.schemas import OpenAIFunctionParsedSchema

logger = Loggers("ToolUtils")

START_WORD = "<|im_start|>"
END_WORD = "<|im_end|>"
USER = "user"
ASSISTANT = "assistant"
TOOL_RESPONSE_START_WORD = "<tool_response>"
TOOL_RESPONSE_END_WORD = "</tool_response>"
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


def process_response(tool_map, tool_parser, hf_tokenizer, data_map, tools_kwargs):
    empty_tensor = torch.tensor([])
    if len(data_map['response_mask']) >= tools_kwargs['max_total_response_length']:
        return empty_tensor

    if data_map['tool_call_num'] >= tools_kwargs['max_tool_calls']:
        return empty_tensor

    tool_calls = tool_parser.extract_tool_calls(data_map['response_ids'])
    if not tool_calls:
        return empty_tensor

    tool_responses_list = []
    for tool_call in tool_calls[: tools_kwargs['max_parallel_calls']]:
        tool_responses_list.append(call_tool(tool_map, tool_call, tools_kwargs))
    if any(isinstance(item, Exception) for item in tool_responses_list):
        return empty_tensor

    tool_response = ''
    for item in tool_responses_list:
        tool_response += item
    tool_response_with_template = f'{START_WORD}{USER}{tool_response}{END_WORD}\n{START_WORD}{ASSISTANT}\n'
    tool_response_ids = hf_tokenizer(tool_response_with_template, add_special_tokens=False,
                                     return_attention_mask=False)['input_ids']
    if len(data_map['response_mask']) + len(tool_response_ids) >= tools_kwargs['max_total_response_length']:
        return empty_tensor

    return torch.tensor(tool_response_ids, device=torch.cuda.current_device(), dtype=torch.int32)


def call_tool(tool_map: dict[str, Any], tool_call: OpenAIFunctionParsedSchema, tools_kwargs: dict[str, Any]):
    tool, instance_id = None, None
    tool_name = tool_call.name
    tool_args = json.loads(tool_call.arguments)
    kwargs = tools_kwargs.get(tool_name, {})
    try:
        tool = tool_map[tool_name]
        instance_id = tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
        tool_execution_response, _, _ = tool.execute(instance_id, tool_args)
    except Exception as e:
        logger.error(f"Error when executing tool: {e}")
        return e
    finally:
        if tool and instance_id:
            tool.release(instance_id)

    return get_tool_response(tool_execution_response, tools_kwargs)


def get_tool_response(tool_response_text: str, kwargs: dict[str, Any]) -> str:
    max_tool_response_length = kwargs.get("max_tool_response_length", 256)
    tool_response_truncate_side = kwargs.get("tool_response_truncate_side", "")
    if tool_response_text and len(tool_response_text) > max_tool_response_length:
        if tool_response_truncate_side == "left":
            tool_response_text = tool_response_text[: max_tool_response_length] + "...(truncated)"
        elif tool_response_truncate_side == "right":
            tool_response_text = "(truncated)..." + tool_response_text[-max_tool_response_length:]
        else:
            length = max_tool_response_length // 2
            tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

    tool_response_text = f'\n{TOOL_RESPONSE_START_WORD}\n{tool_response_text}\n{TOOL_RESPONSE_END_WORD}'
    return tool_response_text


def process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
) -> tuple[int, dict[str, Any]]:
    api_response = None

    current_generation_code = generation

    if fn_name and language == "python":
        # Wrapper assumes stdin_data is a JSON string for function arguments.
        wrapper_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: "
                              f"{{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
"""
        current_generation_code = wrapper_code

    stdin = None if stdin_data is None else str(stdin_data)
    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_str = call_sandbox_api(
                    sandbox_fusion_url=sandbox_fusion_url,
                    code=current_generation_code,
                    stdin=stdin,
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language=language,
                )
        else:
            api_response, error_str = call_sandbox_api(
                sandbox_fusion_url=sandbox_fusion_url,
                code=current_generation_code,
                stdin=stdin,
                compile_timeout=timeout,
                run_timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
    except Exception as e:
        error_str = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        logger.error(f"Case {case_index + 1}: {error_str}")
        traceback.print_exc()

    metadata = {
        "case_index": case_index,
        "input": stdin,
        "expected_output": str(expected_output) if expected_output else None,
        "api_request_error": error_str,
        "api_response": None,
        "status": "success",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
    }
    result_status = True

    if error_str:
        metadata["status"] = "api_error"
        result_status = -1
        logger.error(f"Case {case_index}: API error occurred: {error_str}")
        generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
        logger.error(f"Case {case_index}: code: {generation_to_log}")
        logger.error(f"Case {case_index}: input: {stdin}")
    elif api_response:
        try:
            metadata["run_status"] = api_response["run_result"]["status"]
            metadata["stdout"] = api_response["run_result"]["stdout"]
            metadata["stderr"] = api_response["run_result"]["stderr"]
        except Exception as e:
            logger.error(f"Case {case_index}: API Response parse error: {e}, API Response: {api_response}")

    else:
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
    return result_status, metadata


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
    max_retries: int = 3,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        code: The code string to execute.
        stdin: The standard input string.
        compile_timeout: Compile timeout in seconds.
        run_timeout: Run timeout in seconds.
        language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Request ID: {request_id}] "

    if language not in SUPPORTED_LANGUAGES:
        error_str = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_str)
        return None, error_str

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,
            "files": {},
            "fetch_files": [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,
            )

            if response.status_code == 504:
                last_error = (
                    f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt "
                    f"{attempt + 1}/{max_retries}"
                )
                logger.warning(last_error)
                if attempt < max_retries - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            response.raise_for_status()
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    if last_error:
        logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def process_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = 30,
) -> tuple[str, dict[str, Any]]:
    logger.info(f"Starting batch search for {len(query_list)} queries.")
    api_response = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_str = call_search_api(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    topk=topk,
                    return_scores=True,
                    timeout=timeout,
                )
        else:
            api_response, error_str = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_str = f"API Request Exception during batch search: {e}"
        logger.error(f"Batch search: {error_str}")
        traceback.print_exc()

    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": error_str,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }

    if error_str:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Search error: {error_str}"})
        logger.error(f"Batch search: API error occurred: {error_str}")
    elif api_response:
        try:
            raw_results = api_response.get("result", [])
            if raw_results:
                pretty_results = []
                total_results = 0

                for retrieval in raw_results:
                    formatted = passages2string(retrieval)
                    pretty_results.append(formatted)
                    total_results += len(retrieval) if isinstance(retrieval, list) else 1

                final_result = "\n---\n".join(pretty_results)
                result_text = json.dumps({"result": final_result})
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
                logger.info(f"Batch search: Successful, got {total_results} total results")
            else:
                result_text = json.dumps({"result": "No search results found."})
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                logger.info("Batch search: No results found")
        except Exception as e:
            error_str = f"Error processing search results: {e}"
            result_text = json.dumps({"result": error_str})
            metadata["status"] = "processing_error"
            logger.error(f"Batch search: {error_str}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps({"result": "Unknown API state (no response and no error message)."})
        logger.error("Batch search: Unknown API state.")

    return result_text, metadata


def passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = 30,
    max_retries: int = 10,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Calls the remote search API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "
    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{max_retries}: Calling search API at {retrieval_service_url}"
            )
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{max_retries}"
                )
                logger.warning(last_error)
                if attempt < max_retries - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            response.raise_for_status()
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < max_retries - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < max_retries - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break

    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"
