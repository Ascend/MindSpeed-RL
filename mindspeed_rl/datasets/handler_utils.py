# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os
import bisect
import json
from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Literal, Optional

from mindspeed_rl.datasets.templates import Role
from mindspeed_rl.utils.loggers import Loggers

logger = Loggers(name="handler_utils")

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


@dataclass
class InstructionDatasetAttr:
    r"""
    Dataset attributes.
    """

    """ basic configs """
    load_from: Literal["file"]
    dataset_name: str
    """ extra configs """
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: bool = False
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    """ columns """
    system: Optional[str] = None
    images: Optional[str] = None
    """ columns for the alpaca format """
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

    """ columns for the pairwise dataset """
    chosen: Optional[str] = "chosen"
    rejected: Optional[str] = "rejected"

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))


def get_dataset_list(data_input, dataset_dir, interleave_probs) -> List["InstructionDatasetAttr"]:
    """
    Map multiple dataset attributes to List["InstructionDatasetAttr"]
    through parameters and the data.json mapping file.
    """
    if data_input is not None:
        dataset_names = [ds.split("/")[-1].strip() for ds in data_input.split(",")]
    else:
        dataset_names = []

    try:
        with open(os.path.join(dataset_dir, "dataset_info.json"), "r") as f:
            dataset_info = json.load(f)
    except Exception as err:
        if len(dataset_names) != 0:
            raise ValueError(
                "Cannot open {} due to {}.".format(os.path.join(dataset_dir, "dataset_info.json"), str(err))
            ) from err
        dataset_info = None

    if dataset_info is None:
        raise ValueError(
            "Cannot load {}.".format(os.path.join(dataset_dir, "dataset_info.json"))
        )

    # Multiple Dataset Interleaving Probability
    if interleave_probs is not None:
        interleave_probs = [float(prob.strip()) for prob in interleave_probs.split(",")]

    dataset_list: List[InstructionDatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, "dataset_info.json"))

        if "script_url" in dataset_info[name]:
            dataset_attr = InstructionDatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        else:
            dataset_attr = InstructionDatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")

        if "columns" in dataset_info[name]:
            column_names = ["system", "images"]
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history"])
            else:
                column_names.extend(["messages", "tools"])

            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        dataset_list.append(dataset_attr)

    return dataset_list


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        if len(token) != 1:
            raise ValueError("token length shoule be 1.")
        return token[0]
    else:
        raise ValueError("token should be int or str")


def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: List[int], capacity: int) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def check_dataset_info_map(map_keys, handler_name, column_names, raw_datasets):
    if len(map_keys.keys()) > len(column_names):
        raise ValueError("Please check map_keys")

    for key in map_keys.keys():
        if key not in column_names:
            raise ValueError(f' {key} is unvalid, Please check map_keys')

    if "AlpacaStyle" in handler_name:
        for value in map_keys.values():
            if value and value not in raw_datasets.format['columns']:
                raise ValueError(f' {value} is unvalid, Please check map_keys')


def get_handler_dataset_attr(handler_name, dataset_additional_keys, map_keys, raw_datasets):
    dataset_attr = InstructionDatasetAttr("file", dataset_name=handler_name)
    dataset_attr.dataset_additional_keys = dataset_additional_keys

    if "Pairwise" in handler_name:
        setattr(dataset_attr, "ranking", True)

    if "AlpacaStyle" in handler_name:
        dataset_attr.formatting = "alpaca"
        column_names = ["prompt", "query", "response", "history", "system", "chosen", "rejected"]
        if map_keys is not None:
            check_dataset_info_map(map_keys, handler_name, column_names, raw_datasets)
            for column_name, target_name in map_keys.items():
                setattr(dataset_attr, column_name, target_name)

    return dataset_attr


def convert_alpaca_to_intermediate(sample: Dict[str, List[Any]], dataset_attr: "InstructionDatasetAttr"):
    """
    format sample info
    {
      "instruction": "我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？",
      "input": "",
      "output": "中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。",
      "history": [
       [
        "回答的非常好",
        "感谢你的认可！还有什么需要我帮助的吗？"
       ]
      ]
     }
    ---->>>>
    {
        'prompt': [{'role': 'user', 'content': '回答的非常好'},
                {'role': 'assistant', 'content': '感谢你的认可！还有什么需要我帮助的吗？'},
                {'role': 'user', 'content': '我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？'}],
        'response': [{'role': 'assistant', 'content': '中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。'}],
        'system': [''],
        'tools': ['']
    }
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []

    if dataset_attr.history and (
            isinstance(sample[dataset_attr.history], list) or isinstance(sample[dataset_attr.history], dict)):
        for old_prompt, old_response in sample[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    content = []
    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        content.append(sample[dataset_attr.prompt])

    if dataset_attr.query and sample[dataset_attr.query]:
        content.append(sample[dataset_attr.query])

    prompt.append({"role": Role.USER.value, "content": "\n".join(content)})

    if dataset_attr.ranking:
        if dataset_attr.chosen and isinstance(sample[dataset_attr.chosen], list):
            response = [
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.chosen][0]},
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.rejected][1]},
            ]
        elif dataset_attr.chosen and isinstance(sample[dataset_attr.chosen], str):
            response = [
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": sample[dataset_attr.rejected]},
            ]
        else:
            response = []
    else:
        if dataset_attr.response and isinstance(sample[dataset_attr.response], list):
            response = [
                {"role": Role.ASSISTANT.value, "content": content} for content in sample[dataset_attr.response]
            ]
        elif dataset_attr.response and isinstance(sample[dataset_attr.response], str):
            response = [{"role": Role.ASSISTANT.value, "content": sample[dataset_attr.response]}]
        else:
            response = []

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
    outputs["tools"].append("")

    for add_key in dataset_attr.dataset_additional_keys:
        if add_key != "labels":
            outputs[add_key] = sample[add_key]

    return outputs
