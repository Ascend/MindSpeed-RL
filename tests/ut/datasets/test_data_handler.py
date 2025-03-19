# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import types

import datasets
from datasets import load_dataset

from mindspeed_rl.datasets.data_handler import build_dataset
from mindspeed_rl.datasets.handler_utils import InstructionDatasetAttr, get_handler_dataset_attr, align_dataset, \
    convert_alpaca_to_intermediate

from tests.test_tools.dist_test import DistributedTest

DATA_ORCA_RLHF_JSONL = '/data/for_dt/datasets/orca_rlhf/orca_rlhf.jsonl'


class TestHandler(DistributedTest):
    world_size = 1

    def test_build_dataset_with_non_handler(self):
        args = {
            "input": DATA_ORCA_RLHF_JSONL,
            "workers": 1,
            "streaming": False,
            "handler_name": None,
            "hf_datasets_params": None,
            "cache_dir": None,
            "dataset_additional_keys": [],
        }
        args = types.SimpleNamespace(**args)
        raw_dataset = build_dataset(args)
        assert isinstance(raw_dataset, datasets.arrow_dataset.Dataset)
        assert raw_dataset[0]['system'] == ""
        assert raw_dataset[0]['question'] is not None

    def test_get_handler_dataset_attr(self):
        args = {
            "input": DATA_ORCA_RLHF_JSONL,
            "workers": 1,
            "streaming": False,
            "handler_name": "AlpacaStylePairwiseHandler",
            "hf_datasets_params": None,
            "cache_dir": None,
            "dataset_additional_keys": [],
            "map_keys": {"prompt": "question", "query": "", "system": "system"},
            "overwrite_cache": True,
        }
        args = types.SimpleNamespace(**args)
        raw_dataset = build_dataset(args)
        dataset_attr = get_handler_dataset_attr("AlpacaStylePairwiseHandler", None, None, raw_dataset)
        assert isinstance(dataset_attr, InstructionDatasetAttr)
        assert dataset_attr.formatting == "alpaca"
        assert dataset_attr.dataset_name == "AlpacaStylePairwiseHandler"

    def test_convert_alpaca_to_intermediate(self):
        sample = {
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
        dataset_attr = InstructionDatasetAttr("file", "test", history="history")
        converted_sample = convert_alpaca_to_intermediate(sample, dataset_attr)
        assert converted_sample['prompt'] == [{'role': 'user', 'content': '回答的非常好'},
                                              {'role': 'assistant', 'content': '感谢你的认可！还有什么需要我帮助的吗？'},
                                              {'role': 'user',
                                               'content': '我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？'}]
        assert converted_sample['response'] == [{'role': 'assistant',
                                                 'content': '中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。'}]

    def test_align_dataset(self):
        args = {
            "workers": 1,
            "overwrite_cache": True,
        }
        args = types.SimpleNamespace(**args)
        data_files = [DATA_ORCA_RLHF_JSONL]
        raw_dataset = load_dataset("json", split="train", data_files=data_files, num_proc=4, )
        handler_dataset_attr = get_handler_dataset_attr("AlpacaStylePairwiseHandler",
                                                        [],
                                                        {"prompt": "question", "query": "", "system": "system"},
                                                        raw_dataset)
        aligned_dataset = align_dataset(raw_dataset, handler_dataset_attr, args)

        assert isinstance(aligned_dataset, datasets.arrow_dataset.Dataset)
        assert raw_dataset[0]["question"] == aligned_dataset[0]["prompt"][0]["content"]
        assert aligned_dataset[0]["prompt"][0]["role"] is not None
        assert isinstance(aligned_dataset[0]["tools"], list)
        assert aligned_dataset[0]["tools"][0] == ""
        assert len(raw_dataset) == len(aligned_dataset)

    def test_build_dataset_with_wrong_key(self):
        args = {
            "input": DATA_ORCA_RLHF_JSONL,
            "workers": 1,
            "streaming": False,
            "handler_name": "AlpacaStylePairwiseHandler",
            "hf_datasets_params": None,
            "cache_dir": None,
            "dataset_additional_keys": [],
            "map_keys": {"wrong_key": "question", "query": "", "system": "system"},
            "overwrite_cache": True,
        }
        args = types.SimpleNamespace(**args)
        try:
            raw_dataset = build_dataset(args)
        except Exception as e:
            assert isinstance(e, ValueError)
            assert "wrong_key is invalid, Please check map_key" in str(e)

    def test_build_dataset_with_wrong_value(self):
        args = {
            "input": DATA_ORCA_RLHF_JSONL,
            "workers": 1,
            "streaming": False,
            "handler_name": "AlpacaStylePairwiseHandler",
            "hf_datasets_params": None,
            "cache_dir": None,
            "dataset_additional_keys": [],
            "map_keys": {"prompt": "wrong_value", "query": "", "system": "system"},
            "overwrite_cache": True,
        }
        args = types.SimpleNamespace(**args)
        try:
            raw_dataset = build_dataset(args)
        except Exception as e:
            assert isinstance(e, ValueError)
            assert "wrong_value is invalid, Please check map_key" in str(e)
