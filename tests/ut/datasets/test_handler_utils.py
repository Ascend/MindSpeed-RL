# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import bisect
from unittest.mock import Mock, patch
from functools import partial
import pytest

from mindspeed_rl.datasets.handler_utils import (
    FILEEXT2TYPE,
    InstructionDatasetAttr,
    convert_token_to_id,
    search_for_fit,
    greedy_knapsack,
    check_dataset_info_map,
    get_handler_dataset_attr,
    align_dataset,
    convert_math17k_to_intermediate,
    convert_alpaca_to_intermediate
)
from mindspeed_rl.datasets.templates import Role

from tests.test_tools.dist_test import DistributedTest


class TestHandlerUtils(DistributedTest):
    world_size = 1
    is_dist_test = False

    def test_file_ext_to_type(self):
        assert FILEEXT2TYPE["arrow"] == "arrow"
        assert FILEEXT2TYPE["csv"] == "csv"
        assert FILEEXT2TYPE["json"] == "json"
        assert FILEEXT2TYPE["jsonl"] == "json"
        assert FILEEXT2TYPE["parquet"] == "parquet"
        assert FILEEXT2TYPE["txt"] == "text"

    def test_instruction_dataset_attr(self):
        attr = InstructionDatasetAttr(
            load_from="file",
            dataset_name="test_dataset",
            subset="test_subset",
            folder="test_folder",
            ranking=True,
            formatting="alpaca",
            system="system_col",
            images="images_col",
            prompt="prompt_col",
            query="query_col",
            response="response_col",
            history="history_col",
            chosen="chosen_col",
            rejected="rejected_col"
        )
        
        assert attr.load_from == "file"
        assert attr.dataset_name == "test_dataset"
        assert attr.subset == "test_subset"
        assert attr.folder == "test_folder"
        assert attr.ranking is True
        assert attr.formatting == "alpaca"
        assert attr.system == "system_col"
        assert attr.images == "images_col"
        assert attr.prompt == "prompt_col"
        assert attr.query == "query_col"
        assert attr.response == "response_col"
        assert attr.history == "history_col"
        assert attr.chosen == "chosen_col"
        assert attr.rejected == "rejected_col"

        attr.set_attr("prompt", {"prompt": "new_prompt_col"}, "default_prompt")
        assert attr.prompt == "new_prompt_col"

    def test_convert_token_to_id(self):
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [123]
        
        result = convert_token_to_id("test_token", mock_tokenizer)
        assert result == 123
        mock_tokenizer.encode.assert_called_once_with("test_token", add_special_tokens=False)
        
        with pytest.raises(ValueError):
            convert_token_to_id(123, mock_tokenizer)
        
        mock_tokenizer.encode.return_value = [123, 456]
        with pytest.raises(ValueError):
            convert_token_to_id("test_token", mock_tokenizer)

    def test_search_for_fit(self):
        assert search_for_fit([], 10) == -1
        
        assert search_for_fit([5], 10) == 0
        
        assert search_for_fit([15], 10) == -1
        
        numbers = [1, 3, 5, 7, 9]
        assert search_for_fit(numbers, 6) == 2
        assert search_for_fit(numbers, 10) == 4
        assert search_for_fit(numbers, 0) == -1

    def test_greedy_knapsack(self):
        assert greedy_knapsack([], 10) == []
        
        assert greedy_knapsack([5], 10) == [[5]]
        
        numbers = [2, 3, 4, 5, 6]
        result = greedy_knapsack(numbers, 10)
        
        all_numbers = [
            num
            for knapsack in result
            for num in knapsack
        ]
        assert sorted(all_numbers) == [2, 3, 4, 5, 6]
        
        for knapsack in result:
            assert sum(knapsack) <= 10
        
        numbers = [5, 5, 5, 5]
        result = greedy_knapsack(numbers, 10)
        assert len(result) == 2
        for knapsack in result:
            assert sum(knapsack) == 10
        
        numbers = [3, 3, 4, 4, 5, 5]
        result = greedy_knapsack(numbers, 10)
        all_numbers = [
            num
            for knapsack in result
            for num in knapsack
        ]
        assert sorted(all_numbers) == [3, 3, 4, 4, 5, 5]
        for knapsack in result:
            assert sum(knapsack) <= 10

    def test_check_dataset_info_map(self):
        map_keys = {"prompt": "instruction", "query": "input"}
        column_names = ["prompt", "query", "response"]
        raw_datasets = Mock()
        raw_datasets.format = {"columns": ["instruction", "input", "output"]}
        
        check_dataset_info_map(map_keys, "AlpacaStyleHandler", column_names, raw_datasets)
        
        map_keys = {"invalid": "instruction"}
        with pytest.raises(ValueError):
            check_dataset_info_map(map_keys, "AlpacaStyleHandler", column_names, raw_datasets)
        
        map_keys = {"prompt": "invalid_column"}
        with pytest.raises(ValueError):
            check_dataset_info_map(map_keys, "AlpacaStyleHandler", column_names, raw_datasets)

    def test_get_handler_dataset_attr(self):
        raw_datasets = Mock()
        raw_datasets.format = {"columns": ["instruction", "input", "output"]}
        
        dataset_attr = get_handler_dataset_attr(
            "AlpacaStyleHandler",
            "key1,key2",
            {"prompt": "instruction", "query": "input", "response": "output"},
            raw_datasets
        )
        
        assert dataset_attr.dataset_name == "AlpacaStyleHandler"
        assert dataset_attr.dataset_additional_keys == "key1,key2"
        assert dataset_attr.formatting == "alpaca"
        assert dataset_attr.prompt == "instruction"
        assert dataset_attr.query == "input"
        assert dataset_attr.response == "output"
        assert dataset_attr.ranking is False
        
        dataset_attr = get_handler_dataset_attr(
            "PairwiseAlpacaStyleHandler",
            "key1,key2",
            {"prompt": "instruction", "query": "input", "response": "output"},
            raw_datasets
        )
        
        assert dataset_attr.ranking is True
        
        dataset_attr = get_handler_dataset_attr(
            "Math17kAlpacaStyleHandler",
            "key1,key2",
            {"prompt": "instruction", "query": "input", "response": "output"},
            raw_datasets
        )
        
        assert dataset_attr.formatting == "math17k_alpaca"

    def test_convert_math17k_to_intermediate(self):
        dataset_attr = Mock()
        dataset_attr.history = "history"
        dataset_attr.prompt = "instruction"
        dataset_attr.query = "input"
        dataset_attr.response = "output"
        dataset_attr.system = "system"
        dataset_attr.ranking = False
        dataset_attr.dataset_additional_keys = ["key1", "key2"]
        
        sample = {
            "history": [
                ["old_prompt", "old_response"]
            ],
            "instruction": [{"content": "test instruction"}],
            "input": "",
            "output": "test response",
            "system": "test system",
            "key1": "value1",
            "key2": "value2"
        }
        
        result = convert_math17k_to_intermediate(sample, dataset_attr)
        
        assert len(result["prompt"]) == 3
        assert result["prompt"][0]["role"] == Role.USER.value
        assert result["prompt"][0]["content"] == "old_prompt"
        assert result["prompt"][1]["role"] == Role.ASSISTANT.value
        assert result["prompt"][1]["content"] == "old_response"
        assert result["prompt"][2]["role"] == Role.USER.value
        assert result["prompt"][2]["content"] == "test instruction"
        
        assert len(result["response"]) == 1
        assert result["response"][0]["role"] == Role.ASSISTANT.value
        assert result["response"][0]["content"] == "test response"
        
        assert result["system"] == ["test system"]
        assert result["tools"] == [""]
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_convert_alpaca_to_intermediate(self):
        dataset_attr = Mock()
        dataset_attr.history = "history"
        dataset_attr.prompt = "instruction"
        dataset_attr.query = "input"
        dataset_attr.response = "output"
        dataset_attr.system = "system"
        dataset_attr.ranking = False
        dataset_attr.dataset_additional_keys = ["key1", "key2"]
        
        sample = {
            "history": [
                ["old_prompt", "old_response"]
            ],
            "instruction": "test instruction",
            "input": "test input",
            "output": "test response",
            "system": "test system",
            "key1": "value1",
            "key2": "value2"
        }
        
        result = convert_alpaca_to_intermediate(sample, dataset_attr)
        
        assert len(result["prompt"]) == 3
        assert result["prompt"][0]["role"] == Role.USER.value
        assert result["prompt"][0]["content"] == "old_prompt"
        assert result["prompt"][1]["role"] == Role.ASSISTANT.value
        assert result["prompt"][1]["content"] == "old_response"
        assert result["prompt"][2]["role"] == Role.USER.value
        assert result["prompt"][2]["content"] == "test instruction\ntest input"
        
        assert len(result["response"]) == 1
        assert result["response"][0]["role"] == Role.ASSISTANT.value
        assert result["response"][0]["content"] == "test response"
        
        assert result["system"] == ["test system"]
        assert result["tools"] == [""]
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_convert_alpaca_to_intermediate_ranking(self):
        dataset_attr = Mock()
        dataset_attr.history = None
        dataset_attr.prompt = "instruction"
        dataset_attr.query = "input"
        dataset_attr.response = None
        dataset_attr.system = "system"
        dataset_attr.ranking = True
        dataset_attr.chosen = "chosen"
        dataset_attr.rejected = "rejected"
        dataset_attr.dataset_additional_keys = []
        
        sample = {
            "instruction": "test instruction",
            "input": "test input",
            "chosen": "chosen response",
            "rejected": "rejected response",
            "system": "test system"
        }
        
        result = convert_alpaca_to_intermediate(sample, dataset_attr)
        
        assert len(result["prompt"]) == 1
        assert result["prompt"][0]["role"] == Role.USER.value
        assert result["prompt"][0]["content"] == "test instruction\ntest input"
        
        assert len(result["response"]) == 2
        assert result["response"][0]["role"] == Role.ASSISTANT.value
        assert result["response"][0]["content"] == "chosen response"
        assert result["response"][1]["role"] == Role.ASSISTANT.value
        assert result["response"][1]["content"] == "rejected response"
