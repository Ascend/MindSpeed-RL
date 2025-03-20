# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import torch

from mindspeed_rl.utils.pad_process import (
    remove_padding_and_split_to_list,
    pad_multiple,
    truncate_middle_and_pad, truncate_rows,
)

from tests.test_tools.dist_test import DistributedTest


class TestPadProcess(DistributedTest):
    world_size = 1

    def test_remove_padding_and_split_to_list(self):
        responses = torch.tensor([[1, 2, 3, 4, 0, 0], [4, 5, 6, 1, 1, 0]], dtype=torch.int64)
        eos_token_id = 77
        pad_token_id = 0
        expected_output = [
            torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            torch.tensor([4, 5, 6, 1, 1], dtype=torch.int64)
        ]
        output = remove_padding_and_split_to_list(responses, eos_token_id, pad_token_id)
        assert len(output) == len(expected_output), "After processing, the length is wrong!"
        for out, exp in zip(output, expected_output):
            assert torch.all(out == exp), "The function of remove padding failed!"

    def test_pad_multiple(self):
        data_list = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6])
        ]
        pad_id = 0
        multiple = 2
        expected_output = torch.tensor([
            [1, 2, 3, 0],
            [4, 5, 0, 0],
            [6, 0, 0, 0]
        ])
        output = pad_multiple(data_list, pad_id, multiple)
        assert output.shape == expected_output.shape, "After padding, the shape has been changed!"
        assert torch.allclose(output, expected_output), "After padding, the result has been changed!"

    def test_truncate_middle_and_pad(self):
        input_tensor = torch.tensor([
            [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]]
        ], dtype=torch.float32)

        truncate_lengths = torch.tensor([[1, 4]], dtype=torch.int64)
        pad_value = 0.0
        responses = torch.tensor([2, 3, 4], dtype=torch.int64)

        output = truncate_middle_and_pad(responses, input_tensor, truncate_lengths, pad_value)

        expected_output = torch.tensor([
            [[0.3, 0.4, 0.5], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]]
        ], dtype=torch.float32)

        assert output.shape == expected_output.shape, "After truncating, the shape has been changed!"
        assert torch.allclose(output, expected_output), "After truncating, the result has been changed!"

    def test_truncate_rows(self):
        tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        index_tensor = torch.tensor([[2], [3], [1]])
        expected_output = [torch.tensor([1, 2]), torch.tensor([5, 6, 7]), torch.tensor([9])]
        result = truncate_rows(tensor, index_tensor)
        for res, expected in zip(result, expected_output):
            assert torch.allclose(res, expected, atol=1e-5)
