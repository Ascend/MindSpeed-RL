# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import unittest
from unittest.mock import MagicMock

import torch

from mindspeed_rl.utils.pad_process import (
    remove_padding_and_split_to_list,
    pad_multiple,
    truncate_middle_and_pad,
)


class TestPadProcess(unittest.TestCase):
    def test_remove_padding_and_split_to_list(self):
        responses = torch.tensor([[1, 2, 3, 4, 0, 0], [4, 5, 6, 1, 1, 0]], dtype=torch.int64)
        eos_token_id = 77
        pad_token_id = 0
        expected_output = [
            torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            torch.tensor([4, 5, 6, 1, 1], dtype=torch.int64)
        ]
        output = remove_padding_and_split_to_list(responses, eos_token_id, pad_token_id)
        self.assertEqual(len(output), len(expected_output))
        for out, exp in zip(output, expected_output):
            self.assertTrue(torch.all(out == exp))

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
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(torch.allclose(output, expected_output))

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
        
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(torch.allclose(output, expected_output))