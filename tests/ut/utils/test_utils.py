# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from unittest.mock import MagicMock
import pytest

import torch

from mindspeed_rl.utils.utils import (
    generate_mask,
    generate_position_ids,
    append_to_dict,
    num_floating_point_operations,
    get_batch_metrices_mean
)

from tests.test_tools.dist_test import DistributedTest


class TestUtils(DistributedTest):
    world_size = 1
    is_dist_test = False
    
    def test_generate_mask(self):
        data_pad = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int64)
        seq_lengths = torch.tensor([3, 2], dtype=torch.int64)
        expected_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.int64)
        output = generate_mask(data_pad, seq_lengths)
        assert output.shape == expected_mask.shape, "Acquisition of mask shape failed!"
        assert torch.all(output == expected_mask), "Acquisition of mask value failed!"

    def test_generate_position_ids(self):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        expected_output = [[0, 1, 2], [0, 1, 2]]
        output = generate_position_ids(input_ids)
        assert output == expected_output, "Acquisition of position ids value failed!"

    def test_append_to_dict(self):
        data = {'a': [1]}
        new_data = {'a': 2}
        append_to_dict(data, new_data)
        assert data == {'a': [1, 2]}, "Dict append method failed!"

    def test_num_floating_point_operations(self):
        args = MagicMock()
        args.kv_channels = 64
        args.num_attention_heads = 8
        args.hidden_size = 512
        args.seq_length = 256
        args.num_layers = 6
        args.ffn_hidden_size = 2048
        args.padded_vocab_size = 30522
        args.group_query_attention = False
        args.moe_router_topk = 2
        args.num_experts = None
        args.swiglu = False
        
        batch_size = 32

        expected_operations = 12 * 32 * 256 * 6 * 512 * 512 * (
            (1 + (8 / 8) + (256 / 512)) * (64 * 8 / 512) +
            (2048 / 512) * 1 * 1 +
            (30522 / (2 * 6 * 512))
        )
        actual_operations = num_floating_point_operations(args, batch_size)
        assert actual_operations == expected_operations, "Acquisition of num float operations failed!"

    def test_get_batch_metrices_mean(self):
        metrics_list = [
            {'loss': 0.1, 'reward': 0.8},
            {'loss': 0.2, 'reward': 0.7},
            {'loss': 0.15, 'reward': 0.75}
        ]
        expected_mean = {
            'loss': torch.tensor([0.1, 0.2, 0.15]).mean(),
            'reward': torch.tensor([0.8, 0.7, 0.75]).mean()
        }
        actual_mean = get_batch_metrices_mean(metrics_list)
        assert actual_mean['loss'].item() == expected_mean['loss'].item(), "Acquisition of metric loss mean value failed!"
        assert actual_mean['reward'].item() == expected_mean['reward'].item(), "Acquisition of metric reward mean value failed!"