# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from unittest.mock import Mock, patch, MagicMock
import math
import sys
import os
import random
import pytest
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine

from tests.test_tools.dist_test import DistributedTest


class TestBaseTrainingEngine(DistributedTest):
    world_size = 1
    is_dist_test = False

    def create_mock_training_engine(self, **kwargs):
        class MockTrainingEngine(BaseTrainingEngine):
            def post_process_forward_backward_output(self, output, batch):
                return output
        
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_scheduler = Mock()
        mock_forward_backward_func = Mock()
        
        default_kwargs = {
            'model': mock_model,
            'optimizer': mock_optimizer,
            'opt_param_scheduler': mock_scheduler,
            'forward_backward_func': mock_forward_backward_func,
            'mini_batch_size_per_dp': 2,
            'micro_batch_size': 1,
            'stage': 'test_stage',
            'role': 'test_role'
        }
        default_kwargs.update(kwargs)
        
        return MockTrainingEngine(**default_kwargs)

    def test_base_training_engine_initialization(self):
        engine = self.create_mock_training_engine()
        
        assert engine.mini_batch_size_per_dp == 2
        assert engine.micro_batch_size == 1
        assert engine.stage == 'test_stage'
        assert engine.role == 'test_role'
        assert math.isclose(engine.beta, 0.0)
        assert engine.epochs == 1
        assert engine.shuffle_mini_batch is False
        assert math.isclose(engine.kl_ctrl, 0.0)
        assert math.isclose(engine.clip_ratio, 0.1)
        assert math.isclose(engine.temperature, 1.0)
        assert math.isclose(engine.entropy_coeff, 0.0)
        assert engine.token_level_loss is False
        assert engine.clip_higher_enable is False
        assert math.isclose(engine.clip_ratio_low, 0.1)
        assert math.isclose(engine.clip_ratio_high, 0.1)
        assert math.isclose(engine.cliprange_value, 0.5)
        assert engine.use_remove_padding is False
        assert engine.use_dynamic_bsz is False
        assert engine.context_parallel_size == 1

    def test_base_training_engine_initialization_custom_params(self):
        engine = self.create_mock_training_engine(
            beta=0.1,
            mini_batch_size_per_dp=4,
            epochs=3,
            shuffle_mini_batch=True,
            stage='custom_stage',
            role='custom_role',
            kl_ctrl=0.2,
            clip_ratio=0.3,
            temperature=0.5,
            entropy_coeff=0.01,
            token_level_loss=True,
            clip_higher_enable=True,
            clip_ratio_low=0.05,
            clip_ratio_high=0.2,
            cliprange_value=0.4,
            use_remove_padding=True,
            use_dynamic_bsz=True,
            context_parallel_size=2
        )
        
        assert math.isclose(engine.beta, 0.1)
        assert engine.mini_batch_size_per_dp == 4
        assert engine.epochs == 3
        assert engine.shuffle_mini_batch is True
        assert engine.stage == 'custom_stage'
        assert engine.role == 'custom_role'
        assert math.isclose(engine.kl_ctrl, 0.2)
        assert math.isclose(engine.clip_ratio, 0.3)
        assert math.isclose(engine.temperature, 0.5)
        assert math.isclose(engine.entropy_coeff, 0.01)
        assert engine.token_level_loss is True
        assert engine.clip_higher_enable is True
        assert math.isclose(engine.clip_ratio_low, 0.05)
        assert math.isclose(engine.clip_ratio_high, 0.2)
        assert math.isclose(engine.cliprange_value, 0.4)
        assert engine.use_remove_padding is True
        assert engine.use_dynamic_bsz is True
        assert engine.context_parallel_size == 2

    def test_split_batches_tensor(self):
        engine = self.create_mock_training_engine()
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            'labels': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        }
        
        result = engine._split_batches(batch, batch_size=2, shuffle_mini_batch=False)
        
        assert len(result) == 2
        assert result[0]['input_ids'].shape == (2, 3)
        assert result[1]['input_ids'].shape == (2, 3)
        assert torch.equal(result[0]['input_ids'], torch.tensor([[1, 2, 3], [4, 5, 6]]))
        assert torch.equal(result[1]['input_ids'], torch.tensor([[7, 8, 9], [10, 11, 12]]))

    def test_split_batches_list(self):
        engine = self.create_mock_training_engine()
        
        batch = {
            'input_ids': [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), 
                         torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
        }
        
        result = engine._split_batches(batch, batch_size=2, shuffle_mini_batch=False, keep_list=True)
        
        assert len(result) == 2
        assert len(result[0]['input_ids']) == 2
        assert len(result[1]['input_ids']) == 2
        assert torch.equal(result[0]['input_ids'][0], torch.tensor([1, 2, 3]))
        assert torch.equal(result[0]['input_ids'][1], torch.tensor([4, 5, 6]))
        assert torch.equal(result[1]['input_ids'][0], torch.tensor([7, 8, 9]))
        assert torch.equal(result[1]['input_ids'][1], torch.tensor([10, 11, 12]))

    def test_split_batches_shuffle(self):
        engine = self.create_mock_training_engine()
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        }
        
        random.seed(42)
        result = engine._split_batches(batch, batch_size=2, shuffle_mini_batch=True)
        
        assert len(result) == 2

    def test_split_batches_with_dynamic_bsz(self):
        engine = self.create_mock_training_engine()
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            'prompt_length': torch.tensor([1, 1, 1, 1]),
            'response_length': torch.tensor([2, 2, 2, 2])
        }
        
        result, partitions = engine._split_batches_with_dynamic_bsz(batch, max_packing_token=10, dynamic_max_batch_size=2)
        
        assert len(result) > 0
        assert len(partitions) > 0

    def test_get_loss_meta_func(self):
        engine = self.create_mock_training_engine()
        result = engine.get_loss_meta_func()
        
        assert result == {}