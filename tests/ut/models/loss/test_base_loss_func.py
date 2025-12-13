# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import sys
import os
import pytest
import torch
import numpy as np

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc

from tests.test_tools.dist_test import DistributedTest


class TestBaseLossFunc(DistributedTest):
    world_size = 1
    is_dist_test = False

    def create_mock_loss_func(self):
        # Create a mock implementation of BaseLossFunc for testing
        class MockLossFunc(BaseLossFunc):
            def compute_loss(self, output, batch, forward_only=False, non_loss_data=True, **kwargs):
                # Mock implementation
                loss = torch.tensor(1.0)
                metrics = {"loss": loss}
                return loss, metrics
        
        return MockLossFunc()

    def test_add_loss_meta_info(self):
        # Test add_loss_meta_info method
        loss_func = self.create_mock_loss_func()
        
        # Should not raise an error
        meta_info = {"param1": 1.0, "param2": 2.0}
        loss_func.add_loss_meta_info(meta_info)

    def test_get_compute_log_probs_input_missing_responses(self):
        # Test _get_compute_log_probs_input with missing responses
        output = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
        batch = {
            'prompt_length': torch.tensor([1]),
            'response_length': torch.tensor([1])
        }
        
        with pytest.raises(ValueError, match="The responses is None"):
            BaseLossFunc._get_compute_log_probs_input(output, batch)

    def test_base_loss_func_cannot_be_instantiated_directly(self):
        # Test that BaseLossFunc cannot be instantiated directly due to abstract methods
        # This should raise a TypeError
        with pytest.raises(TypeError):
            BaseLossFunc()

    def test_base_loss_func_can_be_subclassed(self):
        # Test that BaseLossFunc can be subclassed with all abstract methods implemented
        class CompleteLossFunc(BaseLossFunc):
            def compute_loss(self, output, batch, forward_only=False, non_loss_data=True, **kwargs):
                loss = torch.tensor(1.0)
                metrics = {"loss": loss}
                return loss, metrics
        
        # This should work without raising an error
        loss_func = CompleteLossFunc()
        assert isinstance(loss_func, BaseLossFunc)
        assert isinstance(loss_func, CompleteLossFunc)

    def test_base_loss_func_method_signatures(self):
        # Test that the abstract methods have the correct signatures
        import inspect
        
        # Check compute_loss signature
        sig = inspect.signature(BaseLossFunc.compute_loss)
        assert len(sig.parameters) == 6
        assert 'output' in sig.parameters
        assert 'batch' in sig.parameters
        assert 'forward_only' in sig.parameters
        assert 'non_loss_data' in sig.parameters
        assert 'kwargs' in sig.parameters