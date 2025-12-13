# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from unittest.mock import Mock, patch, MagicMock
import sys
import os
import pytest
import torch
import numpy as np

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from mindspeed_rl.models.loss.critic_loss_func import CriticLossFunc
from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory

from tests.test_tools.dist_test import DistributedTest


class TestCriticLossFunc(DistributedTest):
    world_size = 1
    is_dist_test = False

    def setup_method(self):
        # Create a CriticLossFunc instance for testing
        self.loss_func = CriticLossFunc()

    def test_critic_loss_func_initialization(self):
        # Test CriticLossFunc initialization
        assert self.loss_func.cliprange_value == 0.5
        assert hasattr(self.loss_func, 'logprob_computer')

    def test_add_loss_meta_info(self):
        # Test add_loss_meta_info method
        meta_info = {"cliprange_value": 0.3}
        self.loss_func.add_loss_meta_info(meta_info)
        assert self.loss_func.cliprange_value == 0.3

    def test_add_loss_meta_info_none(self):
        # Test add_loss_meta_info with None
        original_value = self.loss_func.cliprange_value
        self.loss_func.add_loss_meta_info(None)
        assert self.loss_func.cliprange_value == original_value

    def test_add_loss_meta_info_empty(self):
        # Test add_loss_meta_info with empty dict
        original_value = self.loss_func.cliprange_value
        self.loss_func.add_loss_meta_info({})
        assert self.loss_func.cliprange_value == original_value

    def test_get_policy_loss_input_valid(self):
        # Test _get_policy_loss_input with valid data
        batch = {
            'responses': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'values': torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            'returns': torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
            'response_length': torch.tensor([2, 2])
        }
        
        response_mask, values, returns, response_length = self.loss_func._get_policy_loss_input(batch)
        
        assert response_mask is not None
        assert values is not None
        assert returns is not None
        assert response_length is not None
        assert response_mask.shape == (2, 3)
        assert values.shape == (2, 3)
        assert returns.shape == (2, 3)
        assert response_length.shape == (2,)

    def test_get_policy_loss_input_missing_responses(self):
        # Test _get_policy_loss_input with missing responses
        batch = {
            'values': torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            'returns': torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
            'response_length': torch.tensor([2, 2])
        }
        
        with pytest.raises(ValueError, match="The responses is None"):
            self.loss_func._get_policy_loss_input(batch)

    def test_get_policy_loss_input_missing_values_returns(self):
        # Test _get_policy_loss_input with missing values and returns
        batch = {
            'responses': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'response_length': torch.tensor([2, 2])
        }
        
        response_mask, values, returns, response_length = self.loss_func._get_policy_loss_input(batch)
        
        assert response_mask is not None
        assert values is None
        assert returns is None
        assert response_length is not None

    def test_get_compute_vpreds_missing_responses(self):
        # Test _get_compute_vpreds with missing responses
        output = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
        batch = {
            'prompt_length': torch.tensor([1]),
            'response_length': torch.tensor([1])
        }
        
        with pytest.raises(ValueError, match="The responses is None"):
            self.loss_func._get_compute_vpreds(output, batch)

    def test_compute_value_loss(self):
        # Test _compute_value_loss method
        vpreds = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        returns = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        values = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        response_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
        cliprange_value = 0.5
        
        vf_loss, vf_clipfrac = self.loss_func._compute_value_loss(
            vpreds, returns, values, response_mask, cliprange_value
        )
        
        assert vf_loss is not None
        assert isinstance(vf_loss, torch.Tensor)
        assert vf_clipfrac is not None
        assert isinstance(vf_clipfrac, torch.Tensor)

    def test_compute_value_loss_no_clipping(self):
        # Test _compute_value_loss with no clipping
        vpreds = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        returns = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        values = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        response_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
        cliprange_value = 0.5
        
        vf_loss, vf_clipfrac = self.loss_func._compute_value_loss(
            vpreds, returns, values, response_mask, cliprange_value
        )
        
        assert vf_loss is not None
        assert vf_clipfrac is not None
        # vf_clipfrac should be 0 since no values are clipped
        assert vf_clipfrac.item() == 0


    def test_loss_func_factory_instantiation(self):
        # Test that CriticLossFunc can be instantiated through the factory
        loss_func = LossFuncFactory.get_instance('ray_ppo', 'critic')
        assert isinstance(loss_func, CriticLossFunc)