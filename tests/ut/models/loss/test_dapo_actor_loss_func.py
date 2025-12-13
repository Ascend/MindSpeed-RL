# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from unittest.mock import Mock, patch, MagicMock
import math
import sys
import os
import pytest
import torch
import numpy as np

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from mindspeed_rl.models.loss.dapo_actor_loss_func import DAPOActorLossFunc
from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory

from tests.test_tools.dist_test import DistributedTest


class TestDAPOActorLossFunc(DistributedTest):
    world_size = 1
    is_dist_test = False

    def setup_method(self):
        # Create a DAPOActorLossFunc instance for testing
        self.loss_func = DAPOActorLossFunc()

    def test_dapo_actor_loss_func_initialization(self):
        # Test DAPOActorLossFunc initialization
        assert math.isclose(self.loss_func.clip_ratio, 0.2)
        assert math.isclose(self.loss_func.entropy_coeff, 0.0)
        assert hasattr(self.loss_func, 'logprob_computer')
        assert not hasattr(self.loss_func, 'kl_ctrl')  # Should be set via add_loss_meta_info
        assert not hasattr(self.loss_func, 'token_level_loss')  # Should be set via add_loss_meta_info
        assert not hasattr(self.loss_func, 'clip_higher_enable')  # Should be set via add_loss_meta_info
        assert not hasattr(self.loss_func, 'clip_ratio_low')  # Should be set via add_loss_meta_info
        assert not hasattr(self.loss_func, 'clip_ratio_high')  # Should be set via add_loss_meta_info

    def test_add_loss_meta_info(self):
        # Test add_loss_meta_info method with all parameters
        meta_info = {
            "clip_ratio": 0.3,
            "kl_ctrl": 0.1,
            "entropy_coeff": 0.01,
            "token_level_loss": True,
            "clip_higher_enable": True,
            "clip_ratio_low": 0.1,
            "clip_ratio_high": 0.3
        }
        self.loss_func.add_loss_meta_info(meta_info)
        
        assert math.isclose(self.loss_func.clip_ratio, 0.3)
        assert math.isclose(self.loss_func.kl_ctrl, 0.1)
        assert math.isclose(self.loss_func.entropy_coeff, 0.01)
        assert math.isclose(self.loss_func.clip_ratio_low, 0.1)
        assert math.isclose(self.loss_func.clip_ratio_high, 0.3)
        assert self.loss_func.token_level_loss is True
        assert self.loss_func.clip_higher_enable is True

    def test_add_loss_meta_info_partial(self):
        # Test add_loss_meta_info with partial parameters
        meta_info = {
            "clip_ratio": 0.3,
            "entropy_coeff": 0.01
        }
        self.loss_func.add_loss_meta_info(meta_info)
        
        assert math.isclose(self.loss_func.clip_ratio, 0.3)
        assert math.isclose(self.loss_func.entropy_coeff, 0.01)
        # Other parameters should remain unchanged
        assert not hasattr(self.loss_func, 'kl_ctrl')
        assert not hasattr(self.loss_func, 'token_level_loss')
        assert not hasattr(self.loss_func, 'clip_higher_enable')
        assert not hasattr(self.loss_func, 'clip_ratio_low')
        assert not hasattr(self.loss_func, 'clip_ratio_high')

    def test_add_loss_meta_info_none(self):
        # Test add_loss_meta_info with None
        original_clip_ratio = self.loss_func.clip_ratio
        original_entropy_coeff = self.loss_func.entropy_coeff
        self.loss_func.add_loss_meta_info(None)
        
        assert math.isclose(self.loss_func.clip_ratio, original_clip_ratio)
        assert math.isclose(self.loss_func.entropy_coeff, original_entropy_coeff)

    def test_add_loss_meta_info_empty(self):
        # Test add_loss_meta_info with empty dict
        original_clip_ratio = self.loss_func.clip_ratio
        original_entropy_coeff = self.loss_func.entropy_coeff
        self.loss_func.add_loss_meta_info({})
        
        assert math.isclose(self.loss_func.clip_ratio, original_clip_ratio)
        assert math.isclose(self.loss_func.entropy_coeff, original_entropy_coeff)

    def test_get_policy_loss_input_valid(self):
        # Test _get_policy_loss_input with valid data
        batch = {
            'responses': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'old_log_prob': torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            'advantages': torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
            'response_length': torch.tensor([2, 2])
        }
        
        response_mask, old_log_prob, advantages = self.loss_func._get_policy_loss_input(batch)
        
        assert response_mask is not None
        assert old_log_prob is not None
        assert advantages is not None
        assert response_mask.shape == (2, 3)
        assert old_log_prob.shape == (2, 3)
        assert advantages.shape == (2, 3)

    def test_get_policy_loss_input_missing_responses(self):
        # Test _get_policy_loss_input with missing responses
        batch = {
            'old_log_prob': torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            'advantages': torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
            'response_length': torch.tensor([2, 2])
        }
        
        with pytest.raises(ValueError, match="The responses is None"):
            self.loss_func._get_policy_loss_input(batch)

    def test_get_policy_loss_input_missing_optional(self):
        # Test _get_policy_loss_input with missing optional parameters
        batch = {
            'responses': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'response_length': torch.tensor([2, 2])
        }
        
        response_mask, old_log_prob, advantages = self.loss_func._get_policy_loss_input(batch)
        
        assert response_mask is not None
        assert old_log_prob is None
        assert advantages is None
        assert response_mask.shape == (2, 3)

    def test_compute_dapo_policy_loss(self):
        # Test _compute_dapo_policy_loss method
        old_log_prob = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        log_prob = torch.tensor([[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]])
        entropy = torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        advantages = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
        clip_ratio_low = 0.1
        clip_ratio_high = 0.3
        token_level_loss = False
        
        result = self.loss_func._compute_dapo_policy_loss(
            old_log_prob, log_prob, entropy, 0.01, advantages, eos_mask, 
            clip_ratio_low, clip_ratio_high, token_level_loss
        )
        
        assert len(result) == 5
        pg_loss, pg_clipfrac, ppo_kl, kl_loss, entropy_loss = result
        assert pg_loss is not None
        assert isinstance(pg_loss, torch.Tensor)
        assert pg_clipfrac is not None
        assert isinstance(pg_clipfrac, torch.Tensor)
        assert ppo_kl is not None
        assert isinstance(ppo_kl, torch.Tensor)
        assert kl_loss is not None
        assert isinstance(kl_loss, torch.Tensor)
        assert entropy_loss is not None
        assert isinstance(entropy_loss, torch.Tensor)

    def test_compute_dapo_policy_loss_token_level(self):
        # Test _compute_dapo_policy_loss with token_level_loss=True
        old_log_prob = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        log_prob = torch.tensor([[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]])
        entropy = torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        advantages = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
        clip_ratio_low = 0.1
        clip_ratio_high = 0.3
        token_level_loss = True
        
        result = self.loss_func._compute_dapo_policy_loss(
            old_log_prob, log_prob, entropy, 0.01, advantages, eos_mask, 
            clip_ratio_low, clip_ratio_high, token_level_loss
        )
        
        assert len(result) == 5
        pg_loss, pg_clipfrac, ppo_kl, kl_loss, entropy_loss = result
        assert pg_loss is not None
        assert isinstance(pg_loss, torch.Tensor)

    def test_compute_dapo_policy_loss_no_old_log_prob(self):
        # Test _compute_dapo_policy_loss with no old_log_prob
        old_log_prob = None
        log_prob = torch.tensor([[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]])
        entropy = torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        advantages = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
        clip_ratio_low = 0.1
        clip_ratio_high = 0.3
        token_level_loss = False
        
        result = self.loss_func._compute_dapo_policy_loss(
            old_log_prob, log_prob, entropy, 0.01, advantages, eos_mask, 
            clip_ratio_low, clip_ratio_high, token_level_loss
        )
        
        assert len(result) == 5
        pg_loss, pg_clipfrac, ppo_kl, kl_loss, entropy_loss = result
        assert pg_loss is not None
        assert isinstance(pg_loss, torch.Tensor)

    def test_loss_func_factory_instantiation(self):
        # Test that DAPOActorLossFunc can be instantiated through the factory
        loss_func = LossFuncFactory.get_instance('ray_dapo', 'actor')
        assert isinstance(loss_func, DAPOActorLossFunc)