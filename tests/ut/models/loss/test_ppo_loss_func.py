# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from unittest.mock import MagicMock, patch
from unittest import TestCase

import torch
import pytest

from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.models.loss.ppo_actor_loss_func import PPOActorLossFunc
from tests.test_tools.dist_test import DistributedTest


class TestPPOActorLossFunc(DistributedTest):

    def test_get_policy_loss_input_value_error(self):
        batch = {'attention_mask': torch.randn(10, 5).zero_(),
                 'prompt_length': torch.randn(10, 5), 'response_length': torch.randn(10, 5)}
        ppo_loss_func = PPOActorLossFunc()
        with pytest.raises(ValueError):
            ppo_loss_func._get_policy_loss_input(batch)


    def test_compute_loss_forward_only(self):
        batch = {'responses': torch.randn(10, 5), 'attention_mask': torch.randn(10, 5).zero_(),
                 'prompt_length': torch.randn(10, 5), 'response_length': torch.randn(10, 5)}
        log_probs = torch.randn(10, 5)
        output = torch.randn(10, 5)
        ppo_loss_func = PPOActorLossFunc()
        with patch.object(BaseLossFunc, "compute_log_probs", return_value=log_probs):
            result = ppo_loss_func.compute_loss(output, batch, forward_only=True)
            assert torch.equal(result, log_probs)
            ppo_loss_func.compute_log_probs.assert_called_once_with(output=output, batch=batch)


    def test_compute_loss_not_forward_only(self):
        output = torch.randn(10, 5)
        batch = {'responses': torch.randn(10, 5), 'attention_mask': torch.randn(10, 5).zero_(),
                 'prompt_length': torch.randn(10, 5), 'response_length': torch.randn(10, 5)}
        log_probs = torch.randn(10, 5)
        entropy = torch.randn(10, 5)
        response_mask, old_log_prob, advantages, ref_log_prob = torch.randn(10, 5), \
            torch.randn(10, 5), torch.randn(10, 5), torch.randn(10, 5)
        ppo_loss_func = PPOActorLossFunc()
        with patch.object(BaseLossFunc, "compute_log_probs", return_value=(log_probs, entropy)):
            with patch.object(PPOActorLossFunc, "_get_policy_loss_input", return_value=(response_mask, old_log_prob, advantages, ref_log_prob)):
                kl_ctrl_value = 0.1
                meta_info = {'clip_ratio': 0.5,
                             'kl_ctrl': MagicMock(return_value=kl_ctrl_value),
                             'entropy_coeff': 0.0}
                ppo_loss_func.add_loss_meta_info(meta_info)
                assert ppo_loss_func.clip_ratio == 0.5
                assert ppo_loss_func.kl_ctrl() == kl_ctrl_value
                result = ppo_loss_func.compute_loss(output, batch, forward_only=False)
                assert result[0] is not None
                ppo_loss_func.compute_log_probs.assert_called_once_with(output=output, batch=batch, update=True)
                ppo_loss_func._get_policy_loss_input.assert_called_once_with(batch=batch)
