# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from unittest.mock import MagicMock, patch
import torch

from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.models.loss.grpo_actor_loss_func import GRPOActorLossFunc
from tests.test_tools.dist_test import DistributedTest


class TestGRPOActorLossFunc(DistributedTest):

    def test_compute_loss_forward_only(self):
        batch = {'responses': torch.randn(10, 5), 'attention_mask': torch.randn(10, 5).zero_(),
                 'prompt_length': torch.randn(10, 5), 'response_length': torch.randn(10, 5)}
        log_probs = torch.randn(10, 5)
        output = torch.randn(10, 5)
        grpo_loss_func = GRPOActorLossFunc()
        with patch.object(BaseLossFunc, "compute_log_probs", return_value=log_probs):
            result = grpo_loss_func.compute_loss(output, batch, forward_only=True)
            assert torch.equal(result, log_probs)
            grpo_loss_func.compute_log_probs.assert_called_once_with(output=output, batch=batch)


    def test_compute_loss_not_forward_only(self):
        output = torch.randn(10, 5)
        batch = {'responses': torch.randn(10, 5), 'attention_mask': torch.randn(10, 5).zero_(),
                 'prompt_length': torch.randn(10, 5), 'response_length': torch.randn(10, 5)}
        log_probs = torch.randn(10, 5)
        response_mask, old_log_prob, advantages, ref_log_prob = torch.randn(10, 5), \
            torch.randn(10, 5), torch.randn(10, 5), torch.randn(10, 5)
        grpo_loss_func = GRPOActorLossFunc()
        with patch.object(BaseLossFunc, "compute_log_probs", return_value=log_probs):
            with patch.object(GRPOActorLossFunc, "_get_policy_loss_input", return_value=(response_mask, old_log_prob, advantages, ref_log_prob)):
                kl_ctrl_value = 0.1
                meta_info = {'clip_ratio': 0.2,
                             'kl_ctrl': MagicMock(return_value=kl_ctrl_value)}
                grpo_loss_func.add_loss_meta_info(meta_info)
                assert grpo_loss_func.clip_ratio == 0.2
                assert grpo_loss_func.kl_ctrl() == kl_ctrl_value
                result = grpo_loss_func.compute_loss(output, batch, forward_only=False)
                assert result[0] is not None
                grpo_loss_func.compute_log_probs.assert_called_once_with(output=output, batch=batch)
                grpo_loss_func._get_policy_loss_input.assert_called_once_with(batch=batch)
