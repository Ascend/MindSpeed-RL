# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from unittest.mock import MagicMock, patch
import torch

from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.models.loss.dapo_actor_loss_func import DAPOActorLossFunc
from tests.test_tools.dist_test import DistributedTest


class TestDAPOActorLossFunc(DistributedTest):
    is_dist_test = False

    def test_compute_loss_forward_only(self):
        batch = {'responses': torch.randn(10, 5), 'attention_mask': torch.randn(10, 5).zero_(),
                 'prompt_length': torch.randn(10, 5), 'response_length': torch.randn(10, 5)}
        log_probs = torch.randn(10, 5)
        output = torch.randn(10, 5)
        dapo_loss_func = DAPOActorLossFunc()
        with patch.object(BaseLossFunc, "compute_log_probs", return_value=(log_probs, None)):
            result = dapo_loss_func.compute_loss(output, batch, forward_only=True)
            assert torch.equal(result, log_probs)
            dapo_loss_func.compute_log_probs.assert_called_once_with(output=output, batch=batch)


    def test_compute_loss_not_forward_only(self):
        output = torch.randn(10, 5)
        batch = {'responses': torch.randn(10, 5), 'attention_mask': torch.randn(10, 5).zero_(),
                 'prompt_length': torch.randn(10, 5), 'response_length': torch.randn(10, 5)}
        log_probs = torch.randn(10, 5)
        entropy = torch.randn(10, 5)
        response_mask, old_log_prob, advantages, ref_log_prob = torch.randn(10, 5), \
            torch.randn(10, 5), torch.randn(10, 5), torch.randn(10, 5)
        dapo_loss_func = DAPOActorLossFunc()
        with patch.object(BaseLossFunc, "compute_log_probs", return_value=(log_probs, entropy)):
            with patch.object(DAPOActorLossFunc, "_get_policy_loss_input", return_value=(response_mask, old_log_prob, advantages)):
                kl_ctrl_value = 0.1
                meta_info = {'clip_ratio': 0.2,
                             'kl_ctrl': MagicMock(return_value=kl_ctrl_value),
                             'entropy_coeff': 0.0,
                             'clip_higher_enable': True,
                             'clip_ratio_low': 0.1,
                             'clip_ratio_high': 0.1,
                             'token_level_loss': 0.2}
                dapo_loss_func.add_loss_meta_info(meta_info)
                assert dapo_loss_func.clip_ratio == 0.2
                assert dapo_loss_func.kl_ctrl() == kl_ctrl_value
                result = dapo_loss_func.compute_loss(output, batch, forward_only=False)
                assert result[0] is not None
                dapo_loss_func.compute_log_probs.assert_called_once_with(output=output, batch=batch, skip_entropy=True)
                dapo_loss_func._get_policy_loss_input.assert_called_once_with(batch=batch)
