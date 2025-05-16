# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Dict, Tuple

import torch

from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.utils.utils import generate_mask
import mindspeed_rl.utils.torch_functional as F


@LossFuncFactory.register_loss('ray_grpo', 'actor')
class GRPOActorLossFunc(BaseLossFunc):
    def __init__(self):
        super().__init__()
        self.clip_ratio = 0.2

    def add_loss_meta_info(self, meta_info: Dict):
        if meta_info is None:
            return
        if "clip_ratio" in meta_info.keys():
            self.clip_ratio = float(meta_info["clip_ratio"])
        if "kl_ctrl" in meta_info.keys():
            self.kl_ctrl = meta_info["kl_ctrl"]

    @staticmethod
    def _get_policy_loss_input(batch: Dict[str, torch.Tensor]):
        if 'responses' not in batch:
            raise ValueError("The responses is None")
        response_mask = generate_mask(batch['responses'], batch['response_length']).npu()
        old_log_prob = batch['old_log_prob'] if 'old_log_prob' in batch else None
        advantages = batch['advantages'] if 'advantages' in batch else None
        ref_log_prob = batch['ref_log_prob'] if 'ref_log_prob' in batch else None
        return response_mask, old_log_prob, advantages, ref_log_prob

    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False, non_loss_data=True) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失函数，子类必须实现。
        :param output: 模型的输出 logits。
        :param batch: 输入数据，包含 responses、attention_mask 等。
        :param forward_only
        :return: 损失值和统计信息。
        """
        # compute log probs
        log_probs = super().compute_log_probs(output=output, batch=batch)
        if forward_only:
            return log_probs

        response_mask, old_log_prob, advantages, ref_log_prob = self._get_policy_loss_input(batch=batch)
        # compute policy loss
        pg_loss, pg_clipfrac, ppo_kl, kl_loss = self._compute_grpo_policy_loss(old_log_prob=old_log_prob,
                                                                      log_prob=log_probs,
                                                                      ref_log_prob=ref_log_prob,
                                                                      advantages=advantages,
                                                                      eos_mask=response_mask,
                                                                      cliprange=self.clip_ratio,
                                                                      kl_ctrl=self.kl_ctrl)
        policy_loss = pg_loss
        stats = {
            'actor/pg_loss': abs(pg_loss.detach().item()),
            'actor/pg_clipfrac': pg_clipfrac.detach().item(),
            'actor/ppo_kl': ppo_kl.detach().item(),
            'actor/kl_loss': kl_loss.detach().item()
        }
        return policy_loss, stats

    @staticmethod
    def _compute_grpo_policy_loss(old_log_prob, log_prob, ref_log_prob, advantages, eos_mask, cliprange, kl_ctrl):
        """
        Args:
            old_log_prob: `(torch.Tensor)`
                shape: (bs, response_length)
            log_prob: `(torch.Tensor)`
                shape: (bs, response_length)
            ref_log_prob `(torch.Tensor)`
                shape: (bs, response_length)
            advantages: `(torch.Tensor)`
                shape: (bs, response_length)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            cliprange: (float)
                The clip range used in GRPO.
            kl_ctrl: (float)
                The kL value


        Returns:
            pg_loss: `a scalar torch.Tensor`
                policy gradient loss computed via GRPO
            pg_clipfrac: (float)
                a float number indicating the fraction of policy gradient loss being clipped

        """
        if old_log_prob is None:
            old_log_prob = log_prob.detach().clone()
        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = F.masked_mean(-negative_approx_kl, eos_mask)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

        pg_mean_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
        pg_mean_clipfrac = F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

        ref_approx_kl = ref_log_prob - log_prob
        ratio_kl = torch.exp(ref_approx_kl)
        kl_losses = ratio_kl - ref_approx_kl - 1
        kl_mean_loss = F.masked_mean(kl_losses, eos_mask)
        kl_loss = kl_mean_loss * kl_ctrl.value
        pg_loss = pg_mean_loss + kl_mean_loss * kl_ctrl.value
        return pg_loss, pg_mean_clipfrac, ppo_kl, kl_loss
