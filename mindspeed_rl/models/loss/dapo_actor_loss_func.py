# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Dict, Tuple

import torch

from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.utils.utils import generate_mask
import mindspeed_rl.utils.torch_functional as F
from mindspeed_rl.utils.utils import MsProbe


@LossFuncFactory.register_loss('ray_dapo', 'actor')
class DAPOActorLossFunc(BaseLossFunc):
    def __init__(self):
        super().__init__()
        self.clip_ratio = 0.2
        self.entropy_coeff = 0.0

    def add_loss_meta_info(self, meta_info: Dict):
        if meta_info is None:
            return
        if "clip_ratio" in meta_info.keys():
            self.clip_ratio = float(meta_info["clip_ratio"])
        if "kl_ctrl" in meta_info.keys():
            self.kl_ctrl = meta_info["kl_ctrl"]
        if "entropy_coeff" in meta_info.keys():
            self.entropy_coeff = meta_info["entropy_coeff"]
        if "token_level_loss" in meta_info.keys():
            self.token_level_loss = bool(meta_info["token_level_loss"])
        if "clip_higher_enable" in meta_info.keys():
            self.clip_higher_enable = bool(meta_info["clip_higher_enable"])
        if "clip_ratio_low" in meta_info.keys():
            self.clip_ratio_low = float(meta_info["clip_ratio_low"])
        if "clip_ratio_high" in meta_info.keys():
            self.clip_ratio_high = float(meta_info["clip_ratio_high"])

    @staticmethod
    def _get_policy_loss_input(batch: Dict[str, torch.Tensor]):
        if 'responses' not in batch:
            raise ValueError("The responses is None")
        response_mask = batch['response_mask'] if 'response_mask' in batch else generate_mask(batch['responses'], batch['response_length']).npu()
        old_log_prob = batch['old_log_prob'] if 'old_log_prob' in batch else None
        advantages = batch['advantages'] if 'advantages' in batch else None
        return response_mask, old_log_prob, advantages

    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False,
                     non_loss_data=True,
                     **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失函数，子类必须实现。
        :param output: 模型的输出 logits。
        :param batch: 输入数据，包含 responses、attention_mask 等。
        :param forward_only: 是否只进行前向计算。
        :return: 损失值和统计信息。
        """
        # compute log probs
        if forward_only:
            log_probs, _ = super().compute_log_probs(output=output, batch=batch, **kwargs)
            return log_probs
        log_probs, entropy = super().compute_log_probs(output=output, batch=batch, skip_entropy=False, **kwargs)

        response_mask, old_log_prob, advantages = self._get_policy_loss_input(batch=batch)
        # compute policy loss
        if not self.clip_higher_enable:
            self.clip_ratio_low = self.clip_ratio
            self.clip_ratio_high = self.clip_ratio

        pg_loss, pg_clipfrac, ppo_kl, kl_loss, entropy_loss = self._compute_dapo_policy_loss(old_log_prob=old_log_prob,
                                                        log_prob=log_probs,
                                                        entropy=entropy,
                                                        entropy_coeff=self.entropy_coeff,
                                                        advantages=advantages,
                                                        eos_mask=response_mask,
                                                        clip_ratio_low=self.clip_ratio_low,
                                                        clip_ratio_high=self.clip_ratio_high,
                                                        token_level_loss=self.token_level_loss)
        
        use_dynamic_bsz = kwargs.get('use_dynamic_bsz', False)
        actual_micro_batch_size = kwargs.get('actual_micro_batch_size', None)
        if use_dynamic_bsz and not forward_only:
            policy_loss = pg_loss * (batch['responses'].size(0) / actual_micro_batch_size)
        else:
            policy_loss = pg_loss
        data_tobe_saved = {
            "old_log_prob": old_log_prob,
            "log_prob": log_probs,
            "advantages": advantages,
            "loss": pg_loss,
            "kl_loss": kl_loss,
        }
        MsProbe.save_data(data_tobe_saved)
        stats = {
            'actor/pg_loss': abs(pg_loss.detach().item()),
            'actor/pg_clipfrac': pg_clipfrac.detach().item(),
            'actor/ppo_kl': ppo_kl.detach().item(),
            'actor/kl_loss': kl_loss.detach().item(),
            'actor/entropy': entropy_loss.detach().item()
        }
        return policy_loss, stats


    @staticmethod
    def _compute_dapo_policy_loss(old_log_prob, log_prob, entropy, entropy_coeff, advantages, eos_mask, clip_ratio_low, clip_ratio_high, token_level_loss):
        """
        Args:
            old_log_prob: `(torch.Tensor)`
                shape: (bs, response_length)
            log_prob: `(torch.Tensor)`
                shape: (bs, response_length)
            advantages: `(torch.Tensor)`
                shape: (bs, response_length)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            clip_ratio_low: (float)
                The low clip range used in DAPO.
            clip_ratio_high: (float)
                The high clip range used in DAPO.

        Returns:
            pg_loss: `a scalar torch.Tensor`
                policy gradient loss computed via DAPO
            pg_clipfrac: (float)
                a float number indicating the fraction of policy gradient loss being clipped

        """
        if old_log_prob is None:
            old_log_prob = log_prob.detach().clone()
        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = F.masked_mean(-negative_approx_kl, eos_mask)
        entropy_loss = F.masked_mean(entropy, eos_mask)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

        if token_level_loss:
            #token-mean
            pg_mean_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
        else:
            #seq-mean-token-mean
            pg_losses = torch.max(pg_losses, pg_losses2)
            pg_mean_loss = torch.sum(pg_losses * eos_mask, dim=-1) / torch.sum(eos_mask, dim=-1)
            pg_mean_loss = torch.mean(pg_mean_loss)

        pg_mean_clipfrac = F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)


        kl_loss = torch.zeros_like(pg_mean_loss)
        pg_loss = pg_mean_loss - entropy_coeff * entropy_loss
        return pg_loss, pg_mean_clipfrac, ppo_kl, kl_loss, entropy_loss