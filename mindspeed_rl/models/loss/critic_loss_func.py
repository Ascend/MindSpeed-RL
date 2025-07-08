# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Dict, Tuple

import torch

from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.utils.utils import generate_mask
import mindspeed_rl.utils.torch_functional as F
from mindspeed_rl.utils.pad_process import truncate_middle_and_pad


@LossFuncFactory.register_loss('ray_ppo', 'critic')
class CriticLossFunc(BaseLossFunc):
    def __init__(self):
        super(CriticLossFunc, self).__init__()
        self.cliprange_value = 0.5

    def add_loss_meta_info(self, meta_info: Dict):
        if meta_info is None:
            return
        if "cliprange_value" in meta_info.keys():
            self.cliprange_value = float(meta_info["cliprange_value"])

    @staticmethod
    def _get_policy_loss_input(batch: Dict[str, torch.Tensor]):
        if 'responses' not in batch:
            raise ValueError("The responses is None")
        response_mask = generate_mask(batch['responses'], batch['response_length']).npu()
        values = batch['values'] if 'values' in batch else None
        returns = batch['returns'] if 'returns' in batch else None
        response_length = batch['response_length']
        return response_mask, values, returns, response_length


    @staticmethod
    def _get_compute_vpreds(output: torch.Tensor, batch: Dict[str, torch.Tensor]):
        if 'responses' not in batch:
            raise ValueError("The responses is None")
        responses = batch['responses']
        truncate_lengths = torch.cat([batch['prompt_length'], batch['prompt_length'] + batch['response_length']], dim=1) - 1
        vpreds = truncate_middle_and_pad(responses, output, truncate_lengths)
        return vpreds


    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False,
                     non_loss_data=True,
                     **kwargs) -> Tuple[torch.Tensor, Dict]:
        vpreds = self._get_compute_vpreds(output, batch)
        if forward_only:
            return vpreds
        response_mask, values, returns, response_length = self._get_policy_loss_input(batch=batch)

        vf_loss, vf_clipfrac = self._compute_value_loss(vpreds=vpreds,
                                                        returns=returns,
                                                        values=values,
                                                        response_mask=response_mask,
                                                        cliprange_value=self.cliprange_value)
        use_dynamic_bsz = kwargs.get('use_dynamic_bsz', False)
        actual_micro_batch_size = kwargs.get('actual_micro_batch_size', None)
        if use_dynamic_bsz and not forward_only:
            value_loss = vf_loss * (batch['responses'].size(0) / actual_micro_batch_size)
        else:
            value_loss = vf_loss
        stats = {
            'critic/vf_loss': abs(vf_loss.detach().item()),
            'critic/vf_clipfrac': vf_clipfrac.detach().item(),
        }
        return value_loss, stats

    @staticmethod
    def _compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
        """
        Args:
            vpreds: `(torch.Tensor)`
                shape: (bs, response_length)
            returns: `(torch.Tensor)`
                shape: (bs, response_length)
            values `(torch.Tensor)`
                shape: (bs, response_length)
            response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            cliprange_value: (float)
                The clip range used in GRPO.


        Returns:
            vf_loss: `a scalar torch.Tensor`
                policy gradient loss computed via GRPO
            vf_clipfrac: (float)
                a float number indicating the fraction of policy gradient loss being clipped

        """
        vpreds = vpreds.squeeze()
        vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
        vf_losses1 = (vpreds - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = 0.5 * F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
        vf_clipfrac = F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
        return vf_loss, vf_clipfrac


