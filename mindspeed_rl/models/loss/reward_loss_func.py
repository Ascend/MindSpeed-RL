# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Dict, Tuple

import torch

from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc


@LossFuncFactory.register_loss('ray_grpo', 'reward')
@LossFuncFactory.register_loss('ray_ppo', 'reward')
class RewardLossFunc(BaseLossFunc):
    def __init__(self):
        super(RewardLossFunc, self).__init__()

    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False,
                     use_dynamic_bsz=False,
                     actual_micro_batch_size=1,
                     non_loss_data=True) -> Tuple[torch.Tensor, Dict]:
        return output