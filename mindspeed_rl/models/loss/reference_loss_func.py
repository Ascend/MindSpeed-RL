# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Dict, Tuple

import torch

from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc


@LossFuncFactory.register_loss('ray_grpo', 'reference')
class ReferenceLossFunc(BaseLossFunc):
    def __init__(self):
        super(ReferenceLossFunc, self).__init__()

    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False, non_loss_data=True) -> Tuple[torch.Tensor, Dict]:
        # compute log probs
        log_probs = super().compute_log_probs(output=output, batch=batch)
        if forward_only:
            return log_probs
        return None
