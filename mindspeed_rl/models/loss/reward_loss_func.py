# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Dict, Tuple

import torch

from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.utils.remove_padding import postprocess_packed_seqs


@LossFuncFactory.register_loss('ray_grpo', 'reward')
class RewardLossFunc(BaseLossFunc):
    def __init__(self):
        super(RewardLossFunc, self).__init__()

    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False,
                     non_loss_data=True,
                     **kwargs) -> Tuple[torch.Tensor, Dict]:
        use_remove_padding = kwargs.get('use_remove_padding', False)
        if use_remove_padding:
            output = postprocess_packed_seqs(
                output=output,
                seqlens_in_batch=kwargs.get('seqlens_in_batch', None),
                cu_seqlens_padded=kwargs.get('cu_seqlens_padded', None),
                seq_len=kwargs.get('seq_len', None)
            )
        return output

