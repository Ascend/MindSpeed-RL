# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Dict, Tuple, Callable

import torch
from torch import Tensor

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from mindspeed_rl.utils.utils import mstx_timer_decorator


class Reward(BaseTrainingEngine):
    """Reward model for RL training.

    This class implements a reward model that computes reward scores for
    generated sequences. It serves as a learned reward function in RL
    algorithms and supports KL divergence penalty computation when beta > 0.

    Attributes:
        model: The network model used for reward computation.
        beta: Weight coefficient for KL divergence penalty.
        stage: Training stage identifier (e.g., 'pretrain', 'finetune').
        temperature: Sampling temperature for probability distribution.
        forward_backward_func: Function for distributed forward-backward computation.
    """

    def __init__(
            self,
            model,
            beta: float = 0,
            stage: str = None,
            temperature: float = 1.0,
            forward_backward_func: Callable = None,
            **kwargs
    ):
        super(Reward, self).__init__(
            model,
            beta=beta,
            stage=stage,
            role='reward',
            temperature=temperature,
            forward_backward_func=forward_backward_func,
            **kwargs
        )

    def post_process_forward_backward_output(self, output: [torch.Tensor],
                                             batch: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        return output, batch

    @mstx_timer_decorator
    def compute_rm_score(self, data: Dict) -> Tensor:
        return super().forward(data)