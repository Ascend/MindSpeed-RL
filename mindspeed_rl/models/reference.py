# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Dict, Callable

import torch
from torch import Tensor

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from mindspeed_rl.utils.utils import mstx_timer_decorator


class Reference(BaseTrainingEngine):
    """
    Reference class. This class implements the simple logics.

    Args:
        model: The network model to be used as a reference.
        beta: float = 0 The weight coefficient for KL divergence (used in algorithms like PPO).
        stage: str = None The training stage identifier (e.g., pretrain/finetune).
        forward_backward_func: Callable = None The forward-backward function for distributed training.
        **kwargs: Additional parameters for base class argument passing.
    """

    def __init__(
            self,
            model,
            beta=0,
            stage=None,
            temperature: float = 1.0,
            forward_backward_func: Callable = None,
            **kwargs
    ):
        super(Reference, self).__init__(
            model,
            beta=beta,
            stage=stage,
            role='reference',
            temperature=temperature,
            forward_backward_func=forward_backward_func,
            **kwargs
        )

    def post_process_forward_backward_output(self, output: [torch.Tensor],
                                             batch: Dict[str, torch.Tensor]) -> (Tensor, Dict):
        return output, batch

    @mstx_timer_decorator
    def compute_log_prob(self, data: Dict) -> (Tensor, Dict):
        return super().forward(data)
