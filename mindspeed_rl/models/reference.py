# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Dict, Callable

import torch
from torch import Tensor

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from mindspeed_rl.utils.utils import mstx_timer_decorator


class Reference(BaseTrainingEngine):
    """Reference model for RL training.

    This class implements a reference model that serves as a baseline policy
    for computing KL divergence penalties in RL algorithms. It inherits from
    BaseTrainingEngine and provides specific implementations for reference
    model operations.

    Attributes:
        model: The network model used as reference.
        beta: Weight coefficient for KL divergence penalty.
        stage: Training stage identifier (e.g., 'pretrain', 'finetune').
        temperature: Sampling temperature for probability distribution.
        forward_backward_func: Function for distributed forward-backward computation.
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