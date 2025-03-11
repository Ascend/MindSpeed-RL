# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Dict, Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine


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
            forward_backward_func: Callable = None,
            **kwargs
    ):
        super(Reference, self).__init__(
            model,
            beta=beta,
            stage=stage,
            role='reference',
            forward_backward_func=forward_backward_func,
            **kwargs
        )

    def post_process_forward_backward_output(self, output: [torch.Tensor],
                                             batch: Dict[str, torch.Tensor]) -> (Tensor, Dict):
        return output, batch

    def compute_log_prob(self, data: DataLoader) -> (Tensor, Dict):
        return super().forward(data)
