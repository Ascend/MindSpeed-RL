# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Dict, Tuple, Callable

import torch
from torch import Tensor

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from mindspeed_rl.utils.utils import mstx_timer_decorator


class Critic(BaseTrainingEngine):
    """
    Reward class. This class implements the simple logics.

    Args:
        model: The network model to be used for reward computation.
        beta: float = 0 The weight coefficient for KL divergence (used in algorithms like PPO).
        stage: str = None The training stage identifier (e.g., pretrain/finetune).
        cliprange_value: float = 0.5   The clipping ratio threshold for critic (limits the policy update range).
        forward_backward_func: Callable = None The forward-backward function for distributed training.
        **kwargs: Additional parameters for base class argument passing.
    """

    def __init__(
            self,
            model,
            optimizer,
            opt_param_scheduler,
            beta: float = 0,
            mini_batch_size_per_dp: int = 1,
            epochs: int = 1,
            shuffle_mini_batch: bool = False,
            stage: str = None,
            clip_ratio: float = 0.1,
            forward_backward_func: Callable = None,
            cliprange_value: float = 0.5,
            **kwargs
    ):
        super(Critic, self).__init__(
            model,
            optimizer,
            opt_param_scheduler,
            beta=beta,
            mini_batch_size_per_dp=mini_batch_size_per_dp,
            epochs=epochs,
            shuffle_mini_batch=shuffle_mini_batch,
            stage=stage,
            clip_ratio=clip_ratio,
            role='critic',
            forward_backward_func=forward_backward_func,
            cliprange_value=cliprange_value,
            **kwargs)

    def get_loss_meta_func(self):
        meta_info = {}
        if self.cliprange_value is not None:
            meta_info["cliprange_value"] = self.cliprange_value
        if self.entropy_coeff is not None:
            meta_info["entropy_coeff"] = self.entropy_coeff
        return meta_info
        
    def post_process_forward_backward_output(self, output: [torch.Tensor],
                                             batch: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        return output, batch

    @mstx_timer_decorator
    def compute_values(self, data: Dict):
        return super().forward(data)
    
    @mstx_timer_decorator
    def update_critic(self, data: Dict, kl_ctrl) -> Tensor:
        return super().update(data, kl_ctrl)
