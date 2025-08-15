# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Dict, Callable

import torch
from torch import Tensor

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from mindspeed_rl.utils.utils import mstx_timer_decorator


class Actor(BaseTrainingEngine):
    """
    Actor class. This class implements the simple logics.
    Args:
        model:  The network model to be trained.
        optimizer: The optimizer for updating model parameters (e.g., Adam).
        opt_param_scheduler: The scheduler for optimizer parameters (e.g., learning rate scheduler).
        beta: float = 0  The weight coefficient for KL divergence (used in algorithms like PPO).
        mini_batch_size_per_dp: int = 1  The size of the mini-batch for each data parallel stage.
        epochs: int = 1   The number of training epochs.
        shuffle_mini_batch: bool = False   Whether to shuffle the mini-batch data at each epoch.
        stage: str = None   The training stage identifier (e.g., pretrain/finetune).
        clip_ratio: float = 0.1   The clipping ratio threshold for PPO (limits the policy update range).
        forward_backward_func: Callable = None   The forward-backward function for distributed training.
        **kwargs:  # Additional parameters for base class argument passing.
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
            temperature: float = 1.0,
            forward_backward_func: Callable = None,
            token_level_loss: bool = True,
            clip_higher_enable: bool = False,
            clip_ratio_low: float = 0.1,
            clip_ratio_high: float = 0.1,
            **kwargs
    ):
        super(Actor, self).__init__(
            model,
            optimizer,
            opt_param_scheduler,
            beta=beta,
            mini_batch_size_per_dp=mini_batch_size_per_dp,
            epochs=epochs,
            shuffle_mini_batch=shuffle_mini_batch,
            stage=stage,
            clip_ratio=clip_ratio,
            temperature=temperature,
            role='actor',
            forward_backward_func=forward_backward_func,
            token_level_loss=token_level_loss,
            clip_higher_enable=clip_higher_enable,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            **kwargs)

    def get_loss_meta_func(self):
        meta_info = {}
        if self.clip_ratio is not None:
            meta_info["clip_ratio"] = self.clip_ratio
        if self.beta is not None:
            meta_info["beta"] = self.beta
        if self.kl_ctrl is not None:
            meta_info["kl_ctrl"] = self.kl_ctrl
        if self.entropy_coeff is not None:
            meta_info["entropy_coeff"] = self.entropy_coeff
        if self.kl_penalty is not None:
            meta_info["kl_penalty"] = self.kl_penalty
        if self.token_level_loss is not None:
            meta_info["token_level_loss"] = self.token_level_loss
        if self.clip_higher_enable is not None:
            meta_info["clip_higher_enable"] = self.clip_higher_enable
        if self.clip_ratio_low is not None:
            meta_info["clip_ratio_low"] = self.clip_ratio_low
        if self.clip_ratio_high is not None:
            meta_info["clip_ratio_high"] = self.clip_ratio_high
        return meta_info

    def post_process_forward_backward_output(self, output: torch.Tensor,
                                             batch: Dict[str, torch.Tensor]) -> (Tensor, Dict):
        return output, batch

    @mstx_timer_decorator
    def compute_log_prob(self, data: Dict) -> (Tensor, Dict):
        return super().forward(data)

    @mstx_timer_decorator
    def update_actor(self, data: Dict, kl_ctrl=None) -> Dict[str, torch.Tensor]:
        return super().update(data, kl_ctrl)
