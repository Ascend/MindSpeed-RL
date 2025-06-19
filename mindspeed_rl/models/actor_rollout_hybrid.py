# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from abc import ABC
from typing import Dict, List, Callable

from torch import Tensor

from mindspeed_rl.models.actor import Actor
from mindspeed_rl.utils.utils import mstx_timer_decorator


class ActorRolloutHybrid(ABC):
    """
    ActorRolloutHybrid class. This class combines training and inference logic for hybrid actor models.

    Args:
        model: The network model to be trained.
        optimizer: The optimizer for updating model parameters (e.g., Adam).
        opt_param_scheduler: The scheduler for optimizer parameters (e.g., learning rate scheduler).
        inference_model: The model used for inference/generation.
        sharding_manager: The manager for handling model sharding (e.g., for distributed training).
        beta: float = 0 The weight coefficient for KL divergence (used in algorithms like PPO).
        mini_batch_size_per_dp: int = 1 The size of the mini-batch for each data parallel stage.
        epochs: int = 1 The number of training epochs.
        shuffle_mini_batch: bool = False Whether to shuffle the mini-batch data at each epoch.
        stage: str = None The training stage identifier (e.g., pretrain/finetune).
        generate_config: GenerateConfig = None Configuration for generation/inference (e.g., vLLM settings).
        clip_ratio: float = 0.1 The clipping ratio threshold for PPO (limits the policy update range).
        forward_backward_func: Callable = None The forward-backward function for distributed training.
        **kwargs: Additional parameters for base class argument passing.
    """

    def __init__(
            self,
            model,
            optimizer,
            opt_param_scheduler,
            inference_model,
            sharding_manager,
            beta: float = 0,
            mini_batch_size_per_dp: int = 1,
            epochs: int = 1,
            shuffle_mini_batch: bool = False,
            stage: str = None,
            clip_ratio: float = 0.1,
            temperature: float = 1.0,
            forward_backward_func: Callable = None,
            micro_batch_size: int = 1,
            **kwargs
    ):

        self.train_actor = Actor(
            model,
            optimizer,
            opt_param_scheduler,
            beta=beta,
            mini_batch_size_per_dp=mini_batch_size_per_dp,
            epochs=epochs,
            shuffle_mini_batch=shuffle_mini_batch,
            clip_ratio=clip_ratio,
            temperature=temperature,
            stage=stage,
            forward_backward_func=forward_backward_func,
            micro_batch_size=micro_batch_size,
            **kwargs
        )
        self.inference_actor = inference_model
        self.sharding_manager = sharding_manager

    @mstx_timer_decorator
    def generate_sequences(self, prompts_list: List[List[int]], **kwargs) -> Tensor:
        responses = self.inference_actor.generate_sequences(prompts_list, **kwargs)[0]
        return responses

    @mstx_timer_decorator
    def compute_log_prob(self, data: Dict) -> Tensor:
        return self.train_actor.compute_log_prob(data)

    @mstx_timer_decorator
    def update_actor(self, data: Dict, kl_ctrl=None) -> Dict[str, Tensor]:
        return self.train_actor.update_actor(data, kl_ctrl)
