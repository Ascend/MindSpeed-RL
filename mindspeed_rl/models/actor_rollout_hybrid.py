# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from abc import ABC
from typing import Dict, List, Callable

from torch import Tensor
from torch.utils.data import DataLoader

from mindspeed_rl.config_cls.vllm_config import vLLMConfig
from mindspeed_rl.models.actor import Actor


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
        mini_batch_size: int = 1 The size of the mini-batch for each training step.
        epochs: int = 1 The number of training epochs.
        shuffle_mini_batch: bool = False Whether to shuffle the mini-batch data at each epoch.
        stage: str = None The training stage identifier (e.g., pretrain/finetune).
        generate_config: vLLMConfig = None Configuration for generation/inference (e.g., vLLM settings).
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
            mini_batch_size: int = 1,
            epochs: int = 1,
            shuffle_mini_batch: bool = False,
            stage: str = None,
            generate_config: vLLMConfig = None,
            clip_ratio: float = 0.1,
            forward_backward_func: Callable = None,
            **kwargs
    ):
        self.generate_config = generate_config

        self.train_actor = Actor(
            model,
            optimizer,
            opt_param_scheduler,
            beta=beta,
            mini_batch_size=mini_batch_size,
            epochs=epochs,
            shuffle_mini_batch=shuffle_mini_batch,
            clip_ratio=clip_ratio,
            stage=stage,
            forward_backward_func=forward_backward_func,
            **kwargs
        )
        self.inference_actor = inference_model
        self.sharding_manager = sharding_manager

    def generate_sequences(self, prompts_list: List[List[int]]) -> Tensor:
        responses = self.inference_actor.generate_sequences(prompts_list)[0]
        return responses

    def compute_log_prob(self, data: DataLoader) -> Tensor:
        return self.train_actor.compute_log_prob(data)

    def update_actor(self, data: DataLoader, kl_ctrl=None) -> Dict[str, Tensor]:
        return self.train_actor.update_actor(data, kl_ctrl)
