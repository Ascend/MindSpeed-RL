# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from abc import ABC
from typing import Dict, List, Callable, Optional

from torch import Tensor

from mindspeed_rl.models.actor import Actor
from mindspeed_rl.utils.utils import mstx_timer_decorator


class ActorRolloutHybrid(ABC):
    """Hybrid actor model combining training and inference logic.

    This abstract base class provides a unified interface for actor models that
    require both training (parameter updates) and inference (sequence generation)
    capabilities. It manages separate training and inference model instances and
    handles distributed training coordination through a sharding manager.

    Attributes:
        train_actor (Actor): Training actor instance for parameter updates.
        inference_actor: Inference model instance for sequence generation.
        sharding_manager: Manager for distributed model sharding.
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
            token_level_loss: bool = True,
            clip_higher_enable: bool = False,
            clip_ratio_low: float = 0.1,
            clip_ratio_high: float = 0.1,
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
            token_level_loss=token_level_loss,
            clip_higher_enable=clip_higher_enable,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            **kwargs
        )
        self.inference_actor = inference_model
        self.sharding_manager = sharding_manager

    @mstx_timer_decorator
    def compute_image_embeds(self, data: Dict) -> Tensor:
        return self.train_actor.compute_image_embeds(data)

    @mstx_timer_decorator
    def generate_sequences(
            self,
            prompts_list: List[List[int]],
            indexes=None,
            async_engine=False,
            stop_singal_func=None,
            **kwargs) -> Tensor:
        if async_engine:
            res = self.inference_actor.async_generate_sequences(
                prompts_list,
                indexes,
                stop_singal_func=stop_singal_func,
                **kwargs,
            )
        else:
            res = self.inference_actor.generate_sequences(prompts_list, **kwargs)[0]
        return res

    @mstx_timer_decorator
    def compute_log_prob(self, data: Dict) -> Tensor:
        return self.train_actor.compute_log_prob(data)

    @mstx_timer_decorator
    def update_actor(self, data: Dict, kl_ctrl=None) -> Dict[str, Tensor]:
        return self.train_actor.update_actor(data, kl_ctrl)