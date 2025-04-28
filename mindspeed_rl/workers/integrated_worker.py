# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import dataclasses
from typing import Callable

import time
import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronOffLoader

from mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorkerBase
from mindspeed_rl.workers.reference_woker import ReferenceWorkerBase
from mindspeed_rl.workers.reward_woker import RewardWorkerBase
from mindspeed_rl.models.reference import Reference
from mindspeed_rl.models.reward import Reward


@ray.remote(resources={"NPU": 0.7})
class IntegratedWorker(ActorHybridWorkerBase, ReferenceWorkerBase, RewardWorkerBase):
    """
    IntegratedWorker class. This class implements the integrated worker for the Actor, Reference and Reward Worker.

    Args:
        megatron_config: MegatronConfig Configuration for Megatron-LM (e.g., model parallelism settings).
        rl_config: RLConfig Configuration for reinforcement learning (e.g., PPO settings).
        generate_config: GenerateConfig Configuration for generation/inference (e.g., vLLM settings).
        model_provider: Callable Function to provide the model instance.
        initialize_func: Callable Function to initialize the model and environment.
        tokenizer: BaseTokenizer = None Object to retrieve the tokenizer.
        get_megatron_module: Callable = megatron_module from get_megatron_module.
        **kwargs: Additional parameters for base class argument passing.
    """

    def __init__(
            self,
            megatron_config: MegatronConfig,
            rl_config: RLConfig,
            generate_config: GenerateConfig,
            model_provider: Callable,
            initialize_func: Callable,
            tokenizer: BaseTokenizer = None,
            get_megatron_module: Callable = None,
            **kwargs
    ):

        # We use Actor as main worker, so only do init for Actor here.
        ActorHybridWorkerBase.__init__(
            self,
            megatron_config,
            rl_config,
            generate_config,
            model_provider=model_provider,
            initialize_func=initialize_func,
            tokenizer=tokenizer,
            get_megatron_module=get_megatron_module,
            **kwargs
        )

        self.update_micro_batch_size = rl_config.update_micro_batch_size

        self.reference = None
        self.ref_model = None
        self.ref_manager = None


    def initialize(self):

        # Based on Actor
        ActorHybridWorkerBase.initialize(self)

        # Add Reference
        self.ref_model = self.get_model(self.model_provider, self.model_type, wrap_with_ddp=False)
        ref_model_load_path = getattr(
            self.rl_config.integrated_mode_config, "ref_model_load_path", None
        ) if self.rl_config.integrated_mode_config is not None else None
        self.load_checkpoint_with_path(self.ref_model, ref_model_load_path)
        self.ref_manager = MegatronOffLoader(self.ref_model, wrap_with_ddp=False)
        self.ref_manager.offload_param()
        self.reference = Reference(
            self.ref_model,
            beta=self.rl_config.beta,
            mini_batch_size=self.rl_config.mini_batch_size,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            micro_batch_size=self.megatron_config.micro_batch_size
        )

    def compute_ref_log_prob(self):
        start_onload_time = time.time()
        self.ref_manager.onload_param()
        end_onload_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/onload", 
                value=[round(end_onload_time, 4), round(start_onload_time, 4)],
                cumulate=True
            )
        ) 
        
        ReferenceWorkerBase.compute_ref_log_prob(self)
        
        start_offload_time = time.time()
        self.ref_manager.offload_param()
        end_offload_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/offload",
                value=[round(end_offload_time, 4), round(start_offload_time, 4)],
                cumulate=True
            )
        )

    def update(self, kl_ctrl=None, skip_actor_log_prob=False):
        # set update mbs
        update_mbs = self.update_micro_batch_size
        mbs = self.actor_hybrid.train_actor.micro_batch_size

        args = self.get_args()

        if update_mbs is not None:
            self.actor_hybrid.train_actor.micro_batch_size = update_mbs
            args.micro_batch_size = update_mbs

        ActorHybridWorkerBase.update(self, kl_ctrl, skip_actor_log_prob)

        args.micro_batch_size = mbs
        self.actor_hybrid.train_actor.micro_batch_size = mbs

    def load_checkpoint_with_path(self, model, path, info=None):
        if path is None:
            self.load_checkpoint(model, None, None)
        else:
            origin_load_dir = getattr(self.get_args(), "load")
            setattr(self.get_args(), "load", path)
            self.load_checkpoint(model, None, None)
            setattr(self.get_args(), "load", origin_load_dir)

