# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import dataclasses
from typing import Callable
from contextlib import contextmanager

import time
import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronOffLoader
from mindspeed_rl.utils.utils import mstx_timer_decorator, profiler_start, profiler_step
from mindspeed_rl.utils.utils import MsProbe

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
        profiler_config: ProfilerConfig, Configuration for profiling.
        msprobe_config: MsprobeConfig, Configuration for msprobe.
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
            profiler_config: ProfilerConfig = None,
            msprobe_config: MsprobeConfig = None,
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
            profiler_config=profiler_config,
            msprobe_config=msprobe_config,
            **kwargs
        )

        self.actor_forward_micro_batch_size = rl_config.actor_forward_micro_batch_size
        self.ref_forward_micro_batch_size = rl_config.ref_forward_micro_batch_size

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
        self.load_checkpoint_with_path(self.ref_model, ref_model_load_path, ckpt_only=True)
        self.ref_manager = MegatronOffLoader(self.ref_model, wrap_with_ddp=False)
        self.ref_manager.offload_param()
        
        self.reference = Reference(
            self.ref_model,
            megatron_config=self.megatron_config,
            beta=self.rl_config.beta,
            mini_batch_size=self.rl_config.mini_batch_size,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            micro_batch_size=self.megatron_config.micro_batch_size,
            use_dynamic_bsz=self.rl_config.use_dynamic_bsz,
            max_packing_token_size=self.rl_config.max_packing_token_size,
            dynamic_max_batch_size=self.rl_config.dynamic_max_batch_size,
            use_remove_padding=self.rl_config.use_remove_padding,
            set_actual_seq_len=self.set_actual_seq_len,
            get_actual_seq_len=self.get_actual_seq_len,
            set_position_ids=self.set_position_ids,
            context_parallel_size=self.megatron_config.context_parallel_size,
            temperature=self.generate_config.sampling_config["temperature"]
        )
        MsProbe.config_init(self.msprobe_config)

    @mstx_timer_decorator
    def compute_ref_log_prob(self):
        start_onload_time = time.time()
        self.ref_manager.onload_param()
        end_onload_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/ref_onload",
                value=[round(end_onload_time, 4), round(start_onload_time, 4)],
                cumulate=True
            )
        )
        compute_log_prob_profiler = profiler_start(self.profiler_config, role="reference_compute_log_prob",
                                            profiler_iteration=self.prof_iteration)
        MsProbe.debugger_start(model=self.ref_model, tag="reference_compute_log_prob")
        if self.ref_forward_micro_batch_size is not None:
            with temporary_micro_batch_size(
                    worker=self.reference,
                    args=self.get_args(),
                    new_mbs=self.ref_forward_micro_batch_size
            ):
                ReferenceWorkerBase.compute_ref_log_prob(self)
        else:
            ReferenceWorkerBase.compute_ref_log_prob(self)
        profiler_step(compute_log_prob_profiler)
        MsProbe.debugger_stop("reference_compute_log_prob")
        start_offload_time = time.time()
        self.ref_manager.offload_param()
        torch.cuda.empty_cache()
        end_offload_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/ref_offload",
                value=[round(end_offload_time, 4), round(start_offload_time, 4)],
                cumulate=True
            )
        )

    def compute_log_prob(self):
        if self.actor_forward_micro_batch_size is not None:
            with temporary_micro_batch_size(
                    worker=self.actor_hybrid.train_actor,
                    args=self.get_args(),
                    new_mbs=self.actor_forward_micro_batch_size
            ):
                ActorHybridWorkerBase.compute_log_prob(self)
        else:
            ActorHybridWorkerBase.compute_log_prob(self)

    def load_checkpoint_with_path(self, model, path, ckpt_only=False):
        """Load model checkpoint from a specified path with flexible control.

        Args:
            model: The model to load checkpoint into.
            path: Path to the checkpoint file/directory. If None, use the path in megatron args.
            ckpt_only: If True, only loads model weights (skips optimizer/RNG states).
        """

        # Backup original arguments if needed
        original_args = {
            'no_load_optim': getattr(self.get_args(), "no_load_optim", None),
            'no_load_rng': getattr(self.get_args(), "no_load_rng", None),
            'load': getattr(self.get_args(), "load", None),
            'iteration': getattr(self.get_args(), "iteration", None),
            'finetune': getattr(self.get_args(), "finetune", None),
            'consumed_train_samples': getattr(self.get_args(), "consumed_train_samples", None),
            'consumed_valid_samples': getattr(self.get_args(), "consumed_valid_samples", None),
        } if ckpt_only or path else {}

        if ckpt_only:
            self._set_args({
                "no_load_optim": True,
                "no_load_rng": True,
                "finetune": True,
                'consumed_train_samples': 0,
                'consumed_valid_samples': 0
            })

        if path is not None:
            self._set_args({"load": path})

        self.load_checkpoint(model, None, None)

        if original_args:
            self._set_args(original_args)

    def _set_args(self, arg_dict):
        for key, value in arg_dict.items():
            if hasattr(self.get_args(), key):
                setattr(self.get_args(), key, value)


@contextmanager
def temporary_micro_batch_size(worker, args, new_mbs):
    original_mbs = args.micro_batch_size
    try:
        worker.micro_batch_size = new_mbs
        args.micro_batch_size = new_mbs
        yield
    finally:
        worker.micro_batch_size = original_mbs
        args.micro_batch_size = original_mbs