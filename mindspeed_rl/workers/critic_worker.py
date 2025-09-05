# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
from typing import Callable
import logging as logger

import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig
from mindspeed_rl.models.critic import Critic
from mindspeed_rl.utils.pad_process import truncate_rows
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader
from mindspeed_rl.utils.utils import MsProbe, mstx_timer_decorator
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.trainer.utils.parallel_state import is_pipeline_last_stage, get_tensor_model_parallel_rank
from mindspeed_rl.utils.utils import num_floating_point_operations, get_attr_wrapped_model, profiler_start, profiler_step


class CriticWorkerBase(BaseWorker):
    """
    RewardWorker class. This class implements the worker logic for reward model training and inference.

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
        super().__init__(
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
        self.reward = None
        self.num_floating_point_operations_so_far = 0
        self.critic_offloader = None
        self.critic_profiler = None
        self.prof_iteration = 1

    def initialize(self):
        self.setup_distributed_rank()
        self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()
        self._set_no_sync_func()
        self.critic_offloader = MegatronOffLoader(
            self.model,
            self.optimizer,
            megatron_config=self.megatron_config,
            distributed_optimizer=self.distributed_optimizer,
            float16_optimizer_with_float16_params=self.float16_optimizer_with_float16_params)
        self.critic_offloader.offload_optimizer()
        self.critic_offloader.offload_grad()
        self.critic_offloader.offload_param()

        self.critic = Critic(
            self.model,
            megatron_config=self.megatron_config,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            mini_batch_size_per_dp=self.rl_config.mini_batch_size
                                   // self.parallel_state.get_data_parallel_world_size(),
            beta=self.rl_config.beta,
            stage=self.megatron_config.stage,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            forward_backward_func=self.forward_backward_func,
            clip_ratio=self.rl_config.clip_ratio,
            micro_batch_size=self.megatron_config.micro_batch_size,
            entropy_coeff=self.rl_config.entropy_coeff,
            cliprange_value=self.rl_config.cliprange_value,

            use_dynamic_bsz=self.rl_config.use_dynamic_bsz,
            max_packing_token_size=self.rl_config.max_packing_token_size,
            dynamic_max_batch_size=self.rl_config.dynamic_max_batch_size,
            use_remove_padding=self.rl_config.use_remove_padding,
            set_actual_seq_len=self.set_actual_seq_len,
            get_actual_seq_len=self.get_actual_seq_len,
            set_position_ids=self.set_position_ids,
            context_parallel_size=self.megatron_config.context_parallel_size
        )
        self.empty_cache()
        self.critic_profiler = profiler_start(self.profiler_config, self.profiler_config.role)
        MsProbe.config_init(self.msprobe_config)

    def init_transfer_dock(self, td, mm_td=None, sampling_transfer_dock=None, mm_sampling_transfer_dock=None):
        self.td = td
        self.mm_td = mm_td
        self.sampling_transfer_dock = sampling_transfer_dock
        self.mm_sampling_transfer_dock = mm_sampling_transfer_dock

    def get_iteration(self):
        return self.args.iteration

    @mstx_timer_decorator
    def compute_values(self):
        self.critic_offloader.onload_param()

        experience_consumer_stage = 'critic_compute_values'
        experience_columns = ['input_ids', 'prompt_length', "responses", "response_length"]
        experience_count = self.rl_config.critic_value_dispatch_size

        sorted_indexes = self.get_dp_range_indexes(
            experience_count,
            use_vllm=False,
        ) if self.rl_config.guarantee_order else None

        critic_compute_values_profiler = profiler_start(
            self.profiler_config, 
            role="critic_compute_values",
            profiler_iteration=self.prof_iteration
        )

        MsProbe.debugger_start(self.model[0], tag='critic_compute_values')

        start_time_defined = False
        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                tp_size=self.megatron_config.tensor_model_parallel_size,
                cp_size=self.megatron_config.context_parallel_size,
                cp_algo=self.megatron_config.context_parallel_algo,
                indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None,
                get_n_samples=False
            )
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:
                output, batch = self.critic.compute_values(batch_data)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    values = torch.cat(output, dim=0).squeeze(-1)
                    values = values.to(torch.float32)
                    values = truncate_rows(values, batch['response_length'])
                    output = {'values': values}
                    self.collect_transfer_dock_data(output, index)

                    MsProbe.save_data({"values": values})

                end_time = time.time()
                ray.get(
                    self.td.update_metrics.remote(
                        "timing/critic_model",
                        value=[round(end_time, 4), round(start_time, 4)],
                        cumulate=True
                    )
                )
                ray.get(
                self.td.update_metrics.remote(
                    "end_time/critic_model",
                    value=[round(end_time, 4)],
                    cumulate=True
                )
            )

        profiler_step(critic_compute_values_profiler)
        MsProbe.debugger_stop('critic_compute_values')
        logger.info("finish compute_values")

    @mstx_timer_decorator
    def update(self, kl_ctrl=None):
        self.critic_offloader.onload_optimizer()
        self.critic_offloader.onload_grad()

        self.args.curr_iteration = self.iteration

        experience_consumer_stage = 'critic_train'

        experience_columns = ['responses', 'advantages', 'values', 'returns',
                             'input_ids', 'response_length', 'prompt_length']

        experience_count = self.rl_config.critic_update_dispatch_size
        
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
        ray.get(self.td.update_metrics.remote(key='critic/lr', value=learning_rate))
        sorted_indexes = self.get_dp_range_indexes(
            experience_count,
            use_vllm=False
            ) if self.rl_config.guarantee_order else None
        critic_update_profiler = profiler_start(
            self.profiler_config, 
            role="critic_update",
            profiler_iteration=self.prof_iteration
        )
        MsProbe.debugger_start(self.model[0], tag='critic_update')
        start_time_defined = False

        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                self.megatron_config.tensor_model_parallel_size,
                self.megatron_config.context_parallel_size,
                self.megatron_config.context_parallel_algo,
                indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None,
                get_n_samples=False
            )
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:
                metrics = self.critic.update_critic(batch_data, kl_ctrl)

                self.args.consumed_train_samples += self.megatron_config.global_batch_size // self.rl_config.n_samples_per_prompt
                self.num_floating_point_operations_so_far += num_floating_point_operations(self.args,
                                                                                           self.megatron_config.global_batch_size)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True) and self.parallel_state.get_tensor_model_parallel_rank() == 0 and self.parallel_state.get_context_parallel_rank() == 0:
                    ray.get(self.td.update_metrics.remote(value=metrics, cumulate=True))
                    ray.get(
                        self.td.update_metrics.remote(
                            "timing/update_critic",
                            value=[round(time.time(), 4), round(start_time, 4)],
                            cumulate=True
                        )
                    )
                    
        self.critic_offloader.offload_optimizer()
        self.critic_offloader.offload_grad()
        self.critic_offloader.offload_param()
        self.empty_cache()

        self.iteration += 1
        profiler_step(critic_update_profiler)
        MsProbe.debugger_stop(tag='critic_update')
        MsProbe.step()
        self.prof_iteration += 1
        profiler_step(self.critic_profiler)
        logger.info("finish critic update")

    def save_ckpt(self, iteration: int):
        self.critic_offloader.onload_optimizer()
        self.critic_offloader.onload_grad()
        self.critic_offloader.onload_param()
        self.save_checkpoint(iteration, self.model, self.optimizer, self.opt_param_scheduler,
                             self.num_floating_point_operations_so_far)
        self.critic_offloader.offload_optimizer()
        self.critic_offloader.offload_grad()
        self.critic_offloader.offload_param()
        self.empty_cache()

    def _build_model_optimizer(self):
        critic_module, optimizer, opt_param_scheduler = self.setup_model_and_optimizer(
            self.model_provider, self.model_type.encoder_or_decoder)

        self.iteration = self.get_iteration()

        return critic_module, optimizer, opt_param_scheduler

    def _set_no_sync_func(self):
        config = get_attr_wrapped_model(self.model[0], 'config', allow_none=False)

        config.grad_scale_func = self.optimizer.scale_loss

        if isinstance(self.model[0], self.distributed_data_parallel) and self.megatron_config.overlap_grad_reduce:
            if config.no_sync_func is not None:
                raise ValueError('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                    'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.model]
            if len(self.model) == 1:
                config.no_sync_func = config.no_sync_func[0]

        config.finalize_model_grads_func = self.finalize_model_grads


@ray.remote(resources={"NPU": 0.3})
class CriticWorker(CriticWorkerBase):
    pass