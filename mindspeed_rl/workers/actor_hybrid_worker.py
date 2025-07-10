# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
import dataclasses
import copy
import gc
from enum import Enum
from typing import Callable
import logging as logger

import ray
from torch import nn
import torch
from transformers import AutoConfig

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.utils.optimizer_module import OptimizerConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig
from mindspeed_rl.models.actor_rollout_hybrid import ActorRolloutHybrid
from mindspeed_rl.models.rollout.vllm_engine import VLLMInferEngine
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.utils.utils import MsProbe
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronShardingManager, MegatronOffLoader
from mindspeed_rl.utils.utils import (num_floating_point_operations, get_attr_wrapped_model, mstx_timer_decorator,
                                      profiler_start, profiler_step, is_multimodal, replace_torch_compile)
from mindspeed_rl.utils.pad_process import remove_padding_and_split_to_list, truncate_rows


class ActorState(Enum):
    NONE = "none"
    INFER = "infer"


class ActorHybridWorkerBase(BaseWorker):
    """
    ActorHybridWorker class. This class implements the hybrid worker logic for training and inference.

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

        self.num_floating_point_operations_so_far = 0
        self.actor_hybrid = None
        self.actor_offloader = None
        self.state = ActorState.NONE
        self.actor_profiler = None
        self.prof_iteration = 1
        self.idx = 0

    def initialize(self):
        self.setup_distributed_rank()
        self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()
        self._set_no_sync_func()
        self.actor_offloader = MegatronOffLoader(self.model, self.optimizer)
        if self.generate_config.offload_train_optimizer:
            self.actor_offloader.offload_optimizer()
        if self.generate_config.offload_train_grad:
            self.actor_offloader.offload_grad()
        if self.generate_config.offload_train_param:
            self.actor_offloader.offload_param()
        with replace_torch_compile():
            self.inference_model = self._build_rollout()
        self.sharding_manager = self._build_sharding_manager()

        if self.generate_config.offload_train_param:
            self.actor_offloader.onload_param()

        self.actor_hybrid = ActorRolloutHybrid(
            self.model,
            megatron_config=self.megatron_config,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            inference_model=self.inference_model,
            sharding_manager=self.sharding_manager,
            beta=self.rl_config.beta,
            mini_batch_size_per_dp=self.rl_config.mini_batch_size
                                   // self.parallel_state.get_data_parallel_world_size(),
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            clip_ratio=self.rl_config.clip_ratio,
            micro_batch_size=self.megatron_config.micro_batch_size,
            use_dynamic_bsz=self.rl_config.use_dynamic_bsz,
            max_packing_token_size=self.rl_config.max_packing_token_size,
            dynamic_max_batch_size=self.rl_config.dynamic_max_batch_size,
            use_remove_padding=self.rl_config.use_remove_padding,
            set_actual_seq_len=self.set_actual_seq_len,
            get_actual_seq_len=self.get_actual_seq_len,
            set_position_ids=self.set_position_ids,
            context_parallel_size=self.megatron_config.context_parallel_size,
            entropy_coeff=self.rl_config.entropy_coeff,
            kl_penalty=self.rl_config.kl_penalty,
            temperature=self.generate_config.sampling_config["temperature"],
            token_level_loss=self.rl_config.token_level_loss,
            clip_higher_enable=self.rl_config.clip_higher_enable,
            clip_ratio_low=self.rl_config.clip_ratio_low,
            clip_ratio_high=self.rl_config.clip_ratio_high
        )
        self.empty_cache()
        self.actor_profiler = profiler_start(self.profiler_config, self.profiler_config.role)
        MsProbe.config_init(self.msprobe_config)

    def init_transfer_dock(self, td, mm_td, sampling_transfer_dock=None):
        self.td = td
        self.mm_td = mm_td
        self.sampling_transfer_dock = sampling_transfer_dock
        self.empty_cache()

    def get_iteration(self):
        return self.args.iteration

    def get_consumed_train_samples(self):
        return self.args.consumed_train_samples

    def enter_infer_mode(self):
        if self.state == ActorState.INFER:
            return

        start_time = time.time()
        self.sharding_manager.enter_infer_mode()
        self.state = ActorState.INFER
        end_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_enter_infer",
                value=[end_time - start_time],
                cumulate=True
            )
        )

    def exit_infer_mode(self):
        if self.state != ActorState.INFER:
            raise RuntimeError
        start_time = time.time()
        self.sharding_manager.exit_infer_mode()
        self.state = ActorState.NONE
        end_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_exit_infer",
                value=[end_time - start_time],
                cumulate=True
            )
        )

    @mstx_timer_decorator
    def update(self, kl_ctrl=None, skip_actor_log_prob=False):
        start_sharding_enter_train = time.time()
        self.sharding_manager.enter_train_mode()
        sharding_train_interval = time.time() - start_sharding_enter_train

        self.args.curr_iteration = self.iteration

        experience_consumer_stage = 'actor_train'

        if self.megatron_config.stage == "ray_dapo":
            experience_columns = ['responses', 'advantages', 'old_log_prob', 'input_ids', 'response_length', 'prompt_length']
        else:
            experience_columns = ['responses', 'advantages', 'old_log_prob', 'ref_log_prob', 'input_ids', 'response_length', 'prompt_length']

        if is_multimodal():
            experience_columns.extend(['attention_mask', 'position_ids'])

        experience_count = self.rl_config.actor_update_dispatch_size

        if skip_actor_log_prob:
            experience_columns.remove('old_log_prob')

        #get lr
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
        ray.get(self.td.update_metrics.remote(key='param/lr', value=learning_rate))
        sorted_indexes = self.get_dp_range_indexes(
            experience_count,
            use_vllm=False,
            assign_batch_size=experience_count
        ) if self.rl_config.guarantee_order else None

        actor_update_profiler = profiler_start(
            self.profiler_config,
            role="actor_update",
            profiler_iteration=self.prof_iteration
        )

        MsProbe.debugger_start(self.model[0], tag='actor_update')

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
                metrics = self.actor_hybrid.update_actor(batch_data, kl_ctrl)

                self.args.consumed_train_samples += self.megatron_config.global_batch_size // self.rl_config.n_samples_per_prompt
                self.num_floating_point_operations_so_far += num_floating_point_operations(self.args,
                                                                                           self.megatron_config.global_batch_size)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True) and self.parallel_state.get_tensor_model_parallel_rank() == 0 and self.parallel_state.get_context_parallel_rank() == 0:
                    ray.get(self.td.update_metrics.remote(value=metrics, cumulate=True))
                    ray.get(
                        self.td.update_metrics.remote(
                            "timing/update",
                            value=[round(time.time(), 4), round(start_time, 4)],
                            cumulate=True
                        )
                    )

        self.iteration += 1
        profiler_step(actor_update_profiler)
        MsProbe.debugger_stop(tag='actor_update')
        MsProbe.step()
        self.prof_iteration += 1
        start_sharding_exit_train = time.time()
        self.sharding_manager.exit_train_mode()
        sharding_train_interval += (time.time() - start_sharding_exit_train)
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_to_train",
                value=[sharding_train_interval],
                cumulate=True
            )
        )
        profiler_step(self.actor_profiler)
        logger.info("finish actor update")

    def save_ckpt(self, iteration: int):
        self.sharding_manager.enter_train_mode()
        self.save_checkpoint(iteration, self.model, self.optimizer, self.opt_param_scheduler,
                             self.num_floating_point_operations_so_far)
        self.sharding_manager.exit_train_mode()

    @mstx_timer_decorator
    def generate_sequences(self):
        sharding_infer_interval = 0
        if not self.rl_config.filter_groups_enable:
            start_sharding_enter_infer = time.time()
            self.sharding_manager.enter_infer_mode()
            sharding_infer_interval = time.time() - start_sharding_enter_infer

        experience_consumer_stage = 'actor_rollout'
        experience_columns = ['prompts', 'prompt_length']
        if is_multimodal():
            experience_columns.extend(['input_ids', 'input_ids_length'])

        experience_count = self.rl_config.actor_rollout_dispatch_size

        pad_token_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod
        sorted_indexes = self.get_dp_range_indexes(experience_count,
                                                   use_vllm=True) if self.rl_config.guarantee_order else None

        actor_generate_profiler = profiler_start(self.profiler_config, role="actor_generate",
                                                 profiler_iteration=self.prof_iteration)
        MsProbe.debugger_start(self.inference_model.model, tag='actor_generate_sequences')

        start_time_defined = False
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                tp_size=self.megatron_config.tensor_model_parallel_size,
                cp_size=self.megatron_config.context_parallel_size,
                cp_algo=self.megatron_config.context_parallel_algo,
                indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None,
                use_vllm=True
            )
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:

                gc.collect()
                torch.cuda.empty_cache()

                if self.rl_config.async_engine:
                    logger.info(f"do async generate process.")
                    prompts_data = batch_data['prompts']
                    prompt_length_data = batch_data['prompt_length']
                    prompts = truncate_rows(prompts_data, prompt_length_data)
                    prompts_list = [prompt.numpy().tolist() for prompt in prompts]
                    self.async_generate_process(experience_count, index, pad_token_id, prompts_list, start_time)
                else:
                    self.sync_generate_process(batch_data, experience_count, index, pad_token_id, start_time)
        profiler_step(actor_generate_profiler)
        MsProbe.debugger_stop('actor_generate_sequences')

        self.idx += 1
        if not self.rl_config.filter_groups_enable:
            start_sharding_exit_infer = time.time()
            self.sharding_manager.exit_infer_mode()
            torch.cuda.empty_cache()
            sharding_infer_interval += (time.time() - start_sharding_exit_infer)

            ray.get(
                self.td.update_metrics.remote(
                    "timing/resharding_to_infer",
                    value=[sharding_infer_interval],
                    cumulate=True
                )
            )
            logger.info("finish generate_sequences")

    def sync_generate_process(self, batch_data, experience_count, index, pad_token_id, start_time):
        indexes = list(range(0, experience_count, self.rl_config.n_samples_per_prompt))
        prompts_data = batch_data['prompts'][indexes]
        prompt_length_data = batch_data['prompt_length'][indexes]
        # preprocess, remove padding
        prompts = truncate_rows(prompts_data, prompt_length_data)
        prompts_list = [prompt.numpy().tolist() for prompt in prompts]

        with replace_torch_compile():
            responses_pad_right = self.actor_hybrid.generate_sequences(copy.deepcopy(prompts_list), indexes,
                                                                    n_samples_per_prompt=self.rl_config.n_samples_per_prompt,
                                                                    async_engine=self.rl_config.async_engine,
                                                                    extra_info=batch_data)
        responses = remove_padding_and_split_to_list(responses_pad_right, self.tokenizer.eod, pad_token_id)
        responses_length = [torch.tensor([len(response)]) for response in responses]
        if is_multimodal():
            prompts_data = batch_data['input_ids'][indexes].cpu().unbind()
        else:
            prompts_data = prompts
        prompts = []
        for prompt in prompts_data:
            for _ in range(self.rl_config.n_samples_per_prompt):
                prompts.append(copy.deepcopy(prompt))
        input_ids_list = []
        for prompt, response in zip(prompts, responses):
            input_ids_list.append(torch.cat((prompt, response), dim=0))
        outputs = {
            'responses': responses,
            'input_ids': input_ids_list,
            'response_length': responses_length
        }
        if is_multimodal():
            outputs['prompt_length'] = batch_data['input_ids_length']
        self.collect_transfer_dock_data(outputs, index, use_vllm=True)
        end_time = time.time()
        MsProbe.save_data({"responses": responses, "prompts": prompts})
        ray.get(
            self.td.update_metrics.remote(
                "timing/rollout",
                value=[round(end_time, 4), round(start_time, 4)],
                cumulate=True
            )
        )

    def async_generate_process(self, experience_count, index, pad_token_id, prompts_list, start_time):
        # inference
        self.actor_hybrid.inference_actor.init_cache_engine()
        with replace_torch_compile():
            response_generator = self.actor_hybrid.generate_sequences(
                copy.deepcopy(prompts_list),
                indexes=index,
                max_tokens=self.generate_config.sampling_config["max_tokens"],
                n_samples_per_prompt=1,
                n=1,
                async_engine=True,
            )
        for samples, idx in response_generator:
            prompts, responses, log_probs = samples
            responses = remove_padding_and_split_to_list(responses, self.tokenizer.eod, pad_token_id)
            responses_length = [torch.tensor([len(response)]) for response in responses]

            input_ids_list = []
            for prompt, response in zip(prompts, responses):
                input_ids_list.append(torch.cat((prompt, response), dim=0))

            outputs = {
                'responses': responses,
                'input_ids': input_ids_list,
                'response_length': responses_length
            }
            self.collect_transfer_dock_data(outputs, idx, use_vllm=True)

        end_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/rollout",
                value=[round(end_time, 4), round(start_time, 4)],
                cumulate=True
            )
        )
        self.actor_hybrid.inference_actor.free_cache_engine()

    @mstx_timer_decorator
    def compute_log_prob(self):
        self.sharding_manager.enter_forward_mode()

        experience_consumer_stage = 'actor_log_prob'
        experience_columns = ['input_ids', 'responses', 'response_length', 'prompt_length']
        if is_multimodal():
            experience_columns.extend(['attention_mask', 'position_ids', 'input_ids_length'])
        experience_count = self.rl_config.actor_logprob_dispatch_size

        sorted_indexes = self.get_dp_range_indexes(
            experience_count,
            use_vllm=False,
            assign_batch_size=experience_count
        ) if self.rl_config.guarantee_order else None

        actor_compute_log_prob_profiler = profiler_start(
            self.profiler_config,
            role="actor_compute_log_prob",
            profiler_iteration=self.prof_iteration
        )

        MsProbe.debugger_start(self.model[0], tag='actor_compute_log_prob')

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
                output, batch = self.actor_hybrid.compute_log_prob(batch_data)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    log_probs = torch.cat(output, dim=0)
                    log_probs = log_probs.to(torch.float32)
                    log_probs = truncate_rows(log_probs, batch['response_length'])
                    output = {'old_log_prob': log_probs}
                    self.collect_transfer_dock_data(output, index)
                end_time = time.time()
                ray.get(
                        self.td.update_metrics.remote(
                            "timing/old_log_p",
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                )
                ray.get(
                        self.td.update_metrics.remote(
                            "end_time/old_log_p",
                            value=[round(end_time, 4)],
                            cumulate=True
                        )
                )
        self.sharding_manager.exit_forward_mode()
        torch.cuda.empty_cache()

        profiler_step(actor_compute_log_prob_profiler)
        MsProbe.debugger_stop('actor_compute_log_prob')
        logger.info("finish compute_log_prob")

    def _build_model_optimizer(self):
        actor_module, optimizer, opt_param_scheduler = self.setup_model_and_optimizer(
            self.model_provider, self.model_type.encoder_or_decoder)

        self.iteration = self.get_iteration()

        return actor_module, optimizer, opt_param_scheduler

    def _build_rollout(self):
        self.actor_model_config = AutoConfig.from_pretrained(
            self.megatron_config.tokenizer_name_or_path, trust_remote_code=self.generate_config.trust_remote_code)

        sampling_config = {"num_completions": self.rl_config.n_samples_per_prompt,
                           **self.generate_config.sampling_config}

        rollout = VLLMInferEngine(
            tokenizer_name_or_path=self.megatron_config.tokenizer_name_or_path,
            train_tensor_parallel_size=self.megatron_config.tensor_model_parallel_size,
            train_pipeline_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            train_expert_parallel_size=self.megatron_config.expert_model_parallel_size,
            train_context_parallel_size=self.megatron_config.context_parallel_size,
            infer_tensor_parallel_size=self.generate_config.infer_tensor_parallel_size,
            infer_pipeline_parallel_size=self.generate_config.infer_pipeline_parallel_size,
            infer_expert_parallel_size=self.generate_config.infer_expert_parallel_size,
            megatron_config=self.megatron_config,
            sampling_config=sampling_config,
            enable_prefix_caching=self.generate_config.enable_prefix_caching,
            num_scheduler_steps=self.generate_config.num_scheduler_steps,
            max_num_seqs=self.generate_config.max_num_seqs,
            max_model_len=self.generate_config.max_model_len,
            dtype=self.generate_config.dtype,
            gpu_memory_utilization=self.generate_config.gpu_memory_utilization,
            trust_remote_code=self.generate_config.trust_remote_code,
            enforce_eager=self.generate_config.enforce_eager,
            torchair_graph=self.generate_config.torchair_graph,
            enable_expert_parallel=self.generate_config.enable_expert_parallel,
            max_num_batched_tokens=self.generate_config.max_num_batched_tokens,
            limit_mm_image_per_prompt=self.generate_config.limit_mm_image_per_prompt,
            limit_mm_video_per_prompt=self.generate_config.limit_mm_video_per_prompt,
            )
        return rollout

    def _build_sharding_manager(self):
        # perform weight resharding between actor and rollout
        sharding_manager = MegatronShardingManager(
            megatron_model=self.model,
            model_config=self.actor_model_config,
            infer_tensor_parallel_size=self.generate_config.infer_tensor_parallel_size,
            infer_pipeline_parallel_size=self.generate_config.infer_pipeline_parallel_size,
            infer_expert_parallel_size=self.generate_config.infer_expert_parallel_size,
            num_layer_list=self.megatron_config.num_layer_list,
            moe_tp_extend_ep=self.megatron_config.moe_tp_extend_ep,
            parallel_state=self.parallel_state,
            inference_engine=self.inference_model,
            optimizer=self.optimizer,
            optimizer_offload=self.generate_config.offload_train_optimizer,
            grad_offload=self.generate_config.offload_train_grad,
            train_param_offload=self.generate_config.offload_train_param,
            enable_validate=self.rl_config.enable_sharding_validate,
            megatron_offloader=self.actor_offloader,
            noop_layers=self.megatron_config.noop_layers
        )
        return sharding_manager

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


@ray.remote(resources={"NPU": 0.7})
class ActorHybridWorker(ActorHybridWorkerBase):
    pass