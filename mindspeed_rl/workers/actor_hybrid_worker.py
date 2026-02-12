# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
import dataclasses
import copy
import gc
import uuid
from enum import Enum
from typing import Callable
import logging as logger

import numpy as np
import ray
import torch
from transformers import AutoConfig

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig
from mindspeed_rl.models.actor_rollout_hybrid import ActorRolloutHybrid
from mindspeed_rl.models.rollout.vllm_engine import VLLMInferEngine
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.utils.utils import MsProbe
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronShardingManager
from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader
from mindspeed_rl.utils.utils import (num_floating_point_operations, get_attr_wrapped_model, mstx_timer_decorator,
                                      profiler_start, profiler_step, is_multimodal, replace_torch_compile)
from mindspeed_rl.utils.pad_process import remove_padding_and_split_to_list, truncate_rows
from mindspeed_rl.utils.zmq_communication import (ZmqServer, ZmqClient, ZmqServerInfo, ZmqClientInfo,
                                                  ZMQ_ROLE_SERVER, ZMQ_ROLE_CLIENT)
from mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state import get_vllm_tp_group_ranks
from mindspeed_rl.trainer.utils.mm_transfer_dock import unpack_mm_experience
from mindspeed_rl.tools.utils.tool_registry import initialize_tools_from_config
from mindspeed_rl.tools.utils.tool_utils import process_response
from mindspeed_rl.tools.utils.tool_parser import ToolParser
from mindspeed_rl.trainer.utils.parallel_state import get_tensor_model_parallel_rank, \
    get_tensor_model_parallel_src_rank, get_tensor_model_parallel_group



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
        self.enable_partial_rollout = self.rl_config.partial_rollout_max_split > 1

    def initialize(self):
        self.setup_distributed_rank()
        self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()
        self._set_no_sync_func()
        self.actor_offloader = MegatronOffLoader(
            self.model,
            self.optimizer,
            megatron_config=self.megatron_config,
            distributed_optimizer=self.distributed_optimizer,
            float16_optimizer_with_float16_params=self.float16_optimizer_with_float16_params)

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

        if self.rl_config.multi_turn_enable:
            tool_list = initialize_tools_from_config(self.rl_config.tool_config_path)
            self.tool_map = {tool.name: tool for tool in tool_list}
            self.tool_parser = ToolParser.get_tool_parser(self.rl_config.tool_parser_format, self.tokenizer.tokenizer)
            self.tools_kwargs = {
                'max_tool_calls': self.rl_config.max_tool_calls,
                'max_parallel_calls': self.rl_config.max_parallel_calls,
                'max_total_response_length': self.rl_config.max_total_response_length,
                'max_tool_response_length': self.rl_config.max_tool_response_length,
                'tool_response_truncate_side': self.rl_config.tool_response_truncate_side
            }

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
            entropy_coeff=self.rl_config.entropy_coeff,
            kl_penalty=self.rl_config.kl_penalty,
            temperature=self.generate_config.sampling_config["temperature"],
            token_level_loss=self.rl_config.token_level_loss,
            clip_higher_enable=self.rl_config.clip_higher_enable,
            clip_ratio_low=self.rl_config.clip_ratio_low,
            clip_ratio_high=self.rl_config.clip_ratio_high,
            reuse_image_embeds=self.rl_config.reuse_image_embeds,
            use_remove_padding=self.rl_config.use_remove_padding,
            use_dynamic_bsz=self.rl_config.use_dynamic_bsz,
            actor_max_packing_token_size=self.rl_config.actor_max_packing_token_size,
            update_max_packing_token_size=self.rl_config.update_max_packing_token_size,
            actor_dynamic_max_batch_size=self.rl_config.actor_dynamic_max_batch_size,
            update_dynamic_max_batch_size=self.rl_config.update_dynamic_max_batch_size,
            set_actual_seq_len=self.set_actual_seq_len,
            get_actual_seq_len=self.get_actual_seq_len,
            set_position_ids=self.set_position_ids,
            context_parallel_size=self.megatron_config.context_parallel_size
        )
        self.empty_cache()
        self.actor_profiler = profiler_start(self.profiler_config, self.profiler_config.role)
        MsProbe.config_init(self.msprobe_config)

    def init_transfer_dock(self, td, mm_td=None, sampling_transfer_dock=None, mm_sampling_transfer_dock=None):
        self.td = td
        self.mm_td = mm_td
        self.sampling_transfer_dock = sampling_transfer_dock
        self.mm_sampling_transfer_dock = mm_sampling_transfer_dock
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
        if self.rl_config.multi_turn_enable:
            experience_columns.extend(['response_mask'])

        experience_count = self.rl_config.actor_update_dispatch_size
        
        if skip_actor_log_prob:
            experience_columns.remove('old_log_prob')

        #get lr
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
        ray.get(self.td.update_metrics.remote(key='actor/lr', value=learning_rate))
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
            batch_data, index = self.dispatch_transfer_dock_data(experience_consumer_stage,
                                                                 experience_columns,
                                                                 experience_count,
                                                                 self.megatron_config.tensor_model_parallel_size,
                                                                 self.megatron_config.context_parallel_size,
                                                                 self.megatron_config.context_parallel_algo,
                                                                 indexes=sorted_indexes.pop(
                                                                     0) if self.rl_config.guarantee_order else None,
                                                                 get_n_samples=self.enable_partial_rollout)
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
        self.empty_cache()

    def get_partial_rollout_stop_signal(self):
        if not self.enable_partial_rollout:
            return False
        td = self.sampling_transfer_dock if self.sampling_transfer_dock else self.td
        return ray.get(
            td.get_update_ready.remote(require_max_age_all_finished=self.rl_config.require_max_age_all_finished))

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
        if self.enable_partial_rollout:
            experience_columns.extend(['responses', 'response_length', 'age'])

        if self.enable_partial_rollout and (self.rl_config.async_engine or self.iteration == self.megatron_config.train_iters - 1):
            td = self.sampling_transfer_dock if self.sampling_transfer_dock else self.td
            incomplete_resp_num = ray.get(td.get_incomplete_response_num.remote())
            experience_count = int(np.ceil(incomplete_resp_num / self.generate_config.data_parallel_size))
        else:
            experience_count = self.rl_config.actor_rollout_dispatch_size

        pad_token_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod
        sorted_indexes = self.get_dp_range_indexes(experience_count,
                                                   use_vllm=True) if self.rl_config.guarantee_order else None

        actor_generate_profiler = profiler_start(self.profiler_config, role="actor_generate",
                                                 profiler_iteration=self.prof_iteration)
        MsProbe.debugger_start(self.inference_model.model, tag='actor_generate_sequences')

        start_time = time.time()
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True, is_generate=True) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                tp_size=self.megatron_config.tensor_model_parallel_size,
                cp_size=self.megatron_config.context_parallel_size,
                cp_algo=self.megatron_config.context_parallel_algo,
                indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None,
                use_vllm=True,
                get_n_samples=not self.enable_partial_rollout,
                enable_partial_rollout=self.enable_partial_rollout,
                is_generate=True
            )

            if batch_data and index:
                if self.rl_config.async_engine:
                    logger.info(f"do async generate process.")
                    self.async_generate_process(batch_data, index, pad_token_id)
                else:
                    self.sync_generate_process(batch_data, experience_count, index, pad_token_id)
            if self.enable_partial_rollout:
                torch.distributed.barrier()
        end_time = time.time()
        ray.get(
                self.td.update_metrics.remote(
                    "timing/rollout",
                    value=[round(end_time, 4), round(start_time, 4)],
                    cumulate=True
                )
        )

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

    def sync_generate_process(self, batch_data, experience_count, index, pad_token_id):
        if not self.enable_partial_rollout:
            indexes = list(range(0, experience_count, self.rl_config.n_samples_per_prompt))
            if self.rl_config.reuse_image_embeds:
                prompts_data = batch_data['input_ids'][indexes]
                prompt_length_data = batch_data['input_ids_length'][indexes]
                # preprocess, remove padding
                prompts = truncate_rows(prompts_data, prompt_length_data, left_pad=True)
            else:
                prompts_data = batch_data['prompts'][indexes]
                prompt_length_data = batch_data['prompt_length'][indexes]
                # preprocess, remove padding
                prompts = truncate_rows(prompts_data, prompt_length_data)
            prompts_list = [prompt.numpy().tolist() for prompt in prompts]
        else:
            prompts_data = batch_data['prompts']
            prompt_length_data = batch_data['prompt_length']
            responses = batch_data['responses']
            responses_length_partial = batch_data['response_length']
            responses_partial = truncate_rows(responses, responses_length_partial)
            prompts = truncate_rows(prompts_data, prompt_length_data)
            prompts_for_vllm = [torch.cat(
                                    (prompt, response), dim=0) for prompt, response in
                                zip(prompts, responses_partial)]
            prompts_list = [prompt.numpy().tolist() for prompt in prompts_for_vllm]
        if self.enable_partial_rollout:
            max_tokens = self.generate_config.sampling_config["max_tokens"] // self.rl_config.partial_rollout_max_split
            responses_pad_right = self.actor_hybrid.generate_sequences(copy.deepcopy(prompts_list),
                                                                       max_tokens=max_tokens, n=1,
                                                                       extra_info=batch_data)
        else:
            responses_pad_right = self.actor_hybrid.generate_sequences(copy.deepcopy(prompts_list),
                                                                       extra_info=batch_data)

        responses = remove_padding_and_split_to_list(responses_pad_right, self.tokenizer.eod, pad_token_id)

        if is_multimodal():
            prompts_data = batch_data['input_ids'][indexes].cpu().unbind()
        else:
            prompts_data = prompts

        if self.enable_partial_rollout:
            new_responses = []
            for response_partial, response in zip(responses_partial, responses):
                new_resp = torch.cat((response_partial, response), dim=0)
                test_resp = new_resp >= self.tokenizer.vocab_size
                if test_resp.sum() > 0:
                    new_resp[test_resp] = 0
                new_responses.append(new_resp)
            responses = new_responses
        else:
            prompts = []
            for prompt in prompts_data:
                for _ in range(self.rl_config.n_samples_per_prompt):
                    prompts.append(copy.deepcopy(prompt))

        responses_length = [torch.tensor([len(response)]) for response in responses]

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

        if self.enable_partial_rollout:
            finish_status = [torch.tensor([0])] * len(responses_length)
            for idx, _ in enumerate(responses):
                if responses[idx][-1] == self.tokenizer.eod or \
                        (prompt_length_data[idx][0] + responses_length[
                            idx][0] >= self.generate_config.max_model_len) or responses_length[
                    idx][0] >= self.generate_config.sampling_config["max_tokens"]:
                    finish_status[idx] = torch.tensor([1])
            outputs["rollout_completed"] = finish_status

        self.collect_transfer_dock_data(outputs, index, use_vllm=True, is_generate=True, sync=is_multimodal())
        MsProbe.save_data({"responses": responses, "prompts": prompts})

    def async_generate_process(self, batch_data, index, pad_token_id):
        self.actor_hybrid.inference_actor.init_cache_engine()
        prompts_data = batch_data['prompts']
        prompt_length_data = batch_data['prompt_length']
        prompts = truncate_rows(prompts_data, prompt_length_data)
        if self.rl_config.multi_turn_enable:
            self.async_multi_turn_rollout(prompts, index)
            return
        if self.enable_partial_rollout:
            responses = batch_data['responses']
            responses_length_partial = batch_data['response_length']
            responses_partial = truncate_rows(responses, responses_length_partial)
            prompts_for_vllm = [torch.cat((prompt, response), dim=0) for prompt, response in zip(prompts, responses_partial)]
            prompts_list = [prompt.numpy().tolist() for prompt in prompts_for_vllm]
        else:
            prompts_list = [prompt.numpy().tolist() for prompt in prompts]
        if self.enable_partial_rollout:
            response_generator = self.actor_hybrid.generate_sequences(
                copy.deepcopy(prompts_list),
                indexes=index,
                n=1,
                async_engine=True,
                stop_singal_func=self.get_partial_rollout_stop_signal,
            )
        else:
            response_generator = self.actor_hybrid.generate_sequences(
                copy.deepcopy(prompts_list),
                indexes=index,
                n=1,
                async_engine=True,
            )

        for samples, idx_output in response_generator:
            prompts, responses, log_probs = samples
            responses = remove_padding_and_split_to_list(responses, self.tokenizer.eod, pad_token_id)

            remove_input_ids = False
            if self.enable_partial_rollout and len(responses[0]) == 1:
                remove_input_ids = True

            if self.enable_partial_rollout:
                responses_partial_new = []
                prompt_length_new = []
                for idx in range(len(responses)):
                    iidx = index.index(idx_output[idx])
                    responses_partial_new.append(responses_partial[iidx])
                    prompt_length_new.append(prompt_length_data[iidx])

                new_responses = []
                for response_partial, response in zip(responses_partial_new, responses):
                    new_resp = torch.cat((response_partial, response), dim=0)
                    test_resp = new_resp >= self.tokenizer.vocab_size
                    if test_resp.sum() > 0:
                        new_resp[test_resp] = 0
                    new_responses.append(new_resp)
                responses = new_responses
            responses_length = [torch.tensor([len(response)]) for response in responses]

            input_ids_list = []
            for prompt, response in zip(prompts, responses):
                input_ids_list.append(torch.cat((prompt, response), dim=0))

            outputs = {
                'responses': responses,
                'input_ids': input_ids_list,
                'response_length': responses_length
            }
            if remove_input_ids:
                outputs.pop("input_ids")

            if self.enable_partial_rollout:
                finish_status = [torch.tensor([0])] * len(responses_length)
                for idx, _ in enumerate(responses):
                    if responses[idx][-1] == self.tokenizer.eod or \
                            prompt_length_new[idx][0].to('cpu') + responses_length[
                        idx][0] >= self.generate_config.max_model_len or responses_length[
                        idx][0] >= self.generate_config.sampling_config["max_tokens"]:
                        finish_status[idx] = torch.tensor([1])
                outputs["rollout_completed"] = finish_status
            self.collect_transfer_dock_data(outputs, idx_output, use_vllm=True, is_generate=True, sync=is_multimodal())
            MsProbe.save_data({"responses": responses, "prompts": prompts})
        self.actor_hybrid.inference_actor.free_cache_engine()

    @torch.no_grad()
    def async_multi_turn_rollout(self, prompts, indexes):
        vocab_size = self.tokenizer.vocab_size
        use_vllm = True
        rank_flg = get_tensor_model_parallel_rank(self.parallel_state, use_vllm) == 0
        prompts_list = [prompt.numpy().tolist() for prompt in prompts]
        response_mask_list = [[] for _ in prompts]
        tool_call_num_list = [torch.tensor([0]) for _ in prompts]

        sampling_params = self.inference_model.sampling_params
        sampling_params.n = 1
        for i, prompt_ids in enumerate(prompts_list):
            request_id = f"req_{indexes[i]}_{uuid.uuid4().hex[:6]}"
            sampling_params.max_tokens = min((self.generate_config.max_model_len - len(prompt_ids)),
                                             self.rl_config.max_total_response_length)
            self.inference_model.engine.add_request(
                request_id=request_id,
                prompt={"prompt_token_ids": prompt_ids},
                params=sampling_params
            )

        while self.inference_model.engine.has_unfinished_requests():
            step_outputs = self.inference_model.engine.step()
            for step_output in step_outputs:
                if step_output.finished:
                    request_id = step_output.request_id
                    idx_output = int(request_id.split("_")[1])
                    index = indexes.index(idx_output)
                    outputs = step_output.outputs
                    response_ids = []
                    for output in outputs:  # List[CompletionOutput], usually len == 1
                        response_ids += output.token_ids
                        break
                    for idx, response_id in enumerate(response_ids):
                        if response_id >= vocab_size:
                            response_ids[idx] = 0
                    response_mask_list[index] += [1.0] * len(response_ids)
                    prompts_list[index] += response_ids
                    data_map = {
                        'response_ids': response_ids,
                        'response_mask': response_mask_list[index],
                        'tool_call_num': tool_call_num_list[index],
                    }

                    tool_response_ids = self.get_tool_response(rank_flg, data_map)
                    if tool_response_ids:
                        response_mask_list[index] += [0.0] * len(tool_response_ids)
                        prompts_list[index] += tool_response_ids
                        tool_call_num_list[index] += 1
                        if len(prompts_list[index]) < self.generate_config.max_model_len:
                            sampling_params.max_tokens = min((self.generate_config.max_model_len - len(prompts_list[index])),
                                                             self.rl_config.max_total_response_length)
                            self.inference_model.engine.add_request(
                                request_id=request_id,
                                prompt={"prompt_token_ids": prompts_list[index]},
                                params=sampling_params
                            )
                            continue

                    prompts = prompts_list[index]
                    response_mask = response_mask_list[index]
                    response_ids = prompts[-len(response_mask):]
                    prompt_ids = prompts[: len(prompts) - len(response_mask)]

                    prompts = [torch.tensor(prompt_ids)]
                    responses = [torch.tensor(response_ids[:self.rl_config.max_total_response_length])]
                    response_mask = [torch.tensor(response_mask[:self.rl_config.max_total_response_length])]
                    tool_call_num = [tool_call_num_list[index]]
                    responses_length = [torch.tensor([len(response)]) for response in responses]
                    input_ids_list = []
                    for prompt, response in zip(prompts, responses):
                        input_ids_list.append(torch.cat((prompt, response), dim=0))

                    outputs = {
                        'responses': responses,
                        'input_ids': input_ids_list,
                        'response_length': responses_length,
                        'response_mask': response_mask,
                        'tool_call_num': tool_call_num,
                    }
                    self.collect_transfer_dock_data(outputs, [idx_output], use_vllm=True, is_generate=True,
                                                    sync=is_multimodal())
                    MsProbe.save_data({"responses": responses, "prompts": prompts})

        self.actor_hybrid.inference_actor.free_cache_engine()

    def get_tool_response(self, rank_flg, data_map, use_vllm=True):
        tool_response_ids_length = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.int64)
        if rank_flg:
            tool_response_ids = process_response(self.tool_map, self.tool_parser, self.tokenizer.tokenizer,
                                                 data_map, self.tools_kwargs)
            tool_response_ids_length = torch.tensor([len(tool_response_ids)], device=torch.cuda.current_device(),
                                                    dtype=torch.int64)

        torch.distributed.broadcast(
            tool_response_ids_length, get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
            group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
        )

        if not rank_flg:
            tool_response_ids = torch.empty(tool_response_ids_length[0], device=torch.cuda.current_device(),
                                            dtype=torch.int32)

        torch.distributed.broadcast(
            tool_response_ids.cuda(), get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
            group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
        )
        return tool_response_ids.cpu().tolist()

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
            use_vllm=False
        ) if self.rl_config.guarantee_order else None

        actor_compute_log_prob_profiler = profiler_start(
            self.profiler_config,
            role="actor_compute_log_prob",
            profiler_iteration=self.prof_iteration
        )

        MsProbe.debugger_start(self.model[0], tag='actor_compute_log_prob')

        start_time_defined = False
        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(experience_consumer_stage,
                                                                 experience_columns,
                                                                 experience_count,
                                                                 tp_size=self.megatron_config.tensor_model_parallel_size,
                                                                 cp_size=self.megatron_config.context_parallel_size,
                                                                 cp_algo=self.megatron_config.context_parallel_algo,
                                                                 indexes=sorted_indexes.pop(
                                                                     0) if self.rl_config.guarantee_order else None,
                                                                 get_n_samples=self.enable_partial_rollout)
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

    @mstx_timer_decorator
    def compute_image_embeds(self):
        experience_consumer_stage = 'actor_image_embeds'
        experience_columns = ['input_ids']
        experience_count = self.rl_config.actor_image_embeds_dispatch_size
        sorted_indexes = self.get_dp_range_indexes(experience_count,
                                                   use_vllm=False) if self.rl_config.guarantee_order else None

        actor_image_embeds_profiler = profiler_start(self.profiler_config, role="actor_image_embeds",
                                                     profiler_iteration=self.prof_iteration)
        MsProbe.debugger_start(self.model[0], tag='actor_image_embeds')
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
                get_n_samples=True
            )
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:
                indexes = list(range(0, experience_count, self.rl_config.n_samples_per_prompt))
                batch_data['input_ids'] = batch_data['input_ids'][indexes]
                output, batch = self.actor_hybrid.compute_image_embeds(batch_data)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    output = torch.cat(output, dim=0)
                    data = {
                                "vit_embeds": output.squeeze(1).cpu(),
                                "image_grid_thw": batch_data['image_grid_thw'],
                                "image_num": batch_data['image_num'],
                                "video_num": batch_data['video_num']
                            }

                    data = unpack_mm_experience(data)

                    output = {'vit_embeds': data['vit_embeds']}
                    index = [i // self.rl_config.n_samples_per_prompt for i in index[::self.rl_config.n_samples_per_prompt]]
                    self.collect_transfer_dock_mm_data(output, index)
                end_time = time.time()
                ray.get(
                        self.td.update_metrics.remote(
                            "timing/image_embeds",
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                )
                ray.get(
                        self.td.update_metrics.remote(
                            "end_time/image_embeds",
                            value=[round(end_time, 4)],
                            cumulate=True
                        )
                )
        profiler_step(actor_image_embeds_profiler)
        MsProbe.debugger_stop('actor_image_embeds')
        logger.info("finish compute_image_embeds")

    def _build_model_optimizer(self):
        actor_module, optimizer, opt_param_scheduler = self.setup_model_and_optimizer(
            self.model_provider, self.model_type.encoder_or_decoder)

        self.iteration = self.get_iteration()

        return actor_module, optimizer, opt_param_scheduler

    def _build_rollout(self):
        self.actor_model_config = AutoConfig.from_pretrained(
            self.megatron_config.tokenizer_name_or_path, trust_remote_code=self.generate_config.trust_remote_code)

        self.actor_model_config.lora_r = self.megatron_config.lora_r
        self.actor_model_config.lora_alpha = self.megatron_config.lora_alpha
        self.actor_model_config.lora_target_modules = self.megatron_config.lora_target_modules

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
            ascend_scheduler_config_enabled=self.generate_config.ascend_scheduler_config_enabled,
            enable_expert_parallel=self.generate_config.enable_expert_parallel,
            expert_map_path=self.generate_config.expert_map_path,
            eplb_token_collects=self.generate_config.eplb_token_collects,
            eplb_token_save_path=self.generate_config.eplb_token_save_path,
            max_num_batched_tokens=self.generate_config.max_num_batched_tokens,
            limit_mm_image_per_prompt=self.generate_config.limit_mm_image_per_prompt,
            limit_mm_video_per_prompt=self.generate_config.limit_mm_video_per_prompt
        )

        if self.rl_config.zmq_communication:
            # if not include these lines, tp and pp rank is 0
            torch.distributed.get_rank()
            vllm_dp_groups = get_vllm_tp_group_ranks()
            if vllm_dp_groups is None:
                raise ValueError("vllm dp groups is None")

            from vllm.distributed import parallel_state as vpu
            if vpu.get_tensor_model_parallel_rank() == 0 and \
                vpu.get_pipeline_model_parallel_group().rank_in_group == 0:
                server_info = ZmqServerInfo()
                server_info.global_rank = self._rank
                server_info.dp_world_size = (vpu.get_tensor_model_parallel_world_size() *
                                             vpu.get_pp_group().world_size)
                server_info.ip_addr = ray._private.services.get_node_ip_address().strip("[]")
                server_info.register_port = self._get_free_port()
                server_info.publisher_port = self._get_free_port()
                server_info.reliability_port = self._get_free_port()
                server_info.use_vllm = True
                self.zmq_role = ZMQ_ROLE_SERVER
                self.zmq_server_vllm = ZmqServer(server_info, vpu)
            else:
                client_info = ZmqClientInfo()
                client_info.global_rank = self._rank
                client_info.use_vllm = True
                self.zmq_role = ZMQ_ROLE_CLIENT
                self.zmq_client_vllm = ZmqClient(client_info, vpu)

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
            share_backbone=self.rl_config.share_backbone,
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