# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
import dataclasses
import copy
from typing import Callable

import ray
from torch import nn
import torch
from transformers import AutoConfig

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.utils.optimizer_module import OptimizerConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.models.actor_rollout_hybrid import ActorRolloutHybrid
from mindspeed_rl.models.rollout.vllm_engine import VLLMInferEngine
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronShardingManager, MegatronOffLoader
from mindspeed_rl.utils.utils import num_floating_point_operations
from mindspeed_rl.utils.pad_process import remove_padding_and_split_to_list, truncate_rows


@ray.remote(resources={"NPU": 0.7})
class ActorHybridWorker(BaseWorker):
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
        super().__init__(
            megatron_config,
            rl_config,
            generate_config,
            model_provider=model_provider,
            initialize_func=initialize_func,
            tokenizer=tokenizer,
            get_megatron_module=get_megatron_module,
            **kwargs
        )

        self.num_floating_point_operations_so_far = 0
        self.actor_hybrid = None
        self.megatron_offloader = None

    def initialize(self):
        self.setup_distributed_rank()
        self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()
        self.megatron_offloader = MegatronOffLoader(self.optimizer, self.model)
        if self.generate_config.offload_train_optimizer:
            self.megatron_offloader.offload_optimizer()
        if self.generate_config.offload_train_grad:
            self.megatron_offloader.offload_grad()
        if self.generate_config.offload_train_param:
            self.megatron_offloader.offload_train_param()

        self.inference_model = self._build_rollout()
        self.sharding_manager = self._build_sharding_manager()

        self.actor_hybrid = ActorRolloutHybrid(
            self.model,
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
            micro_batch_size=self.megatron_config.micro_batch_size
        )
        self.empty_cache()

    def init_transfer_dock(self, td):
        self.td = td
        self.empty_cache()

    def get_iteration(self):
        return self.megatron_config.iteration

    def get_consumed_train_samples(self):
        return self.args.consumed_train_samples

    def update(self, kl_ctrl=None):
        start_time = time.time()
        experience_consumer_stage = 'actor_train'
        experience_colums = ['responses', 'advantages', 'old_log_prob',
                             'ref_log_prob', 'input_ids', 'response_length', 'prompt_length']
        experience_count = self.megatron_config.global_batch_size // self.rl_config.n_samples_per_prompt // self.parallel_state.get_data_parallel_world_size()

        #get lr
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
        ray.get(self.td.update_metrics.remote(key='grpo/lr', value=learning_rate)) 

        while self.all_consumed(experience_consumer_stage) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(experience_consumer_stage, experience_colums,
                                                                  experience_count, self.rl_config.n_samples_per_prompt,
                                                                  self.megatron_config.tensor_model_parallel_size)
            if batch_data and index:
                metrics = self.actor_hybrid.update_actor(batch_data, kl_ctrl)
                self.empty_cache()
                self.args.consumed_train_samples += self.megatron_config.global_batch_size // self.rl_config.n_samples_per_prompt
                self.num_floating_point_operations_so_far += num_floating_point_operations(self.args,
                                                                                           self.megatron_config.global_batch_size)
                if self.parallel_state.is_pipeline_last_stage() and self.parallel_state.get_tensor_model_parallel_rank() == 0:
                    ray.get(self.td.update_metrics.remote(value=metrics))
                    ray.get(
                        self.td.update_metrics.remote(
                            "timing/update", 
                            value=[round(time.time(), 4), round(start_time, 4)], 
                            cumulate=True
                        )
                    )

    def save_ckpt(self, iteration: int):
        self.save_checkpoint(iteration, self.model, self.optimizer, self.opt_param_scheduler,
                             self.num_floating_point_operations_so_far)

    def generate_sequences(self):
        start_time = time.time()
        experience_consumer_stage = 'actor_rollout'
        experience_colums = ['prompts', 'prompt_length']
        experience_count = self.rl_config.experience_count_actor // self.generate_config.data_parallel_size

        self.sharding_manager.reshard_to_infer_mode()
        pad_token_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod

        while self.all_consumed(experience_consumer_stage) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_colums,
                experience_count,
                n_samples_per_prompt=self.rl_config.n_samples_per_prompt,
                tp_size=self.megatron_config.tensor_model_parallel_size,
                use_vllm=True
            )
            if batch_data and index:
                indexes = list(range(0, experience_count * self.rl_config.n_samples_per_prompt,
                                     self.rl_config.n_samples_per_prompt))
                prompts_data = batch_data['prompts'][indexes]
                prompt_length_data = batch_data['prompt_length'][indexes]
                # preprocess, remove padding
                prompts = truncate_rows(prompts_data, prompt_length_data)
                prompts_list = [prompt.numpy().tolist() for prompt in prompts]

                # inference
                responses_pad_right = self.actor_hybrid.generate_sequences(copy.deepcopy(prompts_list))
                responses = remove_padding_and_split_to_list(responses_pad_right, self.tokenizer.eod, pad_token_id)

                responses_length = [torch.tensor([len(response)]) for response in responses]
                # copy prompts (from 1 to n_samples_per_prompt)
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
                ray.get(
                        self.td.update_metrics.remote(
                            "timing/rollout", 
                            value=[round(time.time(), 4), round(start_time, 4)], 
                            cumulate=True
                        )
                )
                self.collect_transfer_dock_data(outputs, index, self.rl_config.n_samples_per_prompt, use_vllm=True)
        self.sharding_manager.reshard_to_train_mode()
        self.empty_cache()

    def compute_log_prob(self):
        experience_consumer_stage = 'actor_log_prob'
        experience_colums = ['input_ids', 'responses', 'response_length', 'prompt_length']
        experience_count = self.rl_config.experience_count_actor // self.parallel_state.get_data_parallel_world_size()

        while self.all_consumed(experience_consumer_stage) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(experience_consumer_stage, experience_colums,
                                                                  experience_count, self.rl_config.n_samples_per_prompt,
                                                                  tp_size=self.megatron_config.tensor_model_parallel_size)
            if batch_data and index:
                output, batch = self.actor_hybrid.compute_log_prob(batch_data)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    log_probs = torch.cat(output, dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                    log_probs = truncate_rows(log_probs, batch['response_length'])
                    output = {'old_log_prob': log_probs}
                    self.collect_transfer_dock_data(output, index, self.rl_config.n_samples_per_prompt)

        self.empty_cache()

    def _get_megatron_optimizer(
            self,
            model,
            no_wd_decay_cond=None,
            scale_lr_cond=None,
            lr_mult=1.0
    ):
        args = self.args
        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)

        optimizer = self.get_megatron_optimizer(config, model, no_wd_decay_cond,
                                                scale_lr_cond, lr_mult)
        opt_param_scheduler = self.get_optimizer_param_scheduler(optimizer)
        return optimizer, opt_param_scheduler

    def _build_model_optimizer(self):
        actor_module = self.get_model(self.model_provider, None)
        if isinstance(actor_module, nn.ModuleList):
            actor_module = [actor_module[0]]
        optimizer, opt_param_scheduler = self._get_megatron_optimizer(model=actor_module)

        # load checkpoint
        if self.megatron_config.load is not None or self.megatron_config.pretrained_checkpoint is not None:
            self.megatron_config.iteration, self.megatron_config.num_floating_point_operations_so_far = self.load_checkpoint(
                actor_module, optimizer, opt_param_scheduler)
        else:
            self.megatron_config.iteration = 0
            self.megatron_config.num_floating_point_operations_so_far = 0

        return actor_module, optimizer, opt_param_scheduler

    def _build_rollout(self):
        self.actor_model_config = AutoConfig.from_pretrained(
            self.megatron_config.tokenizer_name_or_path, trust_remote_code=self.generate_config.trust_remote_code)

        sampling_config = {"num_completions": self.rl_config.n_samples_per_prompt,
                           "best_of": self.rl_config.n_samples_per_prompt,
                           **self.generate_config.sampling_config}

        rollout = VLLMInferEngine(
            tokenizer_name_or_path=self.megatron_config.tokenizer_name_or_path,
            train_tensor_parallel_size=self.megatron_config.tensor_model_parallel_size,
            train_pipeline_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            train_expert_parallel_size=self.megatron_config.expert_model_parallel_size,
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
            trust_remote_code=self.generate_config.trust_remote_code
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
            tp_split_expert=self.rl_config.tp_split_expert,
            parallel_state=self.parallel_state,
            inference_engine=self.inference_model,
            optimizer=self.optimizer,
            optimizer_offload=self.generate_config.offload_train_optimizer,
            grad_offload=self.generate_config.offload_train_grad,
            train_param_offload=self.generate_config.offload_train_param,
            enable_validate=self.rl_config.enable_sharding_validate,
            megatron_offloader=self.megatron_offloader
        )
        return sharding_manager
