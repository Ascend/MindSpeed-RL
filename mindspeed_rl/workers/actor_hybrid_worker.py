# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import dataclasses
import copy
from types import ModuleType
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
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronShardingManager
from mindspeed_rl.utils.utils import num_floating_point_operations
from mindspeed_rl.utils.pad_process import remove_padding_and_split_to_list


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
        parallel_state: ModuleType Module for managing parallel states (e.g., model and data parallelism).
        get_model: Callable = None Function to retrieve the model instance.
        get_megatron_optimizer: Callable = None Function to retrieve the Megatron optimizer.
        get_optimizer_param_scheduler: Callable = None Function to retrieve the optimizer parameter scheduler.
        load_checkpoint: Callable = None Function to load model checkpoints.
        save_checkpoint: Callable = None Function to save model checkpoints.
        get_args: Callable = None Function to retrieve runtime arguments.
        get_tokenizer: Callable = None Function to retrieve the tokenizer.
        get_forward_backward_func: Callable = None Function to retrieve the forward-backward function for training.
        distributed_data_parallel_config: Callable = None Configuration for distributed data parallelism.
        local_ddp: Callable = None Function for local distributed data parallelism.
        unwrap_model: Callable = None Function to unwrap the model for distributed training.
        float16_module: Callable = None Function to handle float16 precision modules.
        **kwargs: Additional parameters for base class argument passing.
    """

    def __init__(
            self,
            megatron_config: MegatronConfig,
            rl_config: RLConfig,
            generate_config: GenerateConfig,
            model_provider: Callable,
            initialize_func: Callable,
            parallel_state: ModuleType,
            get_model: Callable = None,
            get_megatron_optimizer: Callable = None,
            get_optimizer_param_scheduler: Callable = None,
            load_checkpoint: Callable = None,
            save_checkpoint: Callable = None,
            get_args: Callable = None,
            get_tokenizer: Callable = None,
            get_forward_backward_func: Callable = None,
            distributed_data_parallel_config: Callable = None,
            local_ddp: Callable = None,
            unwrap_model: Callable = None,
            float16_module: Callable = None,
            **kwargs
    ):
        super().__init__(
            megatron_config,
            rl_config,
            generate_config,
            model_provider=model_provider,
            initialize_func=initialize_func,
            parallel_state=parallel_state,
            get_model=get_model,
            get_megatron_optimizer=get_megatron_optimizer,
            get_optimizer_param_scheduler=get_optimizer_param_scheduler,
            load_checkpoint=load_checkpoint,
            save_checkpoint=save_checkpoint,
            get_args=get_args,
            get_tokenizer=get_tokenizer,
            get_forward_backward_func=get_forward_backward_func,
            **kwargs
        )

        self.float16_module = float16_module
        self.unwrap_model = unwrap_model
        self.local_ddp = local_ddp
        self.distributed_data_parallel_config = distributed_data_parallel_config
        self.num_floating_point_operations_so_far = 0
        self.actor_hybrid = None

    def initialize(self):
        self.setup_distributed_rank()
        self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()

        self.inference_model = self._build_rollout()
        self.sharding_manager = self._build_sharding_manager()

        self.actor_hybrid = ActorRolloutHybrid(
            self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            inference_model=self.inference_model,
            sharding_manager=self.sharding_manager,
            beta=self.rl_config.beta,
            mini_batch_size=self.rl_config.mini_batch_size // self.parallel_state.get_data_parallel_world_size(),
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            clip_ratio=self.rl_config.clip_ratio,
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
        experience_consumer_stage = 'actor_train'
        experience_colums = ['responses', 'advantages', 'old_log_prob',
                             'ref_log_prob', 'input_ids', 'response_length', 'prompt_length']
        experience_count = self.megatron_config.global_batch_size // self.parallel_state.get_data_parallel_world_size()

        while not ray.get(self.td.all_consumed.remote(experience_consumer_stage)):
            data_loader, index = self.dispatch_transfer_dock_data(experience_consumer_stage, experience_colums,
                                                                  experience_count, self.rl_config.n_samples_per_prompt,
                                                                  self.megatron_config.tensor_model_parallel_size)
            if data_loader and index:
                metrics = self.actor_hybrid.update_actor(data_loader, kl_ctrl)
                self.empty_cache()
                self.args.consumed_train_samples += self.megatron_config.global_batch_size
                self.num_floating_point_operations_so_far += num_floating_point_operations(self.args,
                                                                                           self.megatron_config.global_batch_size)
                if self.parallel_state.is_pipeline_last_stage() and self.parallel_state.get_tensor_model_parallel_rank() == 0:
                    return metrics

    def save_checkpoint(self, iteration: int):
        self._save_checkpoint(iteration, self.model, self.optimizer, self.opt_param_scheduler,
                              self.num_floating_point_operations_so_far)

    def generate_sequences(self):
        experience_consumer_stage = 'actor_rollout'
        experience_colums = ['prompts', 'prompt_length']
        experience_count = self.megatron_config.micro_batch_size

        self.sharding_manager.reshard_to_infer_mode()
        tokenizer = self.get_tokenizer()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id

        while not ray.get(self.td.all_consumed.remote(experience_consumer_stage)):
            data_loader, index = self.dispatch_transfer_dock_data(experience_consumer_stage, experience_colums,
                                                                  experience_count,
                                                                  tp_size=self.megatron_config.tensor_model_parallel_size,
                                                                  use_vllm=True)
            if data_loader and index:
                batch_data = next(iter(data_loader))

                # preprocess, remove padding
                prompts = self.truncate_rows(batch_data['prompts'], batch_data['prompt_length'])
                prompts_list = [prompt.numpy().tolist() for prompt in prompts]

                # inference
                responses_pad_right = self.actor_hybrid.generate_sequences(copy.deepcopy(prompts_list))
                responses = remove_padding_and_split_to_list(responses_pad_right, tokenizer.eos_token_id, pad_token_id)

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
                self.collect_transfer_dock_data(outputs, index, self.rl_config.n_samples_per_prompt, use_vllm=True)
        self.sharding_manager.reshard_to_train_mode()
        self.empty_cache()

    def compute_log_prob(self):
        experience_consumer_stage = 'actor_log_prob'
        experience_colums = ['input_ids', 'responses', 'response_length', 'prompt_length']
        experience_count = self.megatron_config.micro_batch_size

        while not ray.get(self.td.all_consumed.remote(experience_consumer_stage)):
            data_loader, index = self.dispatch_transfer_dock_data(experience_consumer_stage, experience_colums,
                                                                  experience_count, self.rl_config.n_samples_per_prompt,
                                                                  tp_size=self.megatron_config.tensor_model_parallel_size)
            if data_loader and index:
                output, batch = self.actor_hybrid.compute_log_prob(data_loader)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    log_probs = torch.cat(output, dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                    log_probs = self.truncate_rows(log_probs, batch['response_length'])
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
            self.megatron_config.iteration, self.megatron_config.num_floating_point_operations_so_far = self._load_checkpoint(
                actor_module, optimizer, opt_param_scheduler)
        else:
            self.megatron_config.iteration = 0
            self.megatron_config.num_floating_point_operations_so_far = 0

        return actor_module, optimizer, opt_param_scheduler

    def _build_rollout(self):
        self.actor_model_config = AutoConfig.from_pretrained(
            self.megatron_config.tokenizer_name_or_path)

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
            enable_validate=self.rl_config.enable_sharding_validate
        )
        return sharding_manager
