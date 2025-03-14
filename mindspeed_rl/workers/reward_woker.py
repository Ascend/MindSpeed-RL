# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Callable
from types import ModuleType

import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.models.reward import Reward
from mindspeed_rl.trainer.utils.compute_utils import get_last_reward
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker


@ray.remote(resources={"NPU": 0.1})
class RewardWorker(BaseWorker):

    """
    RewardWorker class. This class implements the worker logic for reward model training and inference.

    Args:
        megatron_config: MegatronConfig Configuration for Megatron-LM (e.g., model parallelism settings).
        rl_config: RLConfig Configuration for reinforcement learning (e.g., PPO settings).
        generate_config: GenerateConfig Configuration for generation/inference (e.g., vLLM settings).
        model_provider: Callable Function to provide the model instance.
        initialize_func: Callable Function to initialize the model and environment.
        parallel_state: ModuleType Module for managing parallel states (e.g., model and data parallelism).
        get_model: Callable = None Function to retrieve the model instance.
        load_checkpoint: Callable = None Function to load model checkpoints.
        get_args: Callable = None Function to retrieve runtime arguments.
       tokenizer: BaseTokenizer = None Object to retrieve the tokenizer.
        get_forward_backward_func: Callable = None Function to retrieve the forward-backward function for training.
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
            load_checkpoint: Callable = None,
            get_args: Callable = None,
            tokenizer: BaseTokenizer = None,
            get_forward_backward_func: Callable = None,
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
            load_checkpoint=load_checkpoint,
            get_args=get_args,
            tokenizer=tokenizer,
            get_forward_backward_func=get_forward_backward_func,
            **kwargs
        )
        self.reward = None

    def initialize(self):
        self.setup_distributed_rank()
        self.model = self.get_model(self.model_provider, self.model_type, wrap_with_ddp=False)

        if self.megatron_config.load is not None or self.megatron_config.pretrained_checkpoint is not None:
            self.megatron_config.iteration, self.megatron_config.num_floating_point_operations_so_far = self._load_checkpoint(
                self.model, None, None, strict=False)
        else:
            self.megatron_config.iteration = 0
            self.megatron_config.num_floating_point_operations_so_far = 0

        self.reward = Reward(
            self.model,
            beta=self.rl_config.beta,
            mini_batch_size=self.rl_config.mini_batch_size,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
        )

    def init_transfer_dock(self, td):
        self.td = td

    def compute_rm_score(self):
        experience_consumer_stage = 'reward_scores'
        experience_columns = ['input_ids', 'prompt_length', "responses", "response_length"]
        experience_count = self.megatron_config.micro_batch_size

        while not ray.get(self.td.all_consumed.remote(experience_consumer_stage)):
            data_loader, index = self.dispatch_transfer_dock_data(experience_consumer_stage, experience_columns,
                                                                  experience_count, self.rl_config.n_samples_per_prompt,
                                                                  tp_size=self.megatron_config.tensor_model_parallel_size)
            if data_loader and index:
                output, batch = self.reward.compute_rm_score(data_loader)
                if self.parallel_state.is_pipeline_last_stage():
                    rm_score = torch.cat(output, dim=0).squeeze(-1)  # (bs, seq_size)
                    rm_score = rm_score.gather(dim=1, index=batch['prompt_length'] + batch['response_length'] - 1).to(
                        torch.float32)
                    last_rewards = get_last_reward(
                        rm_score,
                        n_sample_batch=self.rl_config.n_samples_per_prompt
                    )
                    output = {'rm_scores': rm_score, 'token_level_rewards': last_rewards}
                self.collect_transfer_dock_data(output, index, self.rl_config.n_samples_per_prompt)
        self.empty_cache()
