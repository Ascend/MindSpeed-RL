# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
from typing import Callable

import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.models.reward import Reward
from mindspeed_rl.trainer.utils.compute_utils import get_last_reward
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.trainer.utils.parallel_state import is_pipeline_last_stage, get_tensor_model_parallel_rank, get_context_parallel_rank


class RewardWorkerBase(BaseWorker):
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
        self.reward = None

    def initialize(self):
        self.setup_distributed_rank()
        self.model = self.get_model(self.model_provider, self.model_type, wrap_with_ddp=False)

        if self.megatron_config.load is not None or self.megatron_config.pretrained_checkpoint is not None:
            self.megatron_config.iteration, self.megatron_config.num_floating_point_operations_so_far = self.load_checkpoint(
                self.model, None, None, strict=False)
        else:
            self.megatron_config.iteration = 0
            self.megatron_config.num_floating_point_operations_so_far = 0
            
        self.reward = Reward(
            self.model,
            megatron_config=self.megatron_config,
            beta=self.rl_config.beta,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            micro_batch_size=self.megatron_config.micro_batch_size,
            temperature=self.generate_config.sampling_config["temperature"],
            use_dynamic_bsz=self.rl_config.use_dynamic_bsz,
            max_packing_token_size=self.rl_config.ref_max_packing_token_size,
            dynamic_max_batch_size=self.rl_config.ref_dynamic_max_batch_size,
            use_remove_padding=self.rl_config.use_remove_padding,
            set_actual_seq_len=self.set_actual_seq_len,
            get_actual_seq_len=self.get_actual_seq_len,
            set_position_ids=self.set_position_ids,
            context_parallel_size=self.megatron_config.context_parallel_size
        )

    def init_transfer_dock(self, td, mm_td=None, sampling_transfer_dock=None, mm_sampling_transfer_dock=None):
        self.td = td
        self.mm_td = mm_td
        self.sampling_transfer_dock = sampling_transfer_dock
        self.mm_sampling_transfer_dock = mm_sampling_transfer_dock

    def compute_rm_score(self):
        experience_consumer_stage = 'reward_scores'
        experience_columns = ['input_ids', 'prompt_length', "responses", "response_length",
                              *self.megatron_config.dataset_additional_keys]

        experience_count = self.rl_config.reward_dispatch_size

        sorted_indexes = self.get_dp_range_indexes(experience_count,
                                                   use_vllm=False) if self.rl_config.guarantee_order else None

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
                                                                 get_n_samples=self.rl_config.partial_rollout_max_split > 1
                                                                 )
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
                ray.get(
                    self.td.update_metrics.remote(
                        "start_time/reward_model",
                        value=[round(start_time, 4)],
                        cumulate=True
                    )
                )
            if batch_data and index:
                output, batch = self.reward.compute_rm_score(batch_data)
                if self.parallel_state.is_pipeline_last_stage():
                    rm_score = torch.cat(output, dim=0).squeeze(-1)  # (bs, seq_size)
                    rm_score = rm_score.gather(dim=1, index=batch['prompt_length'] + batch['response_length'] - 1).to(
                        torch.float32)
                    last_rewards = get_last_reward(
                        rm_score,
                        n_sample_batch=self.rl_config.n_samples_per_prompt
                    )
                    output = {'rm_scores': rm_score}
                self.collect_transfer_dock_data(output, index)
                end_time = time.time()
                ray.get(
                    self.td.update_metrics.remote(
                        "timing/reward_model",
                        value=[round(end_time, 4), round(start_time, 4)],
                        cumulate=True
                    )
                )
        parallel_state = get_parallel_state()
        use_vllm = False
        if is_pipeline_last_stage(parallel_state, use_vllm) and get_tensor_model_parallel_rank(parallel_state, use_vllm) == 0 and self.parallel_state.get_context_parallel_rank() == 0:
            rwd_end_time = time.time()
            ray.get(
                    self.td.update_metrics.remote(
                        "end_time/reward_model",
                        value=[round(rwd_end_time, 4)]
                    )
            )


@ray.remote(resources={"NPU": 0.1})
class RewardWorker(RewardWorkerBase):
    pass
