# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
from types import ModuleType
from typing import Callable

import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.models.reference import Reference
from mindspeed_rl.utils.pad_process import truncate_rows
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker


@ray.remote(resources={"NPU": 0.3})
class ReferenceWorker(BaseWorker):
    """
    ReferenceWorker class. This class implements the worker logic for reference model inference.

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
        self.reference = None

    def initialize(self):
        self.setup_distributed_rank()
        self.model = self.get_model(self.model_provider, self.model_type, wrap_with_ddp=False)

        if self.megatron_config.load is not None or self.megatron_config.pretrained_checkpoint is not None:
            self.megatron_config.iteration, self.megatron_config.num_floating_point_operations_so_far = self.load_checkpoint(
                self.model, None, None)
        else:
            self.megatron_config.iteration = 0
            self.megatron_config.num_floating_point_operations_so_far = 0

        self.reference = Reference(
            self.model,
            beta=self.rl_config.beta,
            mini_batch_size=self.rl_config.mini_batch_size,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            micro_batch_size=self.megatron_config.micro_batch_size
        )

    def init_transfer_dock(self, td):
        self.td = td

    def compute_log_prob(self):
        start_time = time.time()
        experience_consumer_stage = 'ref_log_prob'
        experience_columns = ['input_ids', 'responses', 'response_length', 'prompt_length']
        experience_count = self.megatron_config.micro_batch_size

        while self.all_consumed(experience_consumer_stage) > 0:
            data_loader, index = self.dispatch_transfer_dock_data(experience_consumer_stage, experience_columns,
                                                                  experience_count, self.rl_config.n_samples_per_prompt,
                                                                  tp_size=self.megatron_config.tensor_model_parallel_size)
            if data_loader and index:
                output, batch = self.reference.compute_log_prob(data_loader)

                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    log_probs = torch.cat(output, dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                    log_probs = truncate_rows(log_probs, batch['response_length'])
                    output = {'ref_log_prob': log_probs}
                    ray.get(
                        self.td.update_metrics.remote(
                            "timing/reference_model", 
                            value=[round(time.time(), 4), round(start_time, 4)], 
                            cumulate=True
                        )
                    )
                    self.collect_transfer_dock_data(output, index, self.rl_config.n_samples_per_prompt)

        self.empty_cache()
