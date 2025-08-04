# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
from typing import Callable
import logging as logger

import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.models.reference import Reference
from mindspeed_rl.utils.pad_process import truncate_rows
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.trainer.utils.parallel_state import is_pipeline_last_stage, get_tensor_model_parallel_rank, get_context_parallel_rank
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.utils import mstx_timer_decorator, is_multimodal

logger = Loggers(__name__)


class ReferenceWorkerBase(BaseWorker):
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

    def init_transfer_dock(self, td, mm_td, sampling_transfer_dock=None):
        self.td = td
        self.mm_td = mm_td
        self.sampling_transfer_dock = sampling_transfer_dock

    @mstx_timer_decorator
    def compute_ref_log_prob(self):
        experience_consumer_stage = 'ref_log_prob'
        experience_columns = ['input_ids', 'responses', 'response_length', 'prompt_length']
        if is_multimodal():
            experience_columns.extend(['attention_mask', 'position_ids', 'input_ids_length'])
        experience_count = self.rl_config.ref_dispatch_size
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
                                                                 get_n_samples=self.rl_config.partial_rollout_max_split > 1)

            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
                ray.get(
                    self.td.update_metrics.remote(
                        "start_time/reference_model",
                        value=[round(start_time, 4)],
                        cumulate=True
                    )
                )
            if batch_data and index:
                output, batch = self.reference.compute_log_prob(batch_data)

                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    log_probs = torch.cat(output, dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                    log_probs = truncate_rows(log_probs, batch['response_length'])
                    output = {'ref_log_prob': log_probs}
                    self.collect_transfer_dock_data(output, index)
                    end_time = time.time()
                    ray.get(
                        self.td.update_metrics.remote(
                            "timing/reference_model",
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                    )

        parallel_state = get_parallel_state()
        use_vllm = False
        if is_pipeline_last_stage(parallel_state, use_vllm) and get_tensor_model_parallel_rank(parallel_state, use_vllm) == 0 and self.parallel_state.get_context_parallel_rank() == 0:
            ref_end_time = time.time()
            ray.get(
                    self.td.update_metrics.remote(
                        "end_time/reference",
                        value=[round(ref_end_time, 4)]
                    )
            )
        logger.info("finish compute ref log prob")


@ray.remote(resources={"NPU": 0.3})
class ReferenceWorker(ReferenceWorkerBase):
    pass
