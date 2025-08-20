# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
from typing import Callable
import logging as logger

import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.models.actor import Actor
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.trainer.utils.parallel_state import is_pipeline_last_stage, get_tensor_model_parallel_rank
from mindspeed_rl.utils.utils import mstx_timer_decorator
from mindspeed_rl.trainer.utils.mm_transfer_dock import unpack_mm_experience


class VitWorkerBase(BaseWorker):
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
        self.vit = None
        self.vit_model = None
        self.vit_manager = None

    def initialize(self):
        self.setup_distributed_rank()
        self.vit_model = self.get_model(self.model_provider, self.model_type, wrap_with_ddp=False)

        if self.megatron_config.load is not None or self.megatron_config.pretrained_checkpoint is not None:
            self.megatron_config.iteration, self.megatron_config.num_floating_point_operations_so_far = self.load_checkpoint(
                self.vit_model, None, None)
        else:
            self.megatron_config.iteration = 0
            self.megatron_config.num_floating_point_operations_so_far = 0

        if self.rl_config.colocate_actor_and_vit:
            self.vit_manager = MegatronOffLoader(self.vit_model, wrap_with_ddp=False)
            self.vit_manager.offload_param()

        self.vit = Actor(
            self.vit_model,
            optimizer=None,
            opt_param_scheduler=None,
            beta=self.rl_config.beta,
            mini_batch_size=self.rl_config.mini_batch_size,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            micro_batch_size=self.megatron_config.micro_batch_size,
        )

    def init_transfer_dock(self, td, mm_td, sampling_transfer_dock=None):
        self.td = td
        self.mm_td = mm_td
        self.sampling_transfer_dock = sampling_transfer_dock
        self.empty_cache()

    @mstx_timer_decorator
    def compute_image_embeds(self):
        if self.rl_config.colocate_actor_and_vit:
            start_onload_time = time.time()
            self.vit_manager.onload_param()
            end_onload_time = time.time()
            ray.get(
                self.td.update_metrics.remote(
                    "timing/onload",
                    value=[round(end_onload_time, 4), round(start_onload_time, 4)],
                    cumulate=True
                )
            )

        experience_consumer_stage = 'actor_image_embeds'
        experience_columns = ['input_ids']
        experience_count = self.rl_config.actor_image_embeds_dispatch_size
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
                                                                 get_n_samples=True)
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:
                indexes = list(range(0, experience_count, self.rl_config.n_samples_per_prompt))
                batch_data['input_ids'] = batch_data['input_ids'][indexes]
                output, batch = self.vit.compute_image_embeds(batch_data)
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
                            "timing/vit_image_emb", 
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                )

        parallel_state = get_parallel_state()
        use_vllm = False
        if is_pipeline_last_stage(parallel_state, use_vllm) and get_tensor_model_parallel_rank(parallel_state, use_vllm) == 0:
            vit_end_time = time.time()
            ray.get(
                    self.td.update_metrics.remote(
                        "end_time/vit",
                        value=[round(vit_end_time, 4)]
                    )
            )
        
        if self.rl_config.colocate_actor_and_vit:
            start_offload_time = time.time()
            self.vit_manager.offload_param()
            end_offload_time = time.time()
            ray.get(
                self.td.update_metrics.remote(
                    "timing/offload",
                    value=[round(end_offload_time, 4), round(start_offload_time, 4)],
                    cumulate=True
                )
            )

        logger.info("finish compute compute image embeds")
        self.empty_cache()


@ray.remote(resources={"NPU": 0.3})
class VitWorker(VitWorkerBase):
    pass
