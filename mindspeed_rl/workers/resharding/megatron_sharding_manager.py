# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Megatron Sharding Mananger:
Manager used to shard weight and offload/onload optimizer from training stage to inference stage
"""
import os
import vllm.distributed.parallel_state as ps

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.workers.resharding.vllm_weight_container import MegatronStyleVllmWeightContainer
from mindspeed_rl.workers.resharding.weight_adaptor import get_weight_adaptor
from mindspeed_rl.utils.utils import mstx_timer_decorator

logger = Loggers(
    name="vllm_engine_inference",
)


class MegatronShardingManager:

    def __init__(
            self,
            inference_engine,
            optimizer,
            optimizer_offload=False,
            grad_offload=False,
            train_param_offload=False,
            share_backbone=False,
            megatron_model=None,
            model_config=None,
            infer_tensor_parallel_size=None,
            infer_pipeline_parallel_size=None,
            infer_expert_parallel_size=None,
            num_layer_list=None,
            moe_tp_extend_ep=None,
            parallel_state=None,
            megatron_offloader=None,
            noop_layers=None
    ):
        """Megatron Sharding Manager initialization.

        Arguments:
            inference_engine (BaseInferEngine): Inference engine instance used for model execution.
            optimizer (MegatronOptimizer): Optimizer instance used for model training.
            optimizer_offload (bool): Whether to offload optimizer operations to a separate device.
            grad_offload (bool): whether to offload gradient computation to CPU during training.
            megatron_model (nn.Module or nn.ModuleList): Megatron model instance.
            model_config (MegatronConfig): Configuration for the model.
            infer_tensor_parallel_size (int): Tensor parallel size during inference.
            infer_pipeline_parallel_size (int): Pipeline parallel size during inference.
            infer_expert_parallel_size (int): Expert parallel size during inference.
            num_layer_list (str): a list of number of layers, seperated by comma; e.g., 4,4,4,4.
            moe_tp_extend_ep (bool): Controls whether expert model parameters are split across multiple GPUs.
            parallel_state (ModuleType): Megatron parallel state of the model.
        """
        self.inference_engine = inference_engine
        self.optimizer = optimizer
        self.train_model = megatron_model
        weight_adaptor = get_weight_adaptor(self.inference_engine.model.__class__.__name__)
        self.weight_adaptor = weight_adaptor(model_config)

        self.vllm_weight_container = MegatronStyleVllmWeightContainer(
            megatron_model=megatron_model,
            vllm_model=self.inference_engine.model,
            model_config=model_config,
            infer_tensor_parallel_size=infer_tensor_parallel_size,
            infer_pipeline_parallel_size=infer_pipeline_parallel_size,
            infer_expert_parallel_size=infer_expert_parallel_size,
            num_layer_list=num_layer_list,
            moe_tp_extend_ep=moe_tp_extend_ep,
            parallel_state=parallel_state,
            weight_adaptor=self.weight_adaptor,
            noop_layers=noop_layers,
            eplb_map=self.inference_engine.eplb_map,
            global_redundant_expert_num=self.inference_engine.global_redundant_expert_num,
            infer_local_num_experts=self.inference_engine.infer_local_num_experts
            )

        self.optimizer_offload = optimizer_offload
        self.grad_offload = grad_offload
        self.train_param_offload = train_param_offload
        self.share_backbone = share_backbone
        self.inference_engine.offload_model_weights()
        self.megatron_offloader = megatron_offloader

    @mstx_timer_decorator
    def offload_infer_params(self):
        infer_weight_buffers = self.vllm_weight_container.weight_buffers
        for buffer in infer_weight_buffers:
            buffer.destroy()

    @mstx_timer_decorator
    def onload_infer_params(self):
        infer_weight_buffers = self.vllm_weight_container.weight_buffers
        for buffer in infer_weight_buffers:
            buffer.rebuild()

    @mstx_timer_decorator
    def enter_infer_mode(self):
        """
        Before:
            Empty or with training param on NPU.

        After:
            Empty.

        Process:
            1. onload training param if needed
            2. onload inference param
            3. do resharding
            4. offload training param
        """

        self.onload_infer_params()

        infer_params = self.vllm_weight_container.get_infer_params()

        if self.train_param_offload:
            self.megatron_offloader.offload_param()
        self.inference_engine.sync_model_weights(infer_params, load_format='megatron')


    @mstx_timer_decorator
    def exit_infer_mode(self):
        """
        Before:
            With inference param on NPU.

        After:
            Empty.

        Process:
            1. offload inference param
        """
        self.inference_engine.offload_model_weights()
        self.offload_infer_params()

    @mstx_timer_decorator
    def enter_forward_mode(self):
        """
        Before:
            Empty.

        After:
            With training param on NPU.

        Process:
            1. onload training param
        """
        if self.train_param_offload and not self.share_backbone:
            self.megatron_offloader.onload_param()

    @mstx_timer_decorator
    def exit_forward_mode(self):
        """
        Before:
            Empty.

        After:
            With training param on NPU.

        Process:
            1. onload training param
        """
        if self.train_param_offload and not self.share_backbone:
            self.megatron_offloader.offload_param()

    @mstx_timer_decorator
    def enter_train_mode(self):
        """
        Before:
            With training param on NPU.

        After:
            With training param, optimizer and grad on NPU.

        Process:
            1. onload training optimizer
            2. onload training grad
        """
        if self.train_param_offload and not self.share_backbone:
            self.megatron_offloader.onload_param()
        if self.optimizer_offload:
            self.megatron_offloader.onload_optimizer()
        if self.grad_offload:
            self.megatron_offloader.onload_grad()

    @mstx_timer_decorator
    def exit_train_mode(self):
        """
        Before:
            With training param, optimizer and grad on NPU.

        After:
            With training param on NPU.

        Process:
            1. offload training optimizer
            2. offload training grad
        """
        if self.optimizer_offload:
            self.megatron_offloader.offload_optimizer()
        if self.grad_offload:
            self.megatron_offloader.offload_grad()
