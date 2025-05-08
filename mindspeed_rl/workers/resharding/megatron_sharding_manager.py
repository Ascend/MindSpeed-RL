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
from itertools import chain
from collections import defaultdict

import torch
import torch.distributed

from mindspeed_rl.workers.resharding.vllm_weight_container import MegatronStyleVllmWeightContainer
from mindspeed_rl.workers.resharding.weight_adaptor import get_weight_adaptor


class MegatronOffLoader:
    def __init__(self, megatron_model=None, optimizer=None, wrap_with_ddp=True):
        self.optimizer = optimizer
        self.model = megatron_model
        self.wrap_with_ddp = wrap_with_ddp

        self.tensor_to_cpu_states_map = dict()

    def offload_grad(self):
        for model in self.model:
            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                self.swap_tensors_to_host(buffer.grad_data, copy_data=False)

    def onload_grad(self):
        for model in self.model:
            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                self.swap_tensors_to_device(buffer.grad_data, copy_data=False)

    def offload_optimizer(self):
        for param_group in self.optimizer.optimizer.param_groups:
            for param in param_group['params']:
                param.data = param.data.to("cpu", non_blocking=False)
        self.optimizer.optimizer.state = self._move_to_device(self.optimizer.optimizer.state, "cpu")

    def onload_optimizer(self):
        for param_group in self.optimizer.optimizer.param_groups:
            for param in param_group['params']:
                param.data = param.data.to(torch.cuda.current_device(), non_blocking=False)
        self.optimizer.optimizer.state = self._move_to_device(self.optimizer.optimizer.state,
                                                              torch.cuda.current_device())

    def _move_to_device(self, data, device):
        if isinstance(data, defaultdict):
            return defaultdict(data.default_factory,
                               {key: self._move_to_device(value, device) for key, value in data.items()})
        elif isinstance(data, dict):
            return {key: self._move_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(device, non_blocking=False)
        else:
            return data

    def offload_param(self):
        if self.wrap_with_ddp:
            for model in self.model:
                for buffer in chain(model.buffers, model.expert_parallel_buffers):
                    self.swap_tensors_to_host(buffer.param_data)
        else:
            for item in self.model:
                item.to('cpu')

    def onload_param(self):
        if self.wrap_with_ddp:
            for model in self.model:
                for buffer in chain(model.buffers, model.expert_parallel_buffers):
                    self.swap_tensors_to_device(buffer.param_data)
        else:
            for item in self.model:
                item.to(torch.cuda.current_device())

    def swap_tensors_to_host(self, tensor, copy_data=True):
        if tensor not in self.tensor_to_cpu_states_map:
            self.tensor_to_cpu_states_map[tensor] = torch.empty_like(tensor, device='cpu')
        if tensor.storage().size() != 0:
            if copy_data:
                cpu_state = self.tensor_to_cpu_states_map[tensor]
                cpu_state.copy_(tensor, non_blocking=False)
            tensor.storage().resize_(0)

    def swap_tensors_to_device(self, tensor, copy_data=True):
        if tensor.storage().size() == 0:
            cpu_state = self.tensor_to_cpu_states_map[tensor]
            tensor.storage().resize_(cpu_state.storage().size())
            if copy_data:
                tensor.copy_(cpu_state, non_blocking=False)


class MegatronShardingManager:

    def __init__(
            self,
            inference_engine,
            optimizer,
            optimizer_offload=False,
            grad_offload=False,
            train_param_offload=False,
            enable_validate=False,
            megatron_model=None,
            model_config=None,
            infer_tensor_parallel_size=None,
            infer_pipeline_parallel_size=None,
            infer_expert_parallel_size=None,
            num_layer_list=None,
            moe_tp_extend_ep=None,
            parallel_state=None,
            megatron_offloader=None
    ):
        """Megatron Sharding Manager initialization.

        Arguments:
            inference_engine (BaseInferEngine): Inference engine instance used for model execution.
            optimizer (MegatronOptimizer): Optimizer instance used for model training.
            optimizer_offload (bool): Whether to offload optimizer operations to a separate device.
            grad_offload (bool): whether to offload gradient computation to CPU during training.
            enable_validate (bool): Whether to enable communication data validate.
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
            enable_validate=enable_validate)

        self.optimizer_offload = optimizer_offload
        self.grad_offload = grad_offload
        self.train_param_offload = train_param_offload
        self.enable_validate = enable_validate
        self.use_distributed_optimizer = self.optimizer.config.use_distributed_optimizer
        self.inference_engine.offload_model_weights()
        self.megatron_offloader = megatron_offloader

    def offload_infer_params(self):
        infer_weight_buffers = self.vllm_weight_container.weight_buffers
        for buffer in infer_weight_buffers:
            buffer.destroy()

    def onload_infer_params(self):
        infer_weight_buffers = self.vllm_weight_container.weight_buffers
        for buffer in infer_weight_buffers:
            buffer.rebuild()

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
        if self.train_param_offload:
            self.megatron_offloader.onload_param()

        self.onload_infer_params()

        infer_params = self.vllm_weight_container.get_infer_params()

        if self.train_param_offload:
            self.megatron_offloader.offload_param()

        self.inference_engine.sync_model_weights(infer_params, load_format='megatron')

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

    def enter_forward_mode(self):
        """
        Before:
            Empty.

        After:
            With training param on NPU.

        Process:
            1. onload training param
        """
        if self.train_param_offload:
            self.megatron_offloader.onload_param()

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
        if self.optimizer_offload:
            self.megatron_offloader.onload_optimizer()
        if self.grad_offload:
            self.megatron_offloader.onload_grad()

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
