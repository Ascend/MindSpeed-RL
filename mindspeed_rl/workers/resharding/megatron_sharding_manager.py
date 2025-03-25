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


class MegatronOffLoader:
    def __init__(
            self,
            optimizer=None,
            megatron_model=None
    ):
        self.optimizer = optimizer
        self.train_model = megatron_model

        self.tensor_to_cpu_states_map = dict()

    def offload_grad(self):
        for model_idx, model in enumerate(self.train_model):
            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                self.swap_tensors_to_host(buffer.grad_data)

    def onload_grad(self):
        for model_idx, model in enumerate(self.train_model):
            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                self.swap_tensors_to_device(buffer.grad_data)

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

    def offload_train_param(self):
        for model_idx, model in enumerate(self.train_model):
            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                self.swap_tensors_to_host(buffer.param_data)

    def onload_train_param(self):
        for model_idx, model in enumerate(self.train_model):
            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                self.swap_tensors_to_device(buffer.param_data)

    def swap_tensors_to_host(self, tensor):
        if tensor not in self.tensor_to_cpu_states_map:
            self.tensor_to_cpu_states_map[tensor] = torch.empty_like(tensor, device='cpu')
        if tensor.storage().size() != 0:
            cpu_state = self.tensor_to_cpu_states_map[tensor]
            cpu_state.copy_(tensor, non_blocking=False)
            tensor.storage().resize_(0)

    def swap_tensors_to_device(self, tensor):
        if tensor.storage().size() == 0:
            cpu_state = self.tensor_to_cpu_states_map[tensor]
            tensor.storage().resize_(cpu_state.storage().size())
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
            tp_split_expert=None,
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
            tp_split_expert (bool): Controls whether expert model parameters are split across multiple GPUs.
            parallel_state (ModuleType): Megatron parallel state of the model.
        """
        self.inference_engine = inference_engine
        self.optimizer = optimizer
        self.train_model = megatron_model

        self.vllm_weight_container = MegatronStyleVllmWeightContainer(
            megatron_model=megatron_model,
            vllm_model=self.inference_engine.model,
            model_config=model_config,
            infer_tensor_parallel_size=infer_tensor_parallel_size,
            infer_pipeline_parallel_size=infer_pipeline_parallel_size,
            infer_expert_parallel_size=infer_expert_parallel_size,
            num_layer_list=num_layer_list,
            tp_split_expert=tp_split_expert,
            parallel_state=parallel_state,
            enable_validate=enable_validate)

        self.optimizer_offload = optimizer_offload
        self.grad_offload = grad_offload
        self.train_param_offload = train_param_offload
        self.enable_validate = enable_validate
        self.use_distributed_optimizer = self.optimizer.config.use_distributed_optimizer
        self.inference_engine.offload_model_weights()
        self.megatron_offloader = megatron_offloader

    def __enter__(self):
        self.reshard_to_infer_mode()

    def __exit__(self, exc_type, exc_value, traceback):
        self.reshard_to_train_mode()

    def reshard_to_train_mode(self):
        self.inference_engine.offload_model_weights()
        self.offload_infer_params()
        torch.cuda.empty_cache()
        if self.optimizer_offload:
            self.megatron_offloader.onload_optimizer()
        if self.train_param_offload:
            self.megatron_offloader.onload_train_param()
        if self.grad_offload:
            self.megatron_offloader.onload_grad()
        # add empty cache after each compute
        torch.cuda.empty_cache()

    def reshard_to_infer_mode(self):
        if self.optimizer_offload:
            self.megatron_offloader.offload_optimizer()
        if self.grad_offload:
            self.megatron_offloader.offload_grad()
        torch.cuda.empty_cache()
        if self.train_param_offload:
            self.megatron_offloader.onload_train_param()

        self.onload_infer_params()
        infer_params = self.vllm_weight_container.get_infer_params()
        if self.train_param_offload:
            self.megatron_offloader.offload_train_param()
        self.inference_engine.sync_model_weights(infer_params, load_format='megatron')
        torch.cuda.empty_cache()

    def offload_infer_params(self):
        infer_weight_buffers = self.vllm_weight_container.weight_buffers
        for buffer in infer_weight_buffers:
            buffer.offload()

    def onload_infer_params(self):
        infer_weight_buffers = self.vllm_weight_container.weight_buffers
        for buffer in infer_weight_buffers:
            buffer.onload()
