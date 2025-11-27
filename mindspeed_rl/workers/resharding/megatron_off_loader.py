# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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
Megatron OFF Loader:
offload/onload optimizer、grad、param
"""


from itertools import chain
from collections import defaultdict

import torch
import torch.distributed
from mindspeed.op_builder import AlgorithmOpBuilder

from mindspeed_rl.utils.utils import mstx_timer_decorator


class MegatronOffLoader:
    def __init__(self, megatron_model=None, optimizer=None, wrap_with_ddp=True, megatron_config=None,
                 distributed_optimizer=None, float16_optimizer_with_float16_params=None):
        from mindspeed.checkpointing import unchain_optimizer
        self.optimizer = unchain_optimizer(optimizer)
        self.model = megatron_model
        self.wrap_with_ddp = wrap_with_ddp
        self.megatron_config = megatron_config
        self.distributed_optimizer = distributed_optimizer
        self.float16_optimizer_with_float16_params = float16_optimizer_with_float16_params

        self.tensor_to_cpu_states_map = dict()
        self.reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr

    def reuse_storage_of_distributed_optimizer_without_dp(self, optimizer):
        """
        This function is designed to re-map addresses when enabling the dist_optimizer and DP=1 scenario.

        For more details, please refer to the
        "reuse_fp32_param_distrib_optimizer_init" function in the "mindspeed/optimizer/distrib_optimizer.py" file
        """
        for i, _ in enumerate(optimizer.buffers):
            shard_res_and_buffer_model_param = optimizer.shard_main_param_res_buffers[i]
            shard_main_param_int32_view_buffer = optimizer.model_param_bucket_and_shard_main_param_int32_view_map[
                shard_res_and_buffer_model_param]
            self.reuse_data_ptr(shard_main_param_int32_view_buffer, shard_res_and_buffer_model_param, 0)

        for i, (_, _) in enumerate(zip(optimizer.model_float16_groups, optimizer.shard_fp32_from_float16_groups)):
            for j, (model_param, _) in enumerate(
                    zip(optimizer.model_float16_groups[i], optimizer.shard_fp32_from_float16_groups[i])):
                gbuf_index, _, bucket_id = optimizer.model_param_gbuf_map[model_param]
                data_start_index, data_end_index, bucket_id = \
                    optimizer.buffers[gbuf_index].param_index_map[model_param]
                self.reuse_data_ptr(optimizer.shard_fp32_from_float16_groups[i][j],
                                    optimizer.shard_main_param_res_buffers[gbuf_index],
                                    data_start_index)

        for i, _ in enumerate(optimizer.buffers):
            buffer_numel = optimizer.buffers[i].param_data.numel()
            self.reuse_data_ptr(optimizer.buffers[i].param_data,
                                optimizer.shard_main_param_res_buffers[i],
                                buffer_numel)

    def reuse_storage_of_distributed_optimizer_with_dp(self, optimizer):
        """
        This function is designed to re-map addresses when enabling the dist_optimizer and DP > 1 scenario.

        For more details, please refer to the
        "reuse_fp32_param_distrib_optimizer_init" function in the "mindspeed/optimizer/distrib_optimizer.py" file
        """
        data_parallel_world_size = torch.distributed.get_world_size(optimizer.data_parallel_group)
        data_parallel_rank = torch.distributed.get_rank(optimizer.data_parallel_group_gloo)
        for i, _ in enumerate(optimizer.buffers):
            bucket_res_numel = 0
            for j, _ in enumerate(optimizer.buffers[i].buckets):
                param_data_dp_numel = optimizer.buffers[i].buckets[j].param_data.numel() // data_parallel_world_size
                shard_main_param_int32_view_bucket = optimizer.model_param_bucket_and_shard_main_param_int32_view_map[
                    optimizer.buffers[i].buckets[j].param_data]
                self.reuse_data_ptr(
                    shard_main_param_int32_view_bucket,
                    optimizer.buffers[i].param_data,
                    (bucket_res_numel * data_parallel_world_size) // 2 + max(0,
                                                                             data_parallel_rank - 1) * param_data_dp_numel // 2)

                bucket_res_numel += param_data_dp_numel

        for i, (_, _) in enumerate(zip(optimizer.model_float16_groups, optimizer.shard_fp32_from_float16_groups)):
            for j, (model_param, _) in enumerate(
                    zip(optimizer.model_float16_groups[i], optimizer.shard_fp32_from_float16_groups[i])):
                world_range = optimizer._get_model_param_range_map(model_param)["gbuf_world_in_bucket"]
                gbuf_index, _, bucket_id = optimizer.model_param_gbuf_map[model_param]
                model_param_buffer = optimizer.buffers[gbuf_index].param_data
                bucket_offset_in_buffer = sum(optimizer.bucket_num_groups[gbuf_index][:bucket_id]) // 2
                model_param_bucket = optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
                model_param_bucket_numel_per_dp = model_param_bucket.numel() // data_parallel_world_size
                shard_fp32_param_bucket_offset = world_range.start if data_parallel_rank == 0 else \
                    world_range.start - model_param_bucket_numel_per_dp * (1 + data_parallel_rank) // 2
                shard_main_param_buffer_start = bucket_offset_in_buffer + shard_fp32_param_bucket_offset
                self.reuse_data_ptr(optimizer.shard_fp32_from_float16_groups[i][j], model_param_buffer,
                                    shard_main_param_buffer_start)

    def reuse_storage_of_float16_optimizer_with_float16_params(self, optimizer):
        """
        This function is designed to re-map addresses when Disabling the dist_optimizer scenario.

        For more details, please refer to the
        "reuse_fp32_param_init" function in the "mindspeed/optimizer/optimizer.py" file
        """
        for i, (_, _) in enumerate(zip(optimizer.float16_groups, optimizer.fp32_from_float16_groups)):
            for j, (_, _) in enumerate(zip(optimizer.float16_groups[i], optimizer.fp32_from_float16_groups[i])):
                res_float16_params_this_group = optimizer.res_float16_groups[i]
                float16_float32_params_this_group = optimizer.float16_float32_groups[i]
                int32_float32_params_this_group = optimizer.int32_float32_groups[i]

                self.reuse_data_ptr(float16_float32_params_this_group[j],
                                    optimizer.fp32_from_float16_groups[i][j], 0)
                self.reuse_data_ptr(int32_float32_params_this_group[j],
                                    optimizer.fp32_from_float16_groups[i][j], 0)
                self.reuse_data_ptr(res_float16_params_this_group[j], float16_float32_params_this_group[j], 0)
                self.reuse_data_ptr(optimizer.float16_groups[i][j],
                                    float16_float32_params_this_group[j],
                                    res_float16_params_this_group[j].numel())

    @mstx_timer_decorator
    def offload_grad(self):
        for model in self.model:
            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                self.swap_tensors_to_host(buffer.grad_data, copy_data=False)

    @mstx_timer_decorator
    def onload_grad(self):
        for model in self.model:
            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                self.swap_tensors_to_device(buffer.grad_data, copy_data=False)

    @mstx_timer_decorator
    def offload_optimizer(self):
        for index, _ in enumerate(self.optimizer):
            if self.megatron_config.reuse_fp32_param:
                if isinstance(self.optimizer[index], self.distributed_optimizer):
                    data_parallel_world_size = torch.distributed.get_world_size(
                        self.optimizer[index].data_parallel_group)

                    if data_parallel_world_size != 1:
                        for param_group in self.optimizer[index].optimizer.param_groups:
                            for param in param_group['params']:
                                param.data = param.data.to("cpu", non_blocking=False)

                        for i, _ in enumerate(self.optimizer[index].shard_main_param_res_buffers):
                            self.swap_tensors_to_host(self.optimizer[index].shard_main_param_res_buffers[i])
            else:
                for param_group in self.optimizer[index].optimizer.param_groups:
                    for param in param_group['params']:
                        if param.data.storage().size() == 0:
                            continue
                        param.data = param.data.to("cpu", non_blocking=False)

            self.optimizer[index].optimizer.state = self._move_to_device(self.optimizer[index].optimizer.state, "cpu")

    @mstx_timer_decorator
    def onload_optimizer(self):
        for index, _ in enumerate(self.optimizer):
            if self.megatron_config.reuse_fp32_param:
                if isinstance(self.optimizer[index], self.distributed_optimizer):
                    data_parallel_world_size = torch.distributed.get_world_size(
                        self.optimizer[index].data_parallel_group)

                    if data_parallel_world_size != 1:
                        for param_group in self.optimizer[index].optimizer.param_groups:
                            for param in param_group['params']:
                                param.data = param.data.to(torch.cuda.current_device(), non_blocking=False)

                        for i, _ in enumerate(self.optimizer[index].shard_main_param_res_buffers):
                            self.swap_tensors_to_device(self.optimizer[index].shard_main_param_res_buffers[i])

                        self.reuse_storage_of_distributed_optimizer_with_dp(self.optimizer[index])
            else:
                for param_group in self.optimizer[index].optimizer.param_groups:
                    for param in param_group['params']:
                        param.data = param.data.to(torch.cuda.current_device(), non_blocking=False)

            self.optimizer[index].optimizer.state = self._move_to_device(self.optimizer[index].optimizer.state,
                                                                         torch.cuda.current_device())

    @mstx_timer_decorator
    def _move_to_device(self, data, device):
        if isinstance(data, defaultdict):
            return defaultdict(data.default_factory,
                               {key: self._move_to_device(value, device) for key, value in data.items()})
        elif isinstance(data, dict):
            return {key: self._move_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, torch.Tensor):
            if data.storage().size() == 0 and device == 'cpu':
                return data
            return data.to(device, non_blocking=False)
        else:
            return data

    @mstx_timer_decorator
    def offload_param(self):
        if self.wrap_with_ddp:
            for index, _ in enumerate(self.optimizer):
                # 开启distributed_optimizer场景
                if isinstance(self.optimizer[index], self.distributed_optimizer):
                    data_parallel_world_size = torch.distributed.get_world_size(
                        self.optimizer[index].data_parallel_group)
                    # 当开启reuse_fp32并且DP等于1时，res和model权重一起卸载
                    if self.megatron_config.reuse_fp32_param and data_parallel_world_size == 1:
                        for i, _ in enumerate(self.optimizer[index].shard_main_param_res_buffers):
                            self.swap_tensors_to_host(self.optimizer[index].shard_main_param_res_buffers[i])
                    else:
                        for model in self.model:
                            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                                self.swap_tensors_to_host(buffer.param_data)
                # 关闭distributed_optimizer场景
                elif isinstance(self.optimizer[index], self.float16_optimizer_with_float16_params):
                    # 开启reuse_fp32特性时，res和model权重统一卸载
                    if self.megatron_config.reuse_fp32_param:
                        for i, _ in enumerate(self.optimizer[index].fp32_from_float16_groups):
                            for j, _ in enumerate(self.optimizer[index].fp32_from_float16_groups[i]):
                                self.swap_tensors_to_host(self.optimizer[index].fp32_from_float16_groups[i][j])
                    else:
                        for model in self.model:
                            for _, param in model.named_parameters():
                                self.swap_tensors_to_host(param)

        else:
            for item in self.model:
                item.to('cpu')

    @mstx_timer_decorator
    def onload_param(self):
        if self.wrap_with_ddp:
            for index, _ in enumerate(self.optimizer):
                # 开启distributed_optimizer场景
                if isinstance(self.optimizer[index], self.distributed_optimizer):
                    data_parallel_world_size = torch.distributed.get_world_size(
                        self.optimizer[index].data_parallel_group)
                    # 当开启reuse_fp32并且DP等于1时，res和model权重一起加载
                    if self.megatron_config.reuse_fp32_param and data_parallel_world_size == 1:
                        for i, _ in enumerate(self.optimizer[index].shard_main_param_res_buffers):
                            self.swap_tensors_to_device(self.optimizer[index].shard_main_param_res_buffers[i])

                        self.reuse_storage_of_distributed_optimizer_without_dp(self.optimizer[index])
                    else:
                        for model in self.model:
                            for buffer in chain(model.buffers, model.expert_parallel_buffers):
                                self.swap_tensors_to_device(buffer.param_data)
                # 关闭distributed_optimizer场景
                elif isinstance(self.optimizer[index], self.float16_optimizer_with_float16_params):
                    # 开启reuse_fp32特性时，res和model权重统一加载
                    if self.megatron_config.reuse_fp32_param:
                        for i, _ in enumerate(self.optimizer[index].fp32_from_float16_groups):
                            for j, _ in enumerate(self.optimizer[index].fp32_from_float16_groups[i]):
                                self.swap_tensors_to_device(self.optimizer[index].fp32_from_float16_groups[i][j])

                        self.reuse_storage_of_float16_optimizer_with_float16_params(self.optimizer[index])
                    else:
                        for model in self.model:
                            for _, param in model.named_parameters():
                                self.swap_tensors_to_device(param)

        else:
            for item in self.model:
                item.to(torch.cuda.current_device())

    @mstx_timer_decorator
    def swap_tensors_to_host(self, tensor, copy_data=True):
        if tensor not in self.tensor_to_cpu_states_map:
            self.tensor_to_cpu_states_map[tensor] = torch.empty_like(tensor, device='cpu')
        if tensor.storage().size() != 0:
            if copy_data:
                cpu_state = self.tensor_to_cpu_states_map[tensor]
                with torch.no_grad():
                    cpu_state.copy_(tensor, non_blocking=False)
            tensor.storage().resize_(0)

    @mstx_timer_decorator
    def swap_tensors_to_device(self, tensor, copy_data=True):
        if tensor.storage().size() == 0:
            cpu_state = self.tensor_to_cpu_states_map[tensor]
            tensor.storage().resize_(cpu_state.storage().size())
            if copy_data:
                with torch.no_grad():
                    tensor.copy_(cpu_state, non_blocking=False)
