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
This file contains utilities to manipulate torch memory buffers
"""

from typing import Dict, List

import torch
from torch import nn


class MemoryBuffer:
    """
    A memory buffer is a contiguous torch tensor that may combine multiple tensors sharing with the underlying
    memory. It must have a unique type to support this behavior.
    """

    def __init__(self, numel: int, numel_padded: int, dtype: torch.dtype):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        # Create a memory buffer with padding for alignment (e.g., 128-bit alignment)
        self.data = torch.zeros(self.numel_padded, dtype=self.dtype,
                                device=torch.cuda.current_device(), requires_grad=False)
        self.tensor_indices = {}  # Stores the mapping of parameter names to their position in the buffer

    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()

    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`.
        """
        end_index = start_index + shape.numel()
        if end_index > self.numel:
            raise IndexError('requested tensor is out of the buffer range.')
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def copy_by_name(self, param_name: str, param):
        """Copy buffer_tensor"""
        buffer_tensor = self.get_by_name(param_name)
        buffer_tensor = buffer_tensor.view(param.shape)
        buffer_tensor.copy_(param)

    def get_by_name(self, param_name: str):
        """
        Retrieve the original tensor view from the buffer based on the param_name.

        Args:
            param_name (str): The name of the parameter (e.g., 'layer1.weight').

        Returns:
            torch.Tensor: The tensor view corresponding to the parameter name.
        """
        if param_name not in self.tensor_indices:
            raise KeyError(f"Parameter {param_name} not found in the buffer.")

        start_index, shape = self.tensor_indices[param_name]
        return self.get(shape, start_index)


def calc_padded_numel(shape: torch.Size, dtype: torch.dtype):
    """for cuda memory alignment, make sure alignment by 128-bits"""
    align_numel = 128 // torch.finfo(dtype).bits
    numel = shape.numel()
    return (numel + align_numel - 1) // align_numel * align_numel


def build_memory_buffer(weight_buffer_meta: Dict[str, Dict]) -> Dict[torch.dtype, MemoryBuffer]:
    """Build the memory buffer given weight_buffer_meta

    Args:
        weight_buffer_meta: contains mapping from name to a dictionary containing shape and dtype of the tensors

    Returns: a large memory buffer for each dtype that can hold all the tensors

    """
    memory_buffers = {}
    total_numel_map = {}  # map from dtype to the total numel
    for name, meta_info in sorted(weight_buffer_meta.items()):
        shape = meta_info['shape']
        dtype = meta_info['dtype']

        if not isinstance(shape, torch.Size):
            raise TypeError("Shape must be an instance of torch.Size")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be an instance of torch.dtype")
        if dtype not in total_numel_map:
            total_numel_map[dtype] = 0

        total_numel_map[dtype] += calc_padded_numel(shape, dtype)

    for dtype, total_numel in total_numel_map.items():
        # Create a buffer for each dtype with the total numel
        memory_buffers[dtype] = MemoryBuffer(total_numel, total_numel, dtype)

    # Now, insert each tensor's index and shape for later retrieval by name
    current_index_map = {}  # This keeps track of the current memory index for each dtype
    for name, meta_info in sorted(weight_buffer_meta.items()):
        shape = meta_info['shape']
        dtype = meta_info['dtype']
        buffer = memory_buffers[dtype]
        tensor_size = calc_padded_numel(shape, dtype)
        start_index = current_index_map.get(dtype, 0)
        current_index_map[dtype] = start_index + tensor_size
        buffer.tensor_indices[name] = (start_index, shape)
    return memory_buffers


def build_model_weight_buffer(model: nn.Module, names_per_pp: List[str], get_weight_buffer_meta):
    memory_buffers = [ModelWeightBuffer(model, weight_names, get_weight_buffer_meta) for weight_names in names_per_pp]
    return memory_buffers


class ModelWeightBuffer:
    """
    A factory class that processes a model's state_dict and returns memory buffers for the model parameters.
    It also provides a mapping between model parameter names and their corresponding memory buffer view.
    """
    def __init__(self, model: nn.Module, weight_names: List, get_weight_buffer_meta):
        self.model = model
        self.get_weight_buffer_meta = get_weight_buffer_meta
        self.weight_buffer_meta = self.get_weight_buffer_meta(self.model, weight_names)
        self.weight_names = list(self.weight_buffer_meta.keys())
        self.memory_buffers = None
        # self.memory_buffers = build_memory_buffer(self.weight_buffer_meta)

    def __getitem__(self, weight_name: str) -> torch.Tensor:
        return self.get_weight_by_name(weight_name)

    def get_weight_by_name(self, weight_name: str) -> torch.Tensor:
        dtype = self.weight_buffer_meta[weight_name]['dtype']
        return self.memory_buffers[dtype].get_by_name(weight_name)

    def copy_by_name(self, weight_name: str, param):
        dtype = self.weight_buffer_meta[weight_name]['dtype']
        self.memory_buffers[dtype].copy_by_name(weight_name, param)

    def offload(self):
        for memory_buffer in self.memory_buffers.values():
            memory_buffer.data = memory_buffer.data.to("cpu", non_blocking=False)

    def onload(self):
        for memory_buffer in self.memory_buffers.values():
            memory_buffer.data = memory_buffer.data.to(torch.cuda.current_device(), non_blocking=False)

    def destroy(self):
        for memory_buffer in self.memory_buffers.values():
            memory_buffer = None
        self.memory_buffers = None

    def rebuild(self):
        if self.memory_buffers is None:
            self.memory_buffers = build_memory_buffer(self.weight_buffer_meta)