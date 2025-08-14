#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
from typing import Tuple

import torch
import vllm.model_executor.layers.rotary_embedding


def m_rotary_embedding_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch_npu
    mrope_section = [0, 0, 0] if positions.ndim == 1 else self.mrope_section

    query, key = torch_npu.npu_mrope(positions.contiguous(),
                                        query.contiguous(),
                                        key.contiguous(),
                                        self.cos_sin_cache.contiguous(),
                                        self.head_size,
                                        mrope_section=mrope_section,
                                        rotary_mode='half')
        
    return query, key


def replace_qwenvl_mrope():
    vllm.model_executor.layers.rotary_embedding.MRotaryEmbedding.forward = m_rotary_embedding_forward
