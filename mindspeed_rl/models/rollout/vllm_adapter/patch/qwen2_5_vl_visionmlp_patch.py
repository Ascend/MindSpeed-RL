# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.


from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm
import vllm.model_executor
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
import vllm.model_executor.models
import vllm.model_executor.models.qwen2_5_vl


class Npu_Qwen2_5_VisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        # 合并 gate_proj 和 up_proj
        self.gate_up_proj = ColumnParallelLinear(
            in_features,
            hidden_features * 2,  # 合并后输出维度翻倍
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj"
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        # 合并后的 gate_up_proj 输出两个部分：x_gate 和 x_up
        gate_up, _ = self.gate_up_proj(x)
        x_gate, x_up = gate_up.chunk(2, dim=-1)  # 按最后一维拆分
        x_gate = self.act_fn(x_gate)
        x_down, _ = self.down_proj(x_gate * x_up)
        return x_down


def replace_qwen2_5_visionmlp():
    vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VisionMLP = Npu_Qwen2_5_VisionMLP
