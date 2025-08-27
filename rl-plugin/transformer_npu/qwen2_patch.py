import torch
import torch_npu
from torch_npu import npu_rotary_mul as apply_rotary_emb
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


def rms_norm_forward(self, x):
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def silu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )


def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueese_dim=1):
    cos = cos.unsqueese(unsqueese_dim)
    sin = sin.unsqueese(unsqueese_dim)
    q_embed = torch_npu.npu_rotary_mul(q.contiguous(), cos, sin).to(q.dtype)
    k_embed = torch_npu.npu_rotary_mul(k.contiguous(), cos, sin).to(k.dtype)
    return q_embed, k_embed


def apply_qwen2_patch():
    Qwen2MLP.forward = silu_forward
    Qwen2RMSNorm.forward = rms_norm_forward
    modeling_qwen2.fused_apply_rotary_pos_emb = fused_apply_rotary_pos_emb