# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.

from typing import Dict
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


class InferParallelConfig:
    def __init__(self, infer_tensor_parallel_size: int, infer_pipeline_parallel_size: int, infer_expert_parallel_size: int):
        self.infer_tensor_parallel_size = infer_tensor_parallel_size
        self.infer_pipeline_parallel_size = infer_pipeline_parallel_size
        self.infer_expert_parallel_size = infer_expert_parallel_size


def load_megatron_weights(actor_weights: Dict, vllm_model: nn.Module,
        infer_paralle_config: InferParallelConfig,
        hf_config: PretrainedConfig):
    model_weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    vllm_model = model_weight_loader(actor_weights, vllm_model, infer_paralle_config, hf_config)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    vllm_model = vllm_model.cuda()
    return vllm_model


def llama_megatron_core_weight_loader(actor_weights: Dict, vllm_model: nn.Module, 
        infer_paralle_config: InferParallelConfig,
        hf_config: PretrainedConfig
) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if name.endswith(".bias") and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        if name not in params_dict.keys():
            continue
        if "lm_head" in name:  # lm_head is not needed since it is tied with embedding
            continue
        if "qkv" in name:
            q_weight, k_weight, v_weight = qkv_split_weight(loaded_weight, infer_paralle_config, hf_config)
            loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
        load_single_weight(params_dict, name, loaded_weight)
    return vllm_model


def qwen_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module,
        infer_paralle_config: InferParallelConfig, hf_config: PretrainedConfig
) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if name not in params_dict.keys():
            continue
        if "qkv" in name:
            if name.endswith('.bias'):
                q_weight, k_weight, v_weight = qkv_split_bias(loaded_weight, infer_paralle_config, hf_config)
                loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
            else:
                q_weight, k_weight, v_weight = qkv_split_weight(loaded_weight, infer_paralle_config, hf_config)
                loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
        
        if "mlp.experts.w13_weight" in name:
            loaded_weight.copy_(loaded_weight.view(
                hf_config.num_experts // infer_paralle_config.infer_expert_parallel_size,
                hf_config.hidden_size,
                -1
            ).transpose(2, 1).contiguous())
        if "mlp.experts.w2_weight" in name:
            loaded_weight.copy_(loaded_weight.view(
                hf_config.num_experts // infer_paralle_config.infer_expert_parallel_size,
                -1,
                hf_config.hidden_size
            ).transpose(2, 1).contiguous())
        load_single_weight(params_dict, name, loaded_weight)
    return vllm_model


def qwen_vl_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module, 
                                  infer_paralle_config: InferParallelConfig, hf_config: PretrainedConfig) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters())
    vision_config = type('obj', (object,), {
        'num_attention_heads': hf_config.vision_config.num_heads,
        'num_key_value_heads': hf_config.vision_config.num_heads,
    })
    
    for name, loaded_weight in actor_weights.items():
        if name not in params_dict.keys():
            continue
        if "qkv" in name:
            if 'visual' in name:
                if name.endswith('.bias'):
                    q_weight, k_weight, v_weight = qkv_split_bias(loaded_weight, infer_paralle_config, vision_config)
                    loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
                else:
                    q_weight, k_weight, v_weight = qkv_split_weight(loaded_weight, infer_paralle_config, vision_config)
                    loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
            else:
                if name.endswith('.bias'):
                    q_weight, k_weight, v_weight = qkv_split_bias(loaded_weight, infer_paralle_config, hf_config)
                    loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
                else:
                    q_weight, k_weight, v_weight = qkv_split_weight(loaded_weight, infer_paralle_config, hf_config)
                    loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
        
        load_single_weight(params_dict, name, loaded_weight)
    
    return vllm_model


def deepseek_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module,
        infer_paralle_config: InferParallelConfig, hf_config: PretrainedConfig
) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "qkv" in name:
            split_dim = hf_config.q_lora_rank if hf_config.q_lora_rank else \
                (hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim) * hf_config.num_attention_heads
            q_name = name.replace("qkv_proj", "q_a_proj" if hf_config.q_lora_rank else "q_proj")
            kv_name = name.replace("qkv_proj", "kv_a_proj_with_mqa")
            load_single_weight(params_dict, q_name, loaded_weight[:split_dim])
            load_single_weight(params_dict, kv_name, loaded_weight[split_dim:])
            continue
        if name not in params_dict.keys():
            raise ValueError(f"unexpected key {name} in deepseek_megatron_weight_loader")
        if "mlp.experts.w13_weight" in name:
            loaded_weight.copy_(loaded_weight.view(hf_config.n_routed_experts // infer_paralle_config.infer_expert_parallel_size, hf_config.hidden_size, -1).transpose(2, 1).contiguous())
        if "mlp.experts.w2_weight" in name:
            loaded_weight.copy_(loaded_weight.view(hf_config.n_routed_experts // infer_paralle_config.infer_expert_parallel_size, -1, hf_config.hidden_size).transpose(2, 1).contiguous())
        load_single_weight(params_dict, name, loaded_weight)
    return vllm_model


def _get_model_weight_loader(arch: str):
    from vllm.model_executor.models import ModelRegistry
    if arch in MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY:
        return MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def qkv_split_weight(query_key_value,
        infer_paralle_config: InferParallelConfig,
        hf_config: PretrainedConfig
):
    infer_tensor_parallel_size = infer_paralle_config.infer_tensor_parallel_size
    nh = hf_config.num_attention_heads // infer_tensor_parallel_size
    ng = hf_config.num_key_value_heads // infer_tensor_parallel_size
    repeats = nh // ng
    qkv_weight = query_key_value.reshape(
        ng,
        repeats + 2,
        query_key_value.shape[0] // ng // (repeats + 2),
        query_key_value.shape[1],
    )
    hidden_size = qkv_weight.shape[-1]
    qw = qkv_weight[:, :repeats, ...].reshape(-1, hidden_size)
    kw = qkv_weight[:, repeats: repeats + 1, ...].reshape(-1, hidden_size)
    vw = qkv_weight[:, repeats + 1:, ...].reshape(-1, hidden_size)
    return qw, kw, vw


def qkv_split_bias(query_key_value, infer_paralle_config: InferParallelConfig, hf_config: PretrainedConfig):
    infer_tensor_parallel_size = infer_paralle_config.infer_tensor_parallel_size
    nh = hf_config.num_attention_heads // infer_tensor_parallel_size
    ng = hf_config.num_key_value_heads // infer_tensor_parallel_size
    repeats = nh // ng
    bias_weight = query_key_value.reshape(
        ng, 
        repeats + 2, 
        query_key_value.shape[0] // ng // (repeats + 2)
    )
    qw = bias_weight[:, :repeats, ...].reshape(-1)
    kw = bias_weight[:, repeats: repeats + 1, ...].reshape(-1)
    vw = bias_weight[:, repeats + 1:, ...].reshape(-1)
    return qw, kw, vw


def load_single_weight(params_dict, name, loaded_weight):
    param = params_dict[name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)


def update_megatron_weight_loader():
    from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
    RowParallelLinear, ReplicatedLinear)
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY = {
        ColumnParallelLinear: parallel_weight_loader,
        MergedColumnParallelLinear: parallel_weight_loader,
        QKVParallelLinear: parallel_weight_loader,
        RowParallelLinear: parallel_weight_loader,
        VocabParallelEmbedding: parallel_weight_loader,
        ParallelLMHead: parallel_weight_loader,
        ReplicatedLinear: parallel_weight_loader,
        FusedMoE: parallel_weight_loader
    }

    for layer_class, weight_loader in LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY.items():
        layer_class.weight_loader = weight_loader


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    if param.size() != loaded_weight.size():
        raise ValueError("The parameter size does not match the loaded weight size.")
    if param.data.dtype != loaded_weight.data.dtype:
        raise ValueError("if we want to shared weights, the data type should also be the same")
    param.data = loaded_weight.data


def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Parallel Linear weight loader."""
    if param.size() != loaded_weight.size():
        error_msg = (
            f"the parameter size is not align with the loaded weight size, param size: {param.size()}, "
            f"loaded_weight size: {loaded_weight.size()}"
        )
        raise ValueError(error_msg)
    if param.data.dtype != loaded_weight.data.dtype:
        raise ValueError("if we want to shared weights, the data type should also be the same")
    param.data = loaded_weight.data


MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY = {
    "LlamaForCausalLM": llama_megatron_core_weight_loader,
    "Qwen2ForCausalLM": qwen_megatron_weight_loader,
    "CustomQwen3MoeForCausalLM": qwen_megatron_weight_loader,
    "DeepseekV3ForCausalLM": deepseek_megatron_weight_loader,
    "DeepseekV2ForCausalLM": deepseek_megatron_weight_loader,
    "CustomDeepseekV3ForCausalLM": deepseek_megatron_weight_loader,
    "Qwen2_5_VLForConditionalGeneration": qwen_vl_megatron_weight_loader
}
