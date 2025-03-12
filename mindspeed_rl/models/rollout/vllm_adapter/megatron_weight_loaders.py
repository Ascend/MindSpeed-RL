# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.

from typing import Dict
import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models import ModelRegistry

from mindspeed_rl.config_cls.megatron_config import MegatronConfig


class InferParallelConfig:
    def __init__(self, infer_tensor_parallel_size: int, infer_pipeline_parallel_size: int, infer_expert_parallel_size: int):
        self.infer_tensor_parallel_size = infer_tensor_parallel_size
        self.infer_pipeline_parallel_size = infer_pipeline_parallel_size
        self.infer_expert_parallel_size = infer_expert_parallel_size


def load_megatron_weights(actor_weights: Dict, vllm_model: nn.Module,
        infer_paralle_config: InferParallelConfig,
        megatron_config: MegatronConfig):
    model_weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    vllm_model = model_weight_loader(actor_weights, vllm_model, infer_paralle_config, megatron_config)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    vllm_model = vllm_model.cuda()
    return vllm_model


def llama_megatron_core_weight_loader(actor_weights: Dict, vllm_model: nn.Module, 
        infer_paralle_config: InferParallelConfig,
        megatron_config: MegatronConfig
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
            q_weight, k_weight, v_weight = qkv_split_weight(loaded_weight, infer_paralle_config, megatron_config)
            loaded_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    return vllm_model


def qwen_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module,
        infer_paralle_config: InferParallelConfig, megatron_config: MegatronConfig
) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if name not in params_dict.keys():
            continue
        if "qkv" in name:
            if name.endswith('.bias'):
                q_weight, k_weight, v_weight = qkv_split_bias(loaded_weight, infer_paralle_config, megatron_config)
                loaded_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            else:
                q_weight, k_weight, v_weight = qkv_split_weight(loaded_weight, infer_paralle_config, megatron_config)
                loaded_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    return vllm_model


def _get_model_weight_loader(arch: str):
    if arch in MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY:
        return MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def qkv_split_weight(query_key_value,
        infer_paralle_config: InferParallelConfig,
        megatron_config: MegatronConfig
):
    infer_tensor_parallel_size = infer_paralle_config.infer_tensor_parallel_size
    nh = megatron_config.num_attention_heads // infer_tensor_parallel_size
    ng = (megatron_config.num_query_groups if megatron_config.group_query_attention else megatron_config.num_attention_heads) // infer_tensor_parallel_size
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


def qkv_split_bias(query_key_value, infer_paralle_config: InferParallelConfig, megatron_config: MegatronConfig):
    infer_tensor_parallel_size = infer_paralle_config.infer_tensor_parallel_size
    nh = megatron_config.num_attention_heads // infer_tensor_parallel_size
    ng = (megatron_config.num_query_groups if megatron_config.group_query_attention else megatron_config.num_attention_heads) // infer_tensor_parallel_size
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


def update_megatron_weight_loader():
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
}


LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY = {
    ColumnParallelLinear: parallel_weight_loader,
    MergedColumnParallelLinear: parallel_weight_loader,
    QKVParallelLinear: parallel_weight_loader,
    RowParallelLinear: parallel_weight_loader,
    VocabParallelEmbedding: parallel_weight_loader,
    ParallelLMHead: parallel_weight_loader
}
