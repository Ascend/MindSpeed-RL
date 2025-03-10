# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.


from typing import Dict
import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models import ModelRegistry

from mindspeed_rl.config_cls.megatron_config import MegatronConfig


def parallel_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
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


# 默认权重加载器
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    if param.size() != loaded_weight.size():
        raise ValueError("The parameter size does not match the loaded weight size.")
    if param.data.dtype != loaded_weight.data.dtype:
        raise ValueError("if we want to shared weights, the data type should also be the same")
    param.data = loaded_weight.data


# GPT-2 权重加载器
def gpt2_weight_loader(actor_weights: Dict, vllm_model: nn.Module, megatron_config: MegatronConfig) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "lm_head.weight" in name:
            # GPT-2 ties the weights of the embedding layer and the final linear layer.
            continue
        if ".attn.bias" in name or ".attn.masked_bias" in name:
            # Skip attention mask.
            # NOTE: "c_attn.bias" should not be skipped.
            continue
        if not name.startswith("transformer."):
            name = "transformer." + name
        param = params_dict[name]
        # The HF's GPT-2 implementation uses Conv1D instead of Linear.
        # Because of this, we need to transpose the weights.
        # Note(zhuohan): the logic below might break quantized models.
        for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
            if conv1d_weight_name not in name:
                continue
            if not name.endswith(".weight"):
                continue
            loaded_weight = loaded_weight.t()
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    return vllm_model


# Llama Megatron 权重加载器
def llama_megatron_weight_loader(actor_weights: Dict, 
        vllm_model: nn.Module, 
        megatron_config: MegatronConfig) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    return vllm_model    


# Llama Megatron Core TE 权重加载器
def llama_megatron_core_te_weight_loader(actor_weights: Dict, vllm_model: nn.Module, megatron_config: MegatronConfig) -> nn.Module:
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", "self_attn.o_proj"),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
        ("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias"),
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        ("mlp.linear_fc2", "mlp.down_proj"),
        ("decoder.final_layernorm", "model.norm"),
        ("output_layer", "lm_head"),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name.endswith(".bias") and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    return vllm_model


# QKV 权重分割函数
def qkv_split_weight(query_key_value, megatron_config: MegatronConfig):
    nh = megatron_config.num_attention_heads // megatron_config.tensor_model_parallel_size
    ng = (megatron_config.num_query_groups if megatron_config.group_query_attention else megatron_config.num_attention_heads) // megatron_config.tensor_model_parallel_size
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


# QKV 偏置分割函数
def qkv_split_bias(query_key_value, megatron_config: MegatronConfig):
    nh = megatron_config.num_attention_heads // megatron_config.tensor_model_parallel_size
    ng = (megatron_config.num_query_groups if megatron_config.group_query_attention else megatron_config.num_attention_heads) // megatron_config.tensor_model_parallel_size
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


# Llama Megatron Core 权重加载器
def llama_megatron_core_weight_loader(actor_weights: Dict, vllm_model: nn.Module, megatron_config: MegatronConfig) -> nn.Module:
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", "self_attn.o_proj"),
        ("input_layernorm", "input_layernorm"),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        ("mlp.linear_fc2", "mlp.down_proj"),
        ("decoder.final_layernorm", "model.norm"),
        ("output_layer", "lm_head"),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name.endswith(".bias") and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        if name not in params_dict.keys():
            continue
        if "lm_head" in name:  # lm_head is not needed since it is tied with embedding
            continue
        if "qkv" in name:
            q_weight, k_weight, v_weight = qkv_split_weight(loaded_weight, megatron_config)
            loaded_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    return vllm_model


# Qwen Megatron 权重加载器
def qwen_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module, megatron_config: MegatronConfig) -> nn.Module:
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", "self_attn.o_proj"),
        ("input_layernorm", "input_layernorm"),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        ("mlp.linear_fc2", "mlp.down_proj"),
        ("decoder.final_layernorm", "model.norm"),
        ("output_layer", "lm_head"),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name not in params_dict.keys():
            continue
        if "qkv" in name:
            if name.endswith('.bias'):
                q_weight, k_weight, v_weight = qkv_split_bias(loaded_weight, megatron_config)
                loaded_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            else:
                q_weight, k_weight, v_weight = qkv_split_weight(loaded_weight, megatron_config)
                loaded_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    return vllm_model


# 替换名称函数
def _replace_name(megatron_name, name_mapping):
    for m_name, v_name in name_mapping:
        if m_name not in megatron_name:
            continue
        if "layers" in megatron_name:  # deal with decoder layers
            megatron_name = megatron_name.replace("decoder", "model")
            megatron_name_list = megatron_name.split(".")
            if "layer_norm_weight" in megatron_name_list or "layer_norm_bias" in megatron_name_list:
                param_name_list = megatron_name_list[:3]
                param_name_list.append(v_name)
                param_name = ".".join(param_name_list)
            else:
                param_name_list = megatron_name_list[:3]
                weight_or_bias = megatron_name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = ".".join(param_name_list)
            return param_name
        else:
            param_name = megatron_name.replace(m_name, v_name)
            return param_name


def mistral_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
    return vllm_model


# 层权重加载器注册表
LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY = {
    ColumnParallelLinear: parallel_weight_loader,
    MergedColumnParallelLinear: parallel_weight_loader,
    QKVParallelLinear: parallel_weight_loader,
    RowParallelLinear: parallel_weight_loader,
    VocabParallelEmbedding: parallel_weight_loader,
    ParallelLMHead: parallel_weight_loader,
    # "default_weight_loader": default_weight_loader
}


# 模型权重加载器注册表
MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY = {
    "GPT2LMHeadModel": gpt2_weight_loader,
    "LlamaForCausalLM": llama_megatron_core_weight_loader,  # use te backend for open-source megatron
    "LLaMAForCausalLM": llama_megatron_core_weight_loader,
    "MistralForCausalLM": mistral_megatron_weight_loader,
    "Qwen2ForCausalLM": qwen_megatron_weight_loader,
}


# 加载 Megatron 权重
def load_megatron_weights(actor_weights: Dict, vllm_model: nn.Module, megatron_config: MegatronConfig):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    vllm_model = weight_loader(actor_weights, vllm_model, megatron_config)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    vllm_model = vllm_model.cuda()
    return vllm_model


# 获取模型权重加载器
def _get_model_weight_loader(arch: str):
    if arch in MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY:
        return MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


# 更新 Megatron 权重加载器
def update_megatron_weight_loader():
    for layer_class, weight_loader in LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY.items():
        layer_class.weight_loader = weight_loader