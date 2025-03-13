#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd.2023-2025. All rights reserved.
import os
import logging

import tensordict
import torch
from torch_npu.contrib import transfer_to_npu

from mindspeed_rl.models.rollout.vllm_engine import VLLMInferEngine
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.utils.loggers import Loggers

tokenizer_name_or_path = "/data/for_dt/tokenizer/Llama-3.2-1B-Instruct/"
weights_path = "/data/for_dt/weights/Llama-3.2-1B-tp1pp1/iter_0000001/mp_rank_00/model_optim_rng.pt"
megatron_dict = {"num_attention_heads": 32,
                 "tensor_model_parallel_size": 1,
                 "num_query_groups": 8,
                 "group_query_attention": True}
sampling_config = {
    "num_completions": 1,  # 每个输入提示生成的独立完成项数量
    "logprobs": 1,  # 返回的 top token 的对数概率数量
    "max_tokens": 128,  # 生成输出的最大 token 数量
    "best_of": 2,  # 内部生成候选完成项的数量，从中选择最佳的一个
    "top_p": 1.0,  # 核采样的累积概率阈值
    "top_k": 50,  # 采样时考虑的最高概率 token 的数量
    "min_p": 0.0,  # token 选择的最小概率阈值
    "temperature": 0.2,  # 控制预测随机性的温度参数
    "detokenize": False  # 是否将生成的 token 转换回可读字符串
}


def main():
    logger = Loggers(
        name="test_vllm_engine",
    )
    logger.info("start test_vllm_engine")

    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]

    logger.info("load megatron weight")
    megatron_st = torch.load(weights_path)
    actor_weights = megatron_st['model']

    # 配置初始化所需的参数

    train_tensor_parallel_size = 1
    train_pipeline_parallel_size = 1
    infer_tensor_parallel_size = 1
    infer_pipeline_parallel_size = 1
    train_expert_parallel_size = 1
    infer_expert_parallel_size = 1
    max_num_seqs = 256
    trust_remote_code = True

    logger.info("enter vllmInferEngine")

    megatron_config = MegatronConfig(megatron_dict, {})
    megatron_config.num_attention_heads = 32
    megatron_config.tensor_model_parallel_size = 1
    megatron_config.num_query_groups = 8
    megatron_config.num_key_value_heads = 8
    megatron_config.group_query_attention = True
    # 初始化 VLLMInferEngine 实例
    inference_engine = VLLMInferEngine(
        megatron_config=megatron_config,
        sampling_config=sampling_config,
        train_expert_parallel_size=train_expert_parallel_size,
        infer_expert_parallel_size=infer_expert_parallel_size,
        tokenizer_name_or_path=tokenizer_name_or_path,
        train_tensor_parallel_size=train_tensor_parallel_size,
        train_pipeline_parallel_size=train_pipeline_parallel_size,
        infer_tensor_parallel_size=infer_tensor_parallel_size,
        infer_pipeline_parallel_size=infer_pipeline_parallel_size,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code
    )

    logger.info("model inited")
    inference_engine.free_cache_engine()
    torch.cuda.empty_cache()
    logger.info("free_cache")

    inference_engine.offload_model_weights()
    logger.info("offload_model")
    torch.cuda.empty_cache()
    logger.info("empty_cache")

    logger.info("enter sync_model_weights")
    inference_engine.sync_model_weights(actor_weights)

    logger.info("enter  init_cache_engine")
    inference_engine.init_cache_engine()

    logger.info("=" * 80)
    logger.info("start chat")
    outputs = inference_engine.chat(conversation)
    logger.info("chat result is ", outputs)

    idx_list = []
    idx_list_per_step = []
    for i in range(2):
        for j in range(4):
            tokens = torch.randint(100, (10,))
            idx_list_per_step.append(tokens.view(-1).cpu().numpy().tolist())
        idx_list.extend(idx_list_per_step)
        idx_list_per_step = []
    logger.info(len(idx_list), [len(i) for i in idx_list])

    logger.info("start test generate_sequences ")
    outputs = inference_engine.generate_sequences(
        idx_list=idx_list,
    )
    logger.info("generate_sequences output is:")
    logger.info(outputs[0])
    logger.info(outputs[0].shape)

    logger.info("input")
    logger.info(idx_list[0])


if __name__ == "__main__":
    main()
