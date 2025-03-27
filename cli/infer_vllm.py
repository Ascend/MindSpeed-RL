import os
import argparse
import logging

import tensordict
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import vllm.distributed.parallel_state as ps

from mindspeed_rl.models.rollout.vllm_engine import VLLMInferEngine
from mindspeed_rl.utils.loggers import Loggers

logger = Loggers(
    name="vllm_engine_inference",
)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='inference args')
    group.add_argument('--tokenizer-name-or-path', type=str,
                        help="Huggingface config path.")
    group.add_argument('--load-format', type=str,
                        choices=["auto", "megatron"], default="auto",
                        help="Vllm weight load format, support auto from huggingface and from megatron format.")
    group.add_argument('--load', type=str,
                        default=None,
                        help="Vllm weight path for megatron load format.")
    group.add_argument('--tensor-parallel-size', type=int,
                        default=1,
                        help="infer tensor parallel size")
    group.add_argument('--query', type=str, default="Write an essay about the importance of higher education.",
                       help='Input query.')
    group.add_argument('--task', type=str,
                       choices=["generation", "chat"], default="chat",
                       help='Inference task, generation or chat.')
    group.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='Device memory ratio allocated for vllm.')

    group = parser.add_argument_group(title='distributed')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--local-rank', type=int, default=int(os.getenv('LOCAL_RANK', '0')),
                       help='Local rank passed from distributed launcher.')

    group = parser.add_argument_group(title='sampling params')
    group.add_argument('--num-completions', type=int, default=1,
                       help='Number of output sequences to return for the given prompt.')
    group.add_argument('--logprobs', type=int, default=1,
                       help='Number of log probabilities to return per output token.')
    group.add_argument('--max-tokens', type=int, default=128,
                       help='Maximum number of tokens to generate per output sequence.')
    group.add_argument('--top-p', type=float, default=1.0,
                       help='Float that controls the cumulative probability of the top tokens to consider.')
    group.add_argument('--top-k', type=int, default=-1,
                       help='Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.')
    group.add_argument('--temperature', type=float, default=1.0,
                       help='Float that controls the randomness of the sampling.')               
    return parser.parse_args()


def process_outputs(outputs):
    res = ""
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        res = res + f"Prompt: {prompt!r}\nGenerated Text: {generated_text!r}\n"
    res = res + "-" * 80
    return res


def main():
    logger.info("start vllm_engine inference")
    args = get_args()

    sampling_config = {
        "num_completions": args.num_completions,  # 每个输入提示生成的独立完成项数量
        "logprobs": args.logprobs,  # 返回的 top token 的对数概率数量
        "max_tokens": args.max_tokens,  # 生成输出的最大 token 数量
        "top_p": args.top_p,  # 核采样的累积概率阈值
        "top_k": args.top_k,  # 采样时考虑的最高概率 token 的数量
        "temperature": args.temperature,  # 控制预测随机性的温度参数
        "detokenize": True  # 是否将生成的 token 转换回可读字符串
    }

    inference_engine = VLLMInferEngine(
        megatron_config=None,
        sampling_config=sampling_config,
        train_expert_parallel_size=1,
        infer_expert_parallel_size=1,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        train_tensor_parallel_size=args.tensor_parallel_size,
        train_pipeline_parallel_size=1,
        infer_tensor_parallel_size=args.tensor_parallel_size,
        infer_pipeline_parallel_size=1,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        load_format=args.load_format
    )

    if args.load_format == "megatron":
        tp_rank = ps._TP.rank_in_group
        weights_path = os.path.join(args.load, f"iter_0000001/mp_rank_{tp_rank:02}/model_optim_rng.pt")
        
        actor_weights = torch.load(weights_path)['model']
        actor_weights = replace_state_dict_name(
            actor_weights, 
            vllm_dict=inference_engine.model.state_dict(),
            arch=inference_engine.model.__class__.__name__)
        logger.info("sync_model_weights")
        inference_engine.sync_model_weights(actor_weights)

        logger.info("init_cache_engine")
        inference_engine.init_cache_engine()

    if args.task == "chat":
        chat_task(inference_engine, args.query)
    elif args.task == "generation":
        generate_task(inference_engine, args.query)


def chat_task(inference_engine, query):
    conversation = [
        {
            "role": "user",
            "content": query,
        },
    ]
    outputs = inference_engine.chat(conversation)
    res = process_outputs(outputs)
    logger.info('Query: {}'.format(query))
    logger.info('Responses:\n{}'.format(res))


def generate_task(inference_engine, query):
    outputs = inference_engine.llm.generate(
        prompts=[query],
        sampling_params=inference_engine.sampling_params,
    )
    res = process_outputs(outputs)
    logger.info('Query: {}'.format(query))
    logger.info('Responses:\n{}'.format(res))


def replace_state_dict_name(state_dict, vllm_dict, arch=None):
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
        # Deepseek add
        ("self_attention.linear_qb", "self_attn.q_b_proj"),
        ("self_attention.linear_kvb", "self_attn.kv_b_proj"),
        ("mlp.router.weight", "mlp.gate.weight"),
        ("mlp.router.expert_bias", "mlp.gate.e_score_correction_bias"),
        ("mlp.shared_experts.linear_fc1", "mlp.shared_experts.gate_up_proj"),
        ("mlp.shared_experts.linear_fc2", "mlp.shared_experts.down_proj"),
        ("mlp.experts.weight1", "mlp.experts.w13_weight"),
        ("mlp.experts.weight2", "mlp.experts.w2_weight"),
        ("self_attention.q_layernorm", "self_attn.q_a_layernorm"),
        ("self_attention.k_layernorm", "self_attn.kv_a_layernorm"),
    ]


    new_state_dict = {}
    for name, loaded_weight in state_dict.items():
        if "_extra_state" in name:
            continue
        if "Deepseek" in arch:
            name = _replace_name_m2v_deepseek(name, params_mapping)
        else:
            name = _replace_name_m2v(name, params_mapping)

        # the router bias in raw weight in fp32
        if "e_score_correction_bias" in name:
            loaded_weight = loaded_weight.to(vllm_dict[name].dtype)

        # to adapter 'copy_' in megatron weight loader to save memory
        if "mlp.experts" in name:
            loaded_weight = loaded_weight.view(vllm_dict[name].shape)

        new_state_dict[name] = loaded_weight
    return new_state_dict


def _replace_name_m2v(name, name_mapping):
    """
    Transfer state dict names from megatron to vllm.
    """
    for m_name, v_name in name_mapping:
        if m_name not in name:
            continue
        if "layers" in name:  # deal with decoder layers
            name = name.replace("decoder", "model")
            name_list = name.split(".")
            if "layer_norm_weight" in name_list or "layer_norm_bias" in name_list:
                param_name_list = name_list[:3]
                param_name_list.append(v_name)
                param_name = ".".join(param_name_list)
            else:
                param_name_list = name_list[:3]
                weight_or_bias = name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = ".".join(param_name_list)
            return param_name
        else:
            param_name = name.replace(m_name, v_name)
            return param_name
    return name


def _replace_name_m2v_deepseek(name, name_mapping):
    """
    Transfer state dict names from megatron to vllm.
    """
    for m_name, v_name in name_mapping:
        if m_name not in name:
            continue
        if "layers" in name:  # deal with decoder layers
            name = name.replace("decoder", "model")
        param_name = name.replace(m_name, v_name)
        return param_name
    return name


if __name__ == "__main__":
    main()
