import json
import argparse
import os
from typing import Dict, List
import logging as logger

import torch

from mindspeed_rl.workers.eplb import eplb


def find_replacement(current_group, expert_idx, weight, used_experts): 
    target_load = weight[expert_idx]
    candidates = [i for i in range(num_experts) if i not in current_group and i not in used_experts]
    if not candidates:
        raise ValueError(f" Unable to find a replacement for expert {expert_idx} ")
    # 找负载最接近的候选专家 索引下标
    closest = min(candidates, key=lambda x: abs(weight[x] - target_load))
    return closest


def tensor_to_json(tensor: torch.Tensor, num_gpus: int, output_path):
    num_layers, num_replicas = tensor.shape
    experts_per_device = num_replicas // num_gpus

    result = {
        "moe_layer_count": num_layers,
        "layer_list": []
    }

    for layer_id in range(num_layers):
        layer_data = {
            "layer_id": layer_id,
            "device_count": num_gpus,
            "device_list": []
        }

        layer_tensor = tensor[layer_id]

        for device_id in range(num_gpus):
            start_idx = device_id * experts_per_device
            end_idx = start_idx + experts_per_device
            device_expert = layer_tensor[start_idx:end_idx].tolist()

            device_data = {
                "device_id": device_id,
                "device_expert": device_expert
            }

            layer_data["device_list"].append(device_data)

        result["layer_list"].append(layer_data)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    logger.info(f" JSON 文件已保存到: {output_path}")


def load_and_aggregate(json_folder: str) -> torch.Tensor:
    # 构造单一文件路径
    filepath = os.path.join(json_folder, "token_collects_all.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # 读取并规范化键为 int，同时统计最大 layer / expert
    with open(filepath, "r") as f:
        raw = json.load(f)

    norm: Dict[int, Dict[int, int]] = {}
    max_layer = -1
    max_expert = -1

    for layer_id_str, experts_map in raw.items():
        layer_id = int(layer_id_str)
        expert_dict = {int(eid): int(cnt) for eid, cnt in experts_map.items()}
        norm[layer_id] = expert_dict

        if expert_dict:
            max_expert = max(max_expert, max(expert_dict.keys()))
        max_layer = max(max_layer, layer_id)

    # 初始化 [num_layers, num_experts] 矩阵
    num_layers = max_layer + 1 if max_layer >= 0 else 0
    num_experts = max_expert + 1 if max_expert >= 0 else 0
    mat = torch.zeros((num_layers, num_experts), dtype=torch.long)

    # 累加到矩阵
    for layer_id, experts in norm.items():
        for eid, cnt in experts.items():
            mat[layer_id, eid] += cnt

    return mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate {layer:{expert:num_tokens}} JSONs into a [layers,experts] tensor.")
    parser.add_argument("--json_folder", type=str, required=True, help="Folder containing per-rank JSON files") 
    parser.add_argument("--num_replicas", type=int, required=True)
    parser.add_argument("--num_groups", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    weight = load_and_aggregate(args.json_folder)
    num_replicas = args.num_replicas
    num_groups = args.num_groups
    num_nodes = args.num_nodes
    num_gpus = args.num_gpus
    output_path = args.output_path

    phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
    #一张卡上的冗余专家数
    experts_per_gpu = num_replicas // num_gpus 
    #总本地专家数
    num_experts = weight.size(1)

    for layer in range(weight.size(0)):
        weight_layer = weight[layer]
        for gpu in range(num_gpus):
            start = gpu * experts_per_gpu
            end = start + experts_per_gpu
            #遍历当前层 当前gpu上的专家列表
            group = phy2log[layer, start:end] 
            #用于记录当前组中已经出现过的专家
            seen = set() 
            #用于记录已经尝试替换但不合适的专家，避免重复尝试
            used_experts = set() 
            for j in range(experts_per_gpu):
                expert = group[j].item()
                if expert in seen:
                    replacement = find_replacement(group.tolist(), expert, weight_layer, used_experts)
                    while replacement in group.tolist():
                        used_experts.add(replacement)
                        replacement = find_replacement(group.tolist(), expert, weight_layer, used_experts)
                    phy2log[layer, start + j] = replacement
                seen.add(phy2log[layer, start + j].item())

    tensor_to_json(phy2log, num_gpus, output_path)