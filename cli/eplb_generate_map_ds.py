# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import json
import argparse
import os
from typing import Dict, List, Set
import logging as logger

import torch

from mindspeed_rl.workers.eplb import eplb


def find_replacement(
        current_group: List[int],
        expert_idx: int,
        weight: torch.Tensor,
        used_experts: Set[int]
) -> int:
    """Find a replacement expert with similar load characteristics.

    Args:
        current_group: List of expert indices currently assigned to the group.
        expert_idx: Index of the expert to be replaced.
        weight: Tensor containing expert load weights.
        used_experts: Set of experts already used in replacement attempts.

    Returns:
        Index of the replacement expert with closest load to target.

    Raises:
        ValueError: If no valid replacement expert can be found.
    """
    num_experts = weight.size(1)
    target_load = weight[expert_idx]
    candidates = [i for i in range(num_experts) if i not in current_group and i not in used_experts]
    if not candidates:
        raise ValueError(f"Unable to find a replacement for expert {expert_idx}")
    # Find candidate expert index with load closest to target
    closest = min(candidates, key=lambda x: abs(weight[x] - target_load))
    return closest


def tensor_to_json(tensor: torch.Tensor, num_gpus: int, output_path: str) -> None:
    """Convert expert assignment tensor to JSON format and save.

    Converts a 2D tensor of shape [num_layers, num_replicas] into a structured
    JSON format organized by layers and devices.

    Args:
        tensor: Expert assignment tensor of shape [num_layers, num_replicas].
        num_gpus: Number of GPU devices.
        output_path: Path to save the output JSON file.
    """
    # Data format conversion: Tensor -> JSON target format
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

    logger.info(f"JSON file saved to: {output_path}")


def load_and_aggregate(json_folder: str) -> torch.Tensor:
    """Load and aggregate expert usage statistics from JSON files.

    Reads multiple JSON files containing per-layer expert token counts,
    aggregates them into a single tensor of shape [num_layers, num_experts].

    Args:
        json_folder: Path to folder containing JSON files with expert statistics.

    Returns:
        Tensor of shape [num_layers, num_experts] containing aggregated counts.

    Raises:
        FileNotFoundError: If no JSON files are found in the specified folder.
    """
    # Read and aggregate all hotspot data
    files: List[str] = [
        os.path.join(json_folder, f)
        for f in os.listdir(json_folder)
        if f.endswith(".json")
    ]
    if not files:
        raise FileNotFoundError(f"No .json files found in: {json_folder}")
    loaded: List[Dict[int, Dict[int, int]]] = []
    max_layer = -1
    max_expert = -1
    for path in files:
        with open(path, "r") as f:
            raw = json.load(f)
        norm: Dict[int, Dict[int, int]] = {}
        for layer_id_str, experts_map in raw.items():
            layer_id = int(layer_id_str)
            expert_dict = {int(eid): int(cnt) for eid, cnt in experts_map.items()}
            norm[layer_id] = expert_dict
            if expert_dict:
                max_expert = max(max_expert, max(expert_dict.keys()))
            max_layer = max(max_layer, layer_id)
        loaded.append(norm)
    # Initialize [num_layers, num_experts] matrix
    num_layers = max_layer + 1 if max_layer >= 0 else 0
    num_experts = max_expert + 1 if max_expert >= 0 else 0
    mat = torch.zeros((num_layers, num_experts), dtype=torch.long)
    # Accumulate counts
    for data in loaded:
        for layer_id, experts in data.items():
            for eid, cnt in experts.items():
                mat[layer_id, eid] += cnt
    return mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate {layer:{expert:num_tokens}} JSONs into a [layers,experts] tensor."
    )
    parser.add_argument(
        "--json_folder",
        type=str,
        required=True,
        help="Folder containing per-rank JSON files"
    )
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
    # Compute redundant experts
    phy2log, log2phy, logcnt = eplb.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    experts_per_gpu = num_replicas // num_gpus
    num_experts = weight.size(1)
    # Deduplicate redundant experts
    for layer in range(weight.size(0)):
        weight_layer = weight[layer]
        for gpu in range(num_gpus):
            start = gpu * experts_per_gpu
            end = start + experts_per_gpu
            # Iterate through expert list of current layer
            group = phy2log[layer, start:end]
            # Track experts already seen in current group
            seen: Set[int] = set()
            # Track experts already tried but unsuitable for replacement
            used_experts: Set[int] = set()
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