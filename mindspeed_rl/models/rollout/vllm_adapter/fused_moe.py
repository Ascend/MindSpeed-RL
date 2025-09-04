# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.

import math
import os
import tempfile
import shutil
import json
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch_npu


from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed import GroupCoordinator
from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE, MoEConfig, UnquantizedFusedMoEMethod)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import FusedMoEState
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.fused_moe import fused_experts, fused_experts_with_all2allv, select_experts, fused_experts_with_all2all_buffer, fused_experts_with_mc2, AscendFusedMoE
import vllm_ascend


_EPLB_TOKEN_COLLECTS = False
_EPLB_TOEKN_SAVE_PATH = "./"


# EPLB global args
def set_EPLB_args(eplb_token_collects, eplb_token_save_path):
    global _EPLB_TOKEN_COLLECTS
    _EPLB_TOKEN_COLLECTS = eplb_token_collects

    global _EPLB_TOEKN_SAVE_PATH
    _EPLB_TOEKN_SAVE_PATH = eplb_token_save_path


def get_EPLB_args():
    return _EPLB_TOKEN_COLLECTS, _EPLB_TOEKN_SAVE_PATH


VLLM_ASCEND_MOE_ALL2ALL_BUFFER: bool = envs_ascend.VLLM_ASCEND_MOE_ALL2ALL_BUFFER


# EPLB add args: log2phy, global_redundant_expert_num
def fused_experts_with_mc2_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expert_map = kwargs.get("expert_map", None)
        global_redundant_expert_num = kwargs.get("global_redundant_expert_num", 0)

        # add moe_expert_num
        if expert_map is not None:
            kwargs["moe_expert_num"] = len(expert_map) + global_redundant_expert_num
        else:
            kwargs["moe_expert_num"] = global_redundant_expert_num

        return fn(*args, **kwargs)
    return wrapper


# # currently expert parallelism implemented with all2all
# # is under-optimized.
def fused_experts_with_all2all(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
    ep_group: GroupCoordinator = None,
    max_num_tokens: Optional[int] = None,
    log2phy: torch.Tensor = None,
    global_redundant_expert_num: int = 0,
):
    if log2phy is not None:
        topk_ids = log2phy[topk_ids]
    
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    num_experts = w1.shape[0]
    device = hidden_states.device

    # EPLB update global_num_experts
    if expert_map is not None:
        global_num_experts = len(expert_map) + global_redundant_expert_num
        local_num_experts = global_num_experts // ep_group.world_size
        row_idx_len = num_tokens * top_k
        row_idx = (torch.arange(0,
                                row_idx_len,
                                dtype=torch.int32,
                                device=device).view(top_k, -1).permute(
                                    1, 0).contiguous())
        active_num = max_num_tokens if max_num_tokens is not None else num_tokens
        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=active_num,
            ))

        global_expert_tokens = torch.bincount(expanded_expert_idx,
                                              minlength=global_num_experts)
        scatter_sizes = global_expert_tokens.view(ep_group.world_size,
                                                  -1).sum(-1)

        gather_sizes = torch.empty_like(scatter_sizes)
        dist.all_to_all_single(gather_sizes,
                               scatter_sizes,
                               group=ep_group.device_group)
        scatter_size_list = scatter_sizes.cpu().tolist()
        gather_size_list = gather_sizes.cpu().tolist()

        expanded_expert_idx = expanded_expert_idx % local_num_experts
        hidden_states = ep_group.all_to_all(hidden_states, 0, 0,
                                            scatter_size_list,
                                            gather_size_list)
        local_expert_idx = ep_group.all_to_all(expanded_expert_idx, 0, 0,
                                               scatter_size_list,
                                               gather_size_list)

        sorted_local_expert_idx, sorted_idx = torch.sort(local_expert_idx)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            sorted_local_expert_idx, local_num_experts).to(torch.int64)

        hidden_states = hidden_states[sorted_idx]
    else:
        row_idx_len = num_tokens * top_k
        row_idx = (torch.arange(0,
                                row_idx_len,
                                dtype=torch.int32,
                                device=topk_weights.device).view(
                                    top_k, -1).permute(1, 0).contiguous())
        active_num = max_num_tokens if max_num_tokens is not None else num_tokens
        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=active_num,
            ))

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
        expert_tokens = expert_tokens.to(torch.int64)

    w1 = w1.transpose(1, 2)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    hidden_states = torch.cat(gate_up_out_list, dim=0)
    hidden_states = torch_npu.npu_swiglu(hidden_states)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    hidden_states = torch.cat(down_out_list, dim=0)

    if expert_map is not None:
        resorted_idx = torch.argsort(sorted_idx)
        hidden_states = hidden_states[resorted_idx]
        hidden_states = ep_group.all_to_all(hidden_states, 0, 0,
                                            gather_size_list,
                                            scatter_size_list)

        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
    else:
        # implementation here when suitable operators become available.
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states, expert_tokens, 0


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe: MoEConfig = None):

        super().__init__(moe=moe)
        vllm_config = get_current_vllm_config()

        self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

        try:
            device_group = get_mc2_group().device_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = None


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        hidden_states_for_share: Optional[Any] = None,
        shared_experts: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:

        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        if global_num_experts == 256:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=top_k,  # topk当前写8
                bias=e_score_correction_bias,
                k_group=topk_group,  # fix: 4
                group_count=num_expert_group,  # fix 8
                group_select_mode=1,  # 0: group中的最大; 1: topk2.sum(fix)
                renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                routed_scaling_factor=1,
                eps=float(1e-20),
            )
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
            )
        
        # collect token number of per expert 
        eplb_token_collects, eplb_token_save_path = get_EPLB_args()
        if eplb_token_collects:
            self.layers = layer.moe_instance_id
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # cur_rank
            local_stats = {self.layers: {eid: 0 for eid in range(global_num_experts)}}
            unique_ids, counts = torch.unique(topk_ids, return_counts=True)
            for uid, cnt in zip(unique_ids.tolist(), counts.tolist()):
                local_stats[self.layers][uid] += cnt
                
            all_stats = [None for _ in range(world_size)]
            dist.all_gather_object(all_stats, local_stats)

            # all ranks
            if rank == 0:
                # merge
                merged_stats = {}
                for rank_stats in all_stats:
                    for layer_id, e_dict in rank_stats.items():
                        if layer_id not in merged_stats:
                            merged_stats[layer_id] = {eid: 0 for eid in range(global_num_experts)}
                        for eid, cnt in e_dict.items():
                            merged_stats[layer_id][eid] += cnt

                filepath = os.path.join(eplb_token_save_path, f"token_collects_all.json")
                if os.path.exists(filepath):
                    with open(filepath, "r") as f:
                        old_stats = json.load(f)
                    old_stats = {int(k): {int(e): v for e, v in vdict.items()} for k, vdict in old_stats.items()}
                else:
                    old_stats = {}

                # merge old_stats
                for layer_id, e_dict in merged_stats.items():
                    if layer_id not in old_stats:
                        old_stats[layer_id] = {eid: 0 for eid in range(global_num_experts)}
                    for eid, cnt in e_dict.items():
                        old_stats[layer_id][eid] += int(cnt)

                with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(filepath)) as tmpfile:
                    json.dump(old_stats, tmpfile, indent=2)
                    tmpfile.flush()
                    os.fsync(tmpfile.fileno())
                    temp_name = tmpfile.name
                shutil.move(temp_name, filepath)

        topk_weights = topk_weights.to(x.dtype)

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance and not self.use_aclgraph:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        fused_moe_state = get_forward_context().fused_moe_state
        if fused_moe_state == FusedMoEState.MC2:
            mc2_mask = kwargs.get("mc2_mask", None)
            # EPLB add args: log2phy, global_redundant_expert_num
            return fused_experts_with_mc2(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                expert_map=expert_map,
                moe_all_to_all_group_name=self.moe_all_to_all_group_name,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
                shared_experts=shared_experts,
                is_torchair=self.torchair_graph_enabled,
                hidden_states_for_share=hidden_states_for_share,
                mc2_mask=mc2_mask,
            )
        elif fused_moe_state == FusedMoEState.AllGather:
            max_num_tokens = self.max_num_batched_tokens if self.use_aclgraph else None
            return fused_experts(hidden_states=x,
                                 w1=layer.w13_weight,
                                 w2=layer.w2_weight,
                                 topk_weights=topk_weights,
                                 topk_ids=topk_ids,
                                 top_k=top_k,
                                 expert_map=expert_map,
                                 max_num_tokens=max_num_tokens)
        elif VLLM_ASCEND_MOE_ALL2ALL_BUFFER:
            max_num_tokens = self.max_num_batched_tokens if self.use_aclgraph else None
            return fused_experts_with_all2all_buffer(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                max_model_len=self.max_model_len,
                global_batch_size=self.global_batch_size,
                expert_map=expert_map,
                ep_group=get_ep_group(),
                max_num_tokens=max_num_tokens)
        elif fused_moe_state == FusedMoEState.All2AllSeq:
            token_dispatcher = kwargs.get("token_dispatcher")
            return fused_experts_with_all2allv(
                token_dispatcher=token_dispatcher,
                probs=topk_weights,
                routing_map=topk_ids,
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
            )
        else:
            max_num_tokens = self.max_num_batched_tokens if self.use_aclgraph else None
            # EPLB add args: log2phy, global_redundant_expert_num
            return fused_experts_with_all2all(hidden_states=x,
                                              w1=layer.w13_weight,
                                              w2=layer.w2_weight,
                                              topk_weights=topk_weights,
                                              topk_ids=topk_ids,
                                              top_k=top_k,
                                              expert_map=expert_map,
                                              ep_group=get_ep_group(),
                                              log2phy=log2phy,
                                              global_redundant_expert_num=global_redundant_expert_num,
                                              max_num_tokens=max_num_tokens)


def AscendFusedMoE_wrapper(init_fn):
    @wraps(init_fn)
    def wrapper(self, *args, **kwargs):
        # original __init__
        init_fn(self, *args, **kwargs)

        # patch vllm-ascend
        ascend_config = get_ascend_config()
        expert_map_path = ascend_config.expert_map_path

        if expert_map_path and os.path.exists(expert_map_path):
            self.expert_load_balancer = ExpertLoadBalancer(
                expert_map_path, self.global_num_experts)
            self.local_num_experts, self.expert_map = (
                self.expert_load_balancer.get_rank_placement_map(
                    self.moe_instance_id, self.ep_rank))
            self.log2phy = self.expert_load_balancer.get_rank_log2phy_map(
                self.moe_instance_id, self.ep_rank).npu()
            self.global_redundant_expert_num = (
                self.expert_load_balancer.get_global_redundant_expert_num())
    return wrapper


vllm_ascend.ops.fused_moe.fused_experts_with_mc2 = fused_experts_with_mc2_wrapper(fused_experts_with_mc2)
vllm_ascend.ops.fused_moe.fused_experts_with_all2all = fused_experts_with_all2all
vllm_ascend.ops.fused_moe.AscendUnquantizedFusedMoEMethod = AscendUnquantizedFusedMoEMethod
vllm_ascend.ops.fused_moe.AscendFusedMoE.__init__ = AscendFusedMoE_wrapper(AscendFusedMoE.__init__)