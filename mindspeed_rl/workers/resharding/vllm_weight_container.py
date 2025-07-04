# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file contains a Megatron style Hybrid Model that shares the weights of the actor with the inference engine.
"""

import hashlib
import re
from functools import partial
from typing import List

import torch
import torch.distributed as dist
import numpy as np

from torch.distributed import new_group
import vllm.distributed.parallel_state as ps

from mindspeed_rl.workers.resharding.memory_buffer import build_model_weight_buffer, calc_padded_numel
import mindspeed_rl.workers.resharding.utils
from mindspeed_rl.workers.resharding.utils import get_tensor_parallel_partition_dim, tp_md5_validate, \
    update_md5_by_rank, compute_md5, validate_md5, _build_infer_param_dict, get_tp_allgather_group, \
    get_tp_allgather_world_size, is_tensor_parallel_param, get_tp_group, is_fake_tp_param
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.utils import is_multimodal

logger = Loggers(__name__)


class MegatronStyleVllmWeightContainer:

    def __init__(self, megatron_model, vllm_model, model_config, infer_tensor_parallel_size,
                 infer_pipeline_parallel_size,
                 infer_expert_parallel_size,
                 num_layer_list,
                 moe_tp_extend_ep=False,
                 parallel_state=None,
                 weight_adaptor=None,
                 enable_validate=False,
                 noop_layers=None) -> None:
        """ Megatron style vllm weight container.

        Arguments:
            megatron_model (nn.Module or nn.ModuleList): Megatron model used for training
            vllm_model (nn.Module or nn.ModuleList): VLLM model used for inference
            model_config (MegatronConfig): Model configuration
            infer_tensor_parallel_size (int): Inference tensor parallel size
            infer_pipeline_parallel_size (int): Inference pipeline parallel size
            infer_expert_parallel_size (int): Inference expert parallel size
            num_layer_list (str): a list of number of layers, seperated by comma; e.g., 4,4,4,4.
            moe_tp_extend_ep (bool): Controls whether expert model parameters are split across multiple GPUs.
            parallel_state (ModuleType): Megatron parallel state of the model.
            weight_adaptor (WeightAdaptor): Provides a set of tools to transfer from training weight to inference weight.
            enable_validate (bool): Whether to enable communication data validate.
        """

        self.vllm_model = vllm_model
        self.model_config = model_config
        self.megatron_model = megatron_model
        self.parallel_state = parallel_state
        self.weight_adaptor = weight_adaptor
        self._num_hidden_layers = self.model_config.num_hidden_layers # 通过tokenier路径下的config.json获取hf的模型
        self._noop_layers = None
        if noop_layers is not None:
            self._noop_layers = [int(layer_idx) for layer_idx in noop_layers.split(',')]
            self._num_hidden_layers += len(self._noop_layers)

        # pp configs
        self._pp_rank = self.parallel_state.get_pipeline_model_parallel_rank()
        self._pp_group = self.parallel_state.get_pipeline_model_parallel_group()
        self._pp_size = self.parallel_state.get_pipeline_model_parallel_world_size()
        self._world_size = dist.get_world_size()

        ## vpp
        self._num_layer_list = self._build_num_layer_list(num_layer_list)
        self._vpp_rank = self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK if self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK else 0
        self._vpp_size = self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE if self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE else 1
        self._vpp_layer_list = self._build_vpp_layer_list(self._num_layer_list)
        ## _noop_layers
        self._global2local_map = self._build_global2local_map(self._vpp_layer_list, self._vpp_size, self._noop_layers) if self._noop_layers is not None else None

        # tp configs
        self._tp_size = self.parallel_state.get_tensor_model_parallel_world_size()
        self._tp_group = self.parallel_state.get_tensor_model_parallel_group()

        # ep configs
        self._ep_size = self.parallel_state.get_expert_model_parallel_world_size()

        if moe_tp_extend_ep:
            self._ep_group = self.parallel_state.get_tensor_and_expert_parallel_group()
            self._ep_size = self._tp_size * self._ep_size
        else:
            self._ep_group = self.parallel_state.get_expert_model_parallel_group()

        if hasattr(self.model_config, "n_routed_experts"):
            self.num_experts = self.model_config.n_routed_experts
            self.num_local_experts = self.num_experts // self._ep_size
        elif hasattr(self.model_config, "num_experts"):
            self.num_experts = self.model_config.num_experts
            self.num_local_experts = self.num_experts // self._ep_size

        # infer configs
        self._infer_tp_size = infer_tensor_parallel_size
        self._infer_pp_size = infer_pipeline_parallel_size
        self._infer_ep_size = infer_expert_parallel_size
        self.moe_tp_extend_ep = moe_tp_extend_ep

        # TODO: infer_expert_tensor_parallel_size and num_process is fixed.
        self.infer_expert_tensor_parallel_size = 1
        self.num_process = 1
        self._infer_ep_size = self._infer_ep_size * self._infer_tp_size
        self.experts_memory_expand_N = self._infer_ep_size // self._ep_size

        # validate parallel configs
        self._validate_parallel_config()

        # md5 validate
        self.enable_validate = enable_validate
        self.origin_params_for_md5 = None
        self.infer_params_for_md5 = None

        self._rank = dist.get_rank()
        self._init_tensor_model_parallel_allgather_group()
        self._init_pipeline_model_parallel_allgather_group()
        self._init_tensor_model_parallel_split_group()
        self._init_weight_buffers()

    def _validate_parallel_config(self):
        if self._infer_pp_size != 1:
            raise ValueError("infer_pp_size != 1 not supported yet")

        if self._infer_ep_size % self._ep_size != 0:
            raise ValueError("The training expert size should be divisibled by the inference expert size.")
        if self._ep_size > 1 and not self.moe_tp_extend_ep:
            raise ValueError("To enable training EP, you need to enable moe_tp_extend_ep and use GroupedMLP.")
        if self._pp_size < self._infer_pp_size:
            raise ValueError(
                "The training pipeline parallel size should be greater than or equal to the inference pipeline "
                "parallel size.")
        if self._pp_size % self._infer_pp_size != 0:
            raise ValueError(
                "The training pipeline parallel size should be an integer multiple of the inference pipeline parallel "
                "size.")
        if self._tp_size > self._infer_tp_size and self._tp_size % self._infer_tp_size != 0:
            raise ValueError(
                "The training tensor parallel size should be an integer multiple of the inference tensor parallel size.")
        # For tp increase, train_tp * dp >= infer_tp, train_tp * dp % infer_tp == 0
        if self._tp_size < self._infer_tp_size:
            if (self._world_size // self._pp_size < self._infer_tp_size or
                    (self._world_size // self._pp_size) % self._infer_tp_size != 0):
                raise ValueError(
                    f"Do not support split train tp size {self._tp_size} to infer tp size {self._infer_tp_size} "
                    f"with train dp size {(self._world_size // (self._tp_size * self._pp_size))}.")

    def get_infer_params(self):
        """
        return the whole weight state dict for vllm, but in megatron style and names,
        needs megatron weight loader to further transfer for vllm
        """

        self._update_weight_buffers_intra_pp()
        self._update_weight_buffers_inter_pp()

        # 执行_update_weight_buffers_ep+_send_receive_experts的前提条件
        if(self.moe_tp_extend_ep and self._infer_ep_size >= self._ep_size):
            self._update_weight_buffers_ep()
            self._send_receive_experts()

        params = self._get_all_params()

        params = _build_infer_param_dict(params=params)
        return params

    def _build_num_layer_list(self, num_layer_list):
        if num_layer_list:
            # multimodal num_layer_list is a list of lists, including vit and llm layers
            if isinstance(num_layer_list[0], List):
                return num_layer_list
            return [int(num_layers) for num_layers in num_layer_list.split(',')]
        if self._num_hidden_layers % self._pp_size != 0:
            raise ValueError("num_layers % pp_size == 0, please specify num_layer_list")
        return [self._num_hidden_layers // self._pp_size for _ in range(self._pp_size)]

    def _build_vpp_layer_list(self, num_layer_list):
        if self._vpp_size <= 1:
            return num_layer_list
        for layers_in_pp_rank in num_layer_list:
            if layers_in_pp_rank % self._vpp_size != 0:
                raise ValueError("num_layers_per_pp % vpp_size != 0, please specify pp_size and vpp_size")
        return [int(layers_in_pp_rank / self._vpp_size) for layers_in_pp_rank in num_layer_list]

    def _build_global2local_map(self, layer_list, vpp_size, noop_layers):
        stage_layers_num = sum(layer_list)
        glb2local_map = []
        for vpp_rank in range(vpp_size):
            start_layer = vpp_rank * stage_layers_num
            for _, layers_in_vpp_rank in enumerate(layer_list):
                layer_idx_list = [
                    layer_idx for layer_idx in range(start_layer, start_layer + layers_in_vpp_rank)
                    if layer_idx not in noop_layers
                ]
                glb2local_map += [layer_idx % layers_in_vpp_rank for layer_idx in layer_idx_list]
                start_layer += layers_in_vpp_rank

        return glb2local_map

    def _unwrap_megatron_model(self, model):
        """
        Remove consecutive 'module.' prefixes from the model based on the state_dict's first key.
        This method only removes 'module.' from the beginning of the key and ignores other occurrences.
        """
        unwraped_model = []
        for model_chunk in model:
            first_key = list(dict(model_chunk.named_parameters()).keys())[0]
            while first_key.startswith("module."):
                model_chunk = model_chunk.module
                first_key = first_key[len("module."):]
            unwraped_model.append(model_chunk)
        return unwraped_model

    def _init_weight_buffers(self):
        """
        Build buffers from vllm state dict. Totally build train pp_size buffers, each buffer corresponds to a pack of megatron weight.
        Return a list of buffers, and a reference dict megatron_param_name->buffer.
        """
        vllm_names = list(dict(self.vllm_model.named_parameters()).keys()) # 获取每个pp内部的weights name
        if is_multimodal():
            layers_num = [sum(num_layer_list) for num_layer_list in self._num_layer_list]
        else:
            layers_num = sum(self._num_layer_list)
        self.weight_names_per_pp = self.weight_adaptor.get_weight_names_per_pp(self._vpp_layer_list, vllm_names,
                                                                               layers_num, self._vpp_size, self._noop_layers)

        self.weight_buffers = build_model_weight_buffer(self.vllm_model, self.weight_names_per_pp,
                                                        self.weight_adaptor.get_weight_buffer_meta
                                                        )

    def trans_ep_params_to_tp(self, megatron_param, name):
        """
        Transfer a GroupedMLP from EP to TP. Currently, assert EP==TP.
        e.g. EP=2 -> TP=2
        Assume we have 4 experts in total.
        We here note e0 for expert 0, and [a0, b0] for the tensor parallel weight for expert 0,
        so we can denote first half weights for all the 4 experts as a0-4 .
        For EP to TP transfer, what we actually need to do is:
        [[e0-1], [e2-3]] -> [[a0-4], [b0-4]]
        We first build a matrix, each column is a rank before transfer, and each row is a rank after transfer.
                    ep0   ep1
                [
        a0-4        a0-1, a2-3,
        b0-4        b0-1, b2-3,
                ]
        When we get this matrix, we only need to do All2All to transfer EP to TP on the EP group.

        So, for ep_rank 0 we need to build [a0-1, b0-1] from [e0-1], i.e.
        [e0-1] <=> [e0, e1] <=> [a0, a1, b0, b1] -> [a0, a1, b0, b1] <=> [a0-1, b0-1]

        For DSv3 model, this function only handles decoder.layers.x.mlp.experts.weight1 and
        decoder.layers.x.mlp.experts.weight2.
        In which, weight 1 is cut by column and contains both gate and up;
              and weight 2 is cut by row.
        """

        # the true ep size, equal to ep_size * tp_size when tp_extend_ep
        if self._ep_size == 1:
            return megatron_param

        tp_size = self._infer_tp_size

        num_experts = self.num_local_experts

        # 1. build ep_tp matrix buffer
        # For megatron param [e0, e1], we make it [a0, a1, b0, b1], in which e0 == [a0, b0]

        # weight1: column cut, be like [g0, u0, g1, u1, ...]
        if 'weight1' in name:

            hidden_size = megatron_param.shape[0]
            megatron_param = torch.cat(megatron_param.view(num_experts, hidden_size, -1).unbind(0), dim=1)

            # We can treat both the gate and the up weight as 2 independent experts.
            num_experts *= 2

        # weight2: row cut, be like [ d0, d1, d2, ...]^T
        elif 'weight2' in name:

            hidden_size = megatron_param.shape[1]
            megatron_param = torch.cat(megatron_param.view(num_experts, -1, hidden_size).unbind(0), dim=0)

            # transpose params to handle uniformly with column cut
            megatron_param = megatron_param.t()

        else:
            return megatron_param

        # chunk to tp * ep parts
        chunks = torch.chunk(megatron_param, tp_size * num_experts, dim=1)

        # re-select by tp-ep order
        # e.g. TP=2 num_experts=4, old order [1,2,3,4,5,6,7,8],  new order [1,3,5,7,2,4,6,8]
        new_order = []
        for i in range(tp_size):
            for j in range(num_experts):
                new_order.append(chunks[i + j * tp_size])

        reordered_x = torch.cat(new_order, dim=1)
        final_chunks = torch.chunk(reordered_x, tp_size, dim=1)

        # 2. do AlltoAll communication
        input_tensor_list = [chunk.contiguous() for chunk in final_chunks]
        output_tensor_list = [torch.empty_like(chunk) for chunk in input_tensor_list]
        torch.distributed.all_to_all(
            output_tensor_list,
            input_tensor_list,
            group=self._ep_group,
            async_op=False
        )
        total_experts = self.num_local_experts * tp_size
        res = torch.cat(output_tensor_list, dim=1).reshape(hidden_size, total_experts, -1)
        if 'weight2' in name:
            return res.permute(1, 2, 0).contiguous()
        return res.permute(1, 0, 2).contiguous()


    def _update_weight_buffers_intra_pp(self):
        """
        Here, we only update the current training pp_rank's buffer.
        """

        def _transfer_from_megatron_division(megatron_param, name):
            """
            Deal with the tp_param form train_tp to infer_tp.
            """
            infer_param = self.allgather_tp_param(megatron_param, name)
            infer_param = self.split_tp_params(infer_param, name)
            infer_param = self.trans_ep_params_to_tp(infer_param, name)
            return infer_param

        pp_rank = self._pp_rank
        weight_names = self.weight_names_per_pp[pp_rank]
        weight_names_meta = self.weight_adaptor.convert_weight_name_meta(weight_names)
        true_megatron_model = self._unwrap_megatron_model(self.megatron_model)
        normal_layer_func = partial(self.weight_adaptor.global2local_layer, num_layer_list=self._vpp_layer_list, global2local_map=self._global2local_map)
        name_pairs = sorted(list(set([(name, vpp_rank, self.weight_adaptor.replace_name_i2t(normal_layer_func(name, vpp_rank=vpp_rank)))
                                    for vpp_rank, names_per_vpp in enumerate(weight_names_meta) for name in names_per_vpp])))

        if self.enable_validate:
            self.origin_params_for_md5 = hashlib.md5()
            self.infer_params_for_md5 = [hashlib.md5() for _ in range(get_tp_allgather_world_size())]

        # 检查 linear_fc1 和 linear_fc2 权重形状是否符合特定关系（fc1 包含门控和扩展参数，因此大小是 fc2 的两倍）。不符合条件的模型不被支持。
        for _, vpp_rank, megatron_name in name_pairs:
            if not megatron_name.startswith("image_encoder") and megatron_name.endswith("linear_fc1.weight"):
                fc2_name = megatron_name.replace("linear_fc1", "linear_fc2")
                megatron_param_fc1 = dict(true_megatron_model[vpp_rank].named_parameters())[megatron_name]
                megatron_param_fc2 = dict(true_megatron_model[vpp_rank].named_parameters())[fc2_name]
                if megatron_param_fc1.shape[0] * megatron_param_fc1.shape[1] != megatron_param_fc2.shape[0] * \
                        megatron_param_fc2.shape[1] * 2:
                    raise ValueError("Only implemented for Llama model which linear_fc1 contains gate and up params.")

        weight_buffer = self.weight_buffers[pp_rank]
        megatron_params_dict = {}
        for vpp_rank in range(self._vpp_size):
            megatron_params_dict.update({vpp_rank: dict(true_megatron_model[vpp_rank].named_buffers())})
            megatron_params_dict[vpp_rank].update(true_megatron_model[vpp_rank].named_parameters())
            megatron_params_dict[vpp_rank] = self.weight_adaptor.adjust_megatron_param_dict(megatron_params_dict[vpp_rank], self._tp_size)

        for hf_name, vpp_rank, megatron_name in name_pairs:
            if((self._infer_ep_size > 1 or self._ep_size > 1) and "mlp.experts" in megatron_name):
                pass
            else:
                megatron_param = megatron_params_dict[vpp_rank][megatron_name]
                param = _transfer_from_megatron_division(megatron_param, megatron_name)
                weight_buffer.copy_by_name(hf_name, param)

        # tp md5 validate
        if self.enable_validate:
            tp_md5_validate(self.infer_params_for_md5, self.origin_params_for_md5,
                            f"rank[{self._rank}] tp params allgather")

    def _update_weight_buffers_ep(self):
        # 构造临时的experts_memory_buffers
        for cur_pp_rank in range(self._pp_size):
            pp_rank = self._pp_rank
            from mindspeed_rl.workers.resharding.memory_buffer import build_experts_memory_buffer, get_weight_buffer_meta_from_buffer
            # Step1 在当前的PP_rank中，设置一个临时的exprts_buffer
            combined_names_per_pp = []
            vpp_stages = self.weight_names_per_pp[cur_pp_rank]
            for weight_names_per_stage in vpp_stages:
                combined_names_per_pp.extend(weight_names_per_stage)
            self.weight_buffer_meta = self.weight_adaptor.get_weight_buffer_meta(self.vllm_model, combined_names_per_pp)
            self.experts_weight_buffer_meta = get_weight_buffer_meta_from_buffer(self.weight_buffer_meta)
            self.experts_memory_buffers = build_experts_memory_buffer(self.experts_weight_buffer_meta, self.experts_memory_expand_N)

            # Step2 将weights_buffer上对应的权重放到experts_buffer中
            if(cur_pp_rank == pp_rank):
                weight_names = self.weight_names_per_pp[pp_rank]
                weight_names_meta = self.weight_adaptor.convert_weight_name_meta(weight_names)
                normal_layer_func = partial(self.weight_adaptor.global2local_layer, num_layer_list=self._vpp_layer_list, global2local_map=self._global2local_map)
                name_pairs = sorted(list(set([(name, vpp_rank, self.weight_adaptor.replace_name_i2t(normal_layer_func(name, vpp_rank=vpp_rank)))
                                      for vpp_rank, names_per_vpp in enumerate(weight_names_meta) for name in names_per_vpp])))
                true_megatron_model = self._unwrap_megatron_model(self.megatron_model)

                megatron_params_dict = {}
                # 拿到当前pp的所有权重
                for vpp_rank in range(self._vpp_size):
                    megatron_params_dict.update({vpp_rank: dict(true_megatron_model[vpp_rank].named_buffers())})
                    megatron_params_dict[vpp_rank].update(true_megatron_model[vpp_rank].named_parameters())
                    megatron_params_dict[vpp_rank] = self.weight_adaptor.adjust_megatron_param_dict(megatron_params_dict[vpp_rank], self._tp_size)

                for hf_name, vpp_rank, megatron_name in name_pairs:
                    if((self._infer_ep_size > 1 or self._ep_size > 1) and "mlp.experts" in megatron_name):
                        megatron_param = megatron_params_dict[vpp_rank][megatron_name]
                        dtype = self.experts_weight_buffer_meta[hf_name]['dtype']
                        self.experts_memory_buffers[dtype].copy_by_name(hf_name, megatron_param)

            # Step3 后续的操作可以复用
            global_src = dist.get_global_rank(group=self._pp_group, group_rank=cur_pp_rank)

            # broadcast专家权重（experts memory buffer中的）
            for dtype, experts_memory_buffer in self.experts_memory_buffers.items():
                dist.broadcast(tensor=experts_memory_buffer.data, src=global_src, group=self._pp_group, async_op=False)
                ep_expand_rank = self._rank // self._ep_size

                # 获取对应的dtype
                for name, tensor_indices_value in sorted(experts_memory_buffer.tensor_indices.items()):
                    shape = tensor_indices_value[1]  # 是*N的
                    index = ep_expand_rank % self.experts_memory_expand_N
                    experts_tensor = experts_memory_buffer.get_by_name(name)
                    experts_tensor_reshape = experts_tensor.view(shape)
                    weight_tensor_infer = experts_tensor_reshape[index]
                    self.weight_buffers[cur_pp_rank].copy_by_name(name, weight_tensor_infer)

            # 卸载专家的buffer
                experts_memory_buffer = None
                self.experts_memory_buffers[dtype] = None

            for memory_buffer in self.experts_memory_buffers.values():
                memory_buffer = None
            self.experts_memory_buffers = None


    def _update_weight_buffers_inter_pp(self):
        """
        Update weight buffers by gathering weights from other pp stage.

        """
        for cur_pp_rank in range(self._pp_size):
            global_src = dist.get_global_rank(group=self._pp_group, group_rank=cur_pp_rank)
            for memory_buffer in self.weight_buffers[cur_pp_rank].memory_buffers.values():
                dist.broadcast(tensor=memory_buffer.data, src=global_src, group=self._pp_group, async_op=False)
            if self.enable_validate:
                md5_tensor = compute_md5(self.weight_buffers[cur_pp_rank])
                if self._rank == global_src:
                    dist.broadcast(md5_tensor, group=self._pp_group, src=global_src, async_op=False)
                else:
                    md5_tensor_src = torch.zeros_like(md5_tensor, dtype=torch.int64, device=torch.cuda.current_device())
                    dist.broadcast(md5_tensor_src, group=self._pp_group, src=global_src, async_op=False)
                    validate_md5(md5_tensor_src, md5_tensor, f"rank[{self._rank}] pp resharding params")


    def get_expert_router(self, cur_rank, train_tp_ep_size, infer_tp_ep_size, world_size):
        for tp_ep_group_id in range(world_size // infer_tp_ep_size):
            tp_ep_group = [i for i in range(tp_ep_group_id * infer_tp_ep_size, (tp_ep_group_id + 1) * infer_tp_ep_size)]
            if cur_rank in tp_ep_group:
                self.INFER_TP_EP_GROUP = tp_ep_group
        stride = infer_tp_ep_size // train_tp_ep_size
        dev_array = np.array(self.INFER_TP_EP_GROUP).reshape(stride, train_tp_ep_size)
        src_router = np.squeeze(dev_array.transpose().reshape(1, infer_tp_ep_size)).tolist()
        src = src_router[cur_rank % infer_tp_ep_size]
        dst = self.INFER_TP_EP_GROUP[src_router.index(cur_rank)]
        return src, dst

    def _send_receive_experts(self):
        cur_rank = dist.get_rank()
        src_rank, dst_rank = self.get_expert_router(cur_rank, self._ep_size, self._infer_ep_size, self._world_size)
        for cur_pp_rank in range(self._pp_size):
            for memory_buffer in self.weight_buffers[cur_pp_rank].memory_buffers.values():
                for name in sorted(memory_buffer.tensor_indices.keys()):
                    if "mlp.experts" in name:
                        # 做收发
                        tensor_to_send = memory_buffer.get_by_name(name)
                        tensor_to_replace = torch.empty_like(tensor_to_send)
                        send_op = dist.P2POp(dist.isend, tensor_to_send, dst_rank)
                        recv_op = dist.P2POp(dist.irecv, tensor_to_replace, src_rank)
                        reqs = dist.batch_isend_irecv([send_op, recv_op])
                        for req in reqs:
                            req.wait()
                        memory_buffer.copy_by_name(name, tensor_to_replace)

    def _get_all_params(self):
        """Get all the parameters of the models in all pp ranks

        Returns:
            params: List[List[Dict[str, Tensor]]]: a list of parameters in all pp, where each is a list of dict
                tensors of each model chunk

        """
        params = []
        for pp_rank in range(self._pp_size):
            params.append([])
            params[pp_rank].append({})
            model_chunk_idx = 0
            weight_buffer = self.weight_buffers[pp_rank]
            for name in weight_buffer.weight_names:
                if 'lora' in name:
                    raise ValueError("not support lora now")
                params[pp_rank][model_chunk_idx][name] = weight_buffer[name]
        return params

    def _init_tensor_model_parallel_allgather_group(self):
        if self._tp_size < self._infer_tp_size:
            return
        if self._tp_size % self._infer_tp_size != 0:
            raise ValueError("self._tp_size must be divisible by self._infer_tp_size")
        tp_allgather_size = self._tp_size
        if mindspeed_rl.workers.resharding.utils._TP_ALLGATHER_GROUP is not None:
            raise RuntimeError("Group for allgather tensor model parallel weight is already initialized")
        num_groups = self._world_size // tp_allgather_size
        for i in range(num_groups):
            ranks = range(i * tp_allgather_size, (i + 1) * tp_allgather_size)
            group = new_group(ranks=ranks)
            if self._rank in ranks:
                mindspeed_rl.workers.resharding.utils._TP_ALLGATHER_GROUP = group

    def _init_pipeline_model_parallel_allgather_group(self):
        if self._pp_size < self._infer_pp_size:
            raise NotImplementedError("Not implemented for infer_pp > train_pp")
        if self._pp_size % self._infer_pp_size != 0:
            raise ValueError(
                "Pipeline model parallel size must be a multiple of inference pipeline model parallel size")
        pp_allgather_size = self._pp_size // self._infer_pp_size
        if mindspeed_rl.workers.resharding.utils._PP_ALLGATHER_GROUP is not None:
            raise RuntimeError("Group for allgather pipeline model parallel weight is already initialized")
        global_pp_group_ranks_list = []
        for pp_group_index in range(self._world_size // self._pp_size):
            self_pp_group_ranks_list = []
            for ranks in range(pp_group_index, pp_group_index + self._world_size, self._world_size // self._pp_size):
                self_pp_group_ranks_list.append(ranks)
            global_pp_group_ranks_list.append(self_pp_group_ranks_list)

        for pp_group_ranks in global_pp_group_ranks_list:
            splited_pp_group_ranks = np.array_split(pp_group_ranks, self._infer_pp_size)
            for ranks in splited_pp_group_ranks:
                cur_group = new_group(ranks=ranks)
                if self._rank in ranks:
                    mindspeed_rl.workers.resharding.utils._PP_ALLGATHER_GROUP = cur_group

    def _init_tensor_model_parallel_split_group(self):
        if self._tp_size >= self._infer_tp_size:
            return
        if self._infer_tp_size % self._tp_size != 0:
            raise ValueError("self._infer_tp_size must be a multiple of self._tp_size")
        if mindspeed_rl.workers.resharding.utils._TP_GROUP is not None:
            raise RuntimeError("Group for tensor model parallel weight is already initialized")
        if self._infer_tp_size > self._tp_size:
            mindspeed_rl.workers.resharding.utils._TP_GROUP = self.parallel_state.get_tensor_model_parallel_group()

    def _default_tp_concat_fn(self, name, param, infer_params):
        """
        name: name of the parameter
        param: training_utils parameters
        infer_params (List[torch.Tensor]): a list of parameters all-gathered from micro_dp_group
        definition so that it is model-agnostic. If the model doesn't implement this function,
        we can throw an error to force user disable TP HybridEngine.
        """

        if 'projector' not in name and 'linear_fc1' in name:
            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            for infer_param in infer_params:
                gate, up = infer_param.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            infer_params = torch.cat((gate, up), dim=0)

        else:
            # concat tensor
            infer_params = torch.cat(infer_params, dim=get_tensor_parallel_partition_dim(param))

        return infer_params

    def split_tp_params(self, param, name):
        """
        name: name of the parameter
        param: training_utils parameters

        1. get full train params through allgather
        2. split train_tp params into groups (size: infer_tp_size)
        3. return the corresponding param from group based on infer tp rank
        """
        if self._infer_tp_size <= self._tp_size or is_fake_tp_param(name, self.moe_tp_extend_ep):
            return param

        tp_group = get_tp_group()

        if is_tensor_parallel_param(param):
            if self._tp_size > 1:
                # allocate a new tensor with proper size
                infer_params = [torch.empty_like(param) for _ in range(self._tp_size)]
                torch.distributed.all_gather(infer_params, param, group=tp_group)
            else:
                infer_params = [param]
            if "linear_fc1.weight" in name:
                # if the tensor is gate and proj
                gate_lst = []
                up_lst = []
                for infer_param in infer_params:
                    gate, up = infer_param.chunk(2)
                    gate_lst.append(gate)
                    up_lst.append(up)
                gate = torch.cat(gate_lst, dim=0)
                up = torch.cat(up_lst, dim=0)

                gate_splits = torch.chunk(gate, self._infer_tp_size, dim=0)
                up_splits = torch.chunk(up, self._infer_tp_size, dim=0)

                new_params_list = [
                    torch.cat([gate_splits[i], up_splits[i]], dim=0)
                    for i in range(self._infer_tp_size)
                ]
            else:
                partition_dim = get_tensor_parallel_partition_dim(param)
                infer_params = torch.cat(infer_params, dim=partition_dim)
                split_params = torch.chunk(infer_params, self._infer_tp_size, dim=partition_dim)
                new_params_list = list(split_params)

            # make_list
            param_list = new_params_list

        else:
            param_list = [param] * self._infer_tp_size

        global_rank = self._rank
        infer_tp_rank_in_group = global_rank % self._infer_tp_size
        return param_list[infer_tp_rank_in_group]

    def allgather_tp_param(self, param, name):
        if self._tp_size <= self._infer_tp_size:
            return param
        tp_allgather_size = get_tp_allgather_world_size()
        tp_allgather_group = get_tp_allgather_group()
        infer_param = param

        if tp_allgather_size <= 1 or is_fake_tp_param(name, self.moe_tp_extend_ep):
            return infer_param

        if is_tensor_parallel_param(param):
            # allocate a new tensor with proper size
            infer_param = [torch.empty_like(param) for _ in range(tp_allgather_size)]
            torch.distributed.all_gather(infer_param, param, group=tp_allgather_group)
            if self.enable_validate:
                update_md5_by_rank(infer_param, param, self.origin_params_for_md5, self.infer_params_for_md5)
            part_len = len(infer_param) // self._infer_tp_size
            start = self._rank % self._infer_tp_size
            part_param = infer_param[part_len * start:part_len * (start + 1)]
            infer_param = self._default_tp_concat_fn(name, param, part_param)

        return infer_param
