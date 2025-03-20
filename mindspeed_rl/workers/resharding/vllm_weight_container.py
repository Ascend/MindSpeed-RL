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

import re
from functools import partial

import torch
import torch.distributed as dist
import numpy as np

from torch.distributed import new_group

from mindspeed_rl.workers.resharding.memory_buffer import build_model_weight_buffer

_PP_ALLGATHER_GROUP = None
_TP_ALLGATHER_GROUP = None
_TP_GROUP = None


class MegatronStyleVllmWeightContainer:

    def __init__(self, megatron_model, vllm_model, model_config, infer_tensor_parallel_size,
                 infer_pipeline_parallel_size,
                 infer_expert_parallel_size, num_layer_list, tp_split_expert=False, parallel_state=None) -> None:
        """ Megatron style vllm weight container.

        Arguments:
            megatron_model (nn.Module or nn.ModuleList): Megatron model used for training
            vllm_model (nn.Module or nn.ModuleList): VLLM model used for inference
            model_config (MegatronConfig): Model configuration
            infer_tensor_parallel_size (int): Inference tensor parallel size
            infer_pipeline_parallel_size (int): Inference pipeline parallel size
            infer_expert_parallel_size (int): Inference expert parallel size
            num_layer_list (str): a list of number of layers, seperated by comma; e.g., 4,4,4,4.
            tp_split_expert (bool): Controls whether expert model parameters are split across multiple GPUs.
            parallel_state (ModuleType): Megatron parallel state of the model.
        """

        self.vllm_model = vllm_model
        self.model_config = model_config
        self.megatron_model = megatron_model
        self.parallel_state = parallel_state
        self._num_hidden_layers = self.model_config.num_hidden_layers

        # pp configs
        self._pp_rank = self.parallel_state.get_pipeline_model_parallel_rank()
        self._pp_group = self.parallel_state.get_pipeline_model_parallel_group()
        self._pp_size = self.parallel_state.get_pipeline_model_parallel_world_size()
        self._num_layer_list = self._build_num_layer_list(num_layer_list)
        self._vpp_size = self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK if self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK else 1
        self._vpp_rank = self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE if self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE else 0

        # tp configs
        self._tp_size = self.parallel_state.get_tensor_model_parallel_world_size()
        self._tp_group = self.parallel_state.get_tensor_model_parallel_group()

        # ep configs
        self._ep_size = self.parallel_state.get_expert_model_parallel_world_size()
        self._ep_group = self.parallel_state.get_expert_model_parallel_group()

        # infer configs
        self._infer_tp_size = infer_tensor_parallel_size
        self._infer_pp_size = infer_pipeline_parallel_size
        self._infer_ep_size = infer_expert_parallel_size

        self._world_size = dist.get_world_size()

        # validate parallel configs
        self._validate_parallel_config()

        self._rank = dist.get_rank()
        self.tp_split_expert = tp_split_expert
        self._init_tensor_model_parallel_allgather_group()
        self._init_pipeline_model_parallel_allgather_group()
        self._init_tensor_model_parallel_split_group()
        self._init_weight_buffers()


    def _validate_parallel_config(self):
        if self._infer_pp_size != 1:
            raise ValueError("infer_pp_size != 1 not supported yet")
        if self._infer_ep_size != self._ep_size:
            raise ValueError("The training expert size should be equal to the inference expert size.")
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
        params = self._get_all_params()

        params = _build_infer_param_dict(params=params)
        return params

    def _build_num_layer_list(self, num_layer_list):
        if num_layer_list:
            return [int(num_layers) for num_layers in num_layer_list.split(',')]
        if self._num_hidden_layers % self._pp_size != 0:
            raise ValueError("num_layers % pp_size == 0, please specify num_layer_list")
        return [self._num_hidden_layers // self._pp_size for _ in range(self._pp_size)]

    def _get_weight_names_per_pp(self):
        end_layer = self.model_config.num_hidden_layers - 1

        def get_weight_names_in_range(layer_range, names: list, layer_name='layers') -> list:
            """
            Extract weights in a given range and also include the weights before and after the range as needed.
            """
            start, end = layer_range
            last_layer_index = end_layer
            names_in_range = []

            # add names before decoder layers
            if start == 0:
                for name in names:
                    if layer_name not in name:
                        names_in_range.append(name)
                    else:
                        break

            for name in names:
                # Extract layer number from weight
                match = re.match(r'.*\.layers\.(\d+)', name)
                if match:
                    layer_num = int(match.group(1))
                    if start <= layer_num <= end:
                        names_in_range.append(name)

            # add names after decode layers
            if end == last_layer_index:
                for name in reversed(names):
                    if layer_name not in name:
                        names_in_range.append(name)
                    else:
                        break
            return names_in_range

        pp_size = self._pp_size
        vllm_names = list(dict(self.vllm_model.named_parameters()).keys())
        num_layers = self.model_config.num_hidden_layers
        pp_layers_range = []
        start_layer = 0
        for layers_in_pp_rank in self._num_layer_list:
            pp_layers_range.append((start_layer, start_layer + layers_in_pp_rank - 1))
            start_layer += layers_in_pp_rank
        weight_names_per_pp = [get_weight_names_in_range(layer_range, vllm_names) for layer_range in pp_layers_range]
        return weight_names_per_pp

    def _unwrap_megatron_model(self, model):
        """
        Remove consecutive 'module.' prefixes from the model based on the state_dict's first key.
        This method only removes 'module.' from the beginning of the key and ignores other occurrences.
        """
        model = model[0]
        first_key = list(dict(model.named_parameters()).keys())[0]
        while first_key.startswith("module."):
            model = model.module
            first_key = first_key[len("module."):]  # 更新键，去掉一个module.
        return model

    def _init_weight_buffers(self):
        """
        Build buffers from vllm state dict. Totally build train pp_size buffers, each buffer corresponds to a pack of megatron weight.
        Return a list of buffers, and a reference dict megatron_param_name->buffer.
        """
        self.params_mapping = [
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
        self.weight_names_per_pp = self._get_weight_names_per_pp()
        self.weight_buffers = build_model_weight_buffer(self.vllm_model, self.weight_names_per_pp)

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
            return infer_param

        def _global2local_layer(name, num_layer_list):
            """
            Transform the model name in each model_chunk in global space to local space
            """
            layer_name = 'layers'

            if layer_name in name:  # belong to an intermediate layer
                split_name = name.split('.')
                # find the num next to split_name
                for i, name in enumerate(split_name):
                    if name == layer_name:
                        break
                layer_num_idx = i + 1
                # check the name
                if len(split_name) < layer_num_idx + 1 or not split_name[layer_num_idx].isdigit():
                    raise ValueError(f'split_name = {split_name}')
                # increment layer_num_idx by layer_offset
                global_idx = int(split_name[layer_num_idx])
                for layers_in_pp in num_layer_list:
                    global_idx -= layers_in_pp
                    if global_idx < 0:
                        local_index = global_idx + layers_in_pp
                        break
                split_name[layer_num_idx] = str(local_index)
                name = '.'.join(split_name)  # weight name in inference_tp_model
            return name

        pp_rank = self._pp_rank
        weight_buffer = self.weight_buffers[pp_rank]
        true_megatron_model = self._unwrap_megatron_model(self.megatron_model)
        normal_layer_func = partial(_global2local_layer, num_layer_list=self._num_layer_list)
        name_pairs = sorted(list(set([(name, _replace_name_v2m(normal_layer_func(name), self.params_mapping))
                                      for name in weight_buffer.weight_names])))
        for hf_name, megatron_name in name_pairs:
            if megatron_name.endswith("linear_fc1.weight"):
                fc2_name = megatron_name.replace("linear_fc1", "linear_fc2")
                megatron_param_fc1 = dict(true_megatron_model.named_parameters())[megatron_name]
                megatron_param_fc2 = dict(true_megatron_model.named_parameters())[fc2_name]
                if megatron_param_fc1.shape[0] * megatron_param_fc1.shape[1] != megatron_param_fc2.shape[0] * megatron_param_fc2.shape[1] * 2:
                    raise ValueError("Only implemented for Llama model which linear_fc1 contains gate and up params.")
        for hf_name, megatron_name in name_pairs:
            megatron_param = dict(true_megatron_model.named_parameters())[megatron_name]
            param = _transfer_from_megatron_division(megatron_param, megatron_name)
            weight_buffer[hf_name].copy_(param)

    def _update_weight_buffers_inter_pp(self):
        """
        Update weight buffers by gathering weights from other pp stage.

        """
        for cur_pp_rank in range(self._pp_size):
            global_src = dist.get_global_rank(group=self._pp_group, group_rank=cur_pp_rank)
            for memory_buffer in self.weight_buffers[cur_pp_rank].memory_buffers.values():
                dist.broadcast(tensor=memory_buffer.data, src=global_src, group=self._pp_group, async_op=False)

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
        tp_allgather_size = self._tp_size // self._infer_tp_size
        global _TP_ALLGATHER_GROUP
        if _TP_ALLGATHER_GROUP is not None:
            raise RuntimeError("Group for allgather tensor model parallel weight is already initialized")
        num_groups = self._world_size // tp_allgather_size
        for i in range(num_groups):
            ranks = range(i * tp_allgather_size, (i + 1) * tp_allgather_size)
            group = new_group(ranks=ranks)
            if self._rank in ranks:
                _TP_ALLGATHER_GROUP = group

    def _init_pipeline_model_parallel_allgather_group(self):
        if self._pp_size < self._infer_pp_size:
            raise NotImplementedError("Not implemented for infer_pp > train_pp")
        if self._pp_size % self._infer_pp_size != 0:
            raise ValueError(
                "Pipeline model parallel size must be a multiple of inference pipeline model parallel size")
        pp_allgather_size = self._pp_size // self._infer_pp_size
        global _PP_ALLGATHER_GROUP
        if _PP_ALLGATHER_GROUP is not None:
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
                    _PP_ALLGATHER_GROUP = cur_group

    def _init_tensor_model_parallel_split_group(self):
        if self._tp_size >= self._infer_tp_size:
            return
        if self._infer_tp_size % self._tp_size != 0:
            raise ValueError("self._infer_tp_size must be a multiple of self._tp_size")
        global _TP_GROUP
        if _TP_GROUP is not None:
            raise RuntimeError("Group for tensor model parallel weight is already initialized")
        if self._infer_tp_size > self._tp_size:
            _TP_GROUP = self.parallel_state.get_tensor_model_parallel_group()
            
    def _default_tp_concat_fn(self, name, param, infer_params):
        """
        name: name of the parameter
        param: training_utils parameters
        infer_params (List[torch.Tensor]): a list of parameters all-gathered from micro_dp_group
        definition so that it is model-agnostic. If the model doesn't implement this function,
        we can throw an error to force user disable TP HybridEngine.
        """

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
        if self._infer_tp_size <= self._tp_size:
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

        if tp_allgather_size <= 1:
            return infer_param

        if is_tensor_parallel_param(param):
            # allocate a new tensor with proper size
            infer_param = [torch.empty_like(param) for _ in range(tp_allgather_size)]
            torch.distributed.all_gather(infer_param, param, group=tp_allgather_group)
            infer_param = self._default_tp_concat_fn(name, param, infer_param)

        return infer_param


def _replace_name_v2m(vllm_name, name_mapping):
    """
    Transfer state dict names from vllm to megatron.
    This function works in the opposite direction of _replace_name.
    """
    for m_name, v_name in name_mapping:
        if v_name not in vllm_name:
            continue
        if "layers" in vllm_name:  # deal with decoder layers
            vllm_name = vllm_name.replace("model", "decoder")
            vllm_name_list = vllm_name.split(".")
            if "layer_norm_weight" in vllm_name_list or "layer_norm_bias" in vllm_name_list:
                param_name_list = vllm_name_list[:3]
                param_name_list.append(m_name)
                param_name = ".".join(param_name_list)
            else:
                param_name_list = vllm_name_list[:3]
                weight_or_bias = vllm_name_list[-1]
                param_name_list.append(m_name)
                param_name_list.append(weight_or_bias)
                param_name = ".".join(param_name_list)
            return param_name
        else:
            param_name = vllm_name.replace(v_name, m_name)
            return param_name


def _build_infer_param_dict(params):
    """
    params: List[List[Dict[str, param]]]
        params contains a list of pp, with a list of vpp named_parameters in each vpp chunk.
    output: Dict[str, param]

    """
    infer_param = {}
    for param_list in params:
        for param_dict in param_list:
            for name, param in param_dict.items():
                infer_param[name] = param

    return infer_param


def get_tp_group():
    return _TP_GROUP


def get_tp_world_size():
    return torch.distributed.get_world_size(group=get_tp_group())


def get_tp_rank():
    return torch.distributed.get_rank(group=get_tp_group())


def get_tp_allgather_group():
    if _TP_ALLGATHER_GROUP is None:
        raise ValueError("TP AllGather Group is not initialized")
    return _TP_ALLGATHER_GROUP


def get_tp_allgather_world_size():
    return torch.distributed.get_world_size(group=get_tp_allgather_group())


def get_tp_allgather_rank():
    return torch.distributed.get_rank(group=get_tp_allgather_group())


def get_pp_allgather_group():
    if _PP_ALLGATHER_GROUP is None:
        raise ValueError("PP AllGather Group is not initialized")
    return _PP_ALLGATHER_GROUP


def get_pp_allgather_world_size():
    return torch.distributed.get_world_size(group=get_pp_allgather_group())


def get_pp_allgather_rank():
    return torch.distributed.get_rank(group=get_pp_allgather_group())


def is_tensor_parallel_param(param):
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel)


def get_tensor_parallel_partition_dim(param):
    if not is_tensor_parallel_param(param):
        raise TypeError("Parameter is not tensor parallel")
    return param.partition_dim
