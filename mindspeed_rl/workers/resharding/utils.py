import hashlib
import logging

import torch

_PP_ALLGATHER_GROUP = None
_TP_ALLGATHER_GROUP = None
_TP_GROUP = None


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


def get_tp_allgather_group():
    if _TP_ALLGATHER_GROUP is None:
        raise ValueError("TP AllGather Group is not initialized")
    return _TP_ALLGATHER_GROUP


def get_tp_allgather_world_size():
    return torch.distributed.get_world_size(group=get_tp_allgather_group())


def get_pp_allgather_group():
    if _PP_ALLGATHER_GROUP is None:
        raise ValueError("PP AllGather Group is not initialized")
    return _PP_ALLGATHER_GROUP


def is_tensor_parallel_param(param):
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel)


def get_tensor_parallel_partition_dim(param):
    if not is_tensor_parallel_param(param):
        raise TypeError("Parameter is not tensor parallel")
    return param.partition_dim


def is_fake_tp_param(name, moe_tp_extended_ep):
    return 'mlp.experts.weight' in name and moe_tp_extended_ep
