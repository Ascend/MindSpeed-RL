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


def tp_md5_validate(infer_params_for_md5, origin_params_for_md5, log_prefix):
    md5_tensor = bytes_to_tensor(origin_params_for_md5)
    origin_params_md5_allgather_tensor = []
    for _ in range(get_tp_allgather_world_size()):
        origin_params_md5_allgather_tensor.append(torch.empty_like(md5_tensor))
    torch.distributed.all_gather(origin_params_md5_allgather_tensor, md5_tensor, group=get_tp_allgather_group())
    for index, params in enumerate(infer_params_for_md5):
        recv_md5_tensor = bytes_to_tensor(params)
        validate_md5(origin_params_md5_allgather_tensor[index], recv_md5_tensor, log_prefix)


def update_md5_by_rank(infer_param, param, origin_params_for_md5, infer_params_for_md5):
    # compute current param' md5 value at current rank
    param_bytes = param.data.to(torch.float32).cpu().numpy().tobytes()
    origin_params_for_md5.update(param_bytes)
    # Calculate the md5 values of all received params in the TP group, separated by rank
    for index, recv_param in enumerate(infer_param):
        recv_param_bytes = recv_param.data.to(torch.float32).cpu().numpy().tobytes()
        infer_params_for_md5[index].update(recv_param_bytes)


def bytes_to_tensor(bytes_data):
    md5_tensor = torch.tensor([int(h, 16) for h in bytes_data.hexdigest()], dtype=torch.int64,
                              device=torch.cuda.current_device())
    return md5_tensor


def compute_md5(model):
    hash_value = hashlib.md5()
    for memory_buffer in model.memory_buffers.values():
        param_bytes = memory_buffer.data.detach().to(torch.float32).cpu().numpy().tobytes()
        hash_value.update(param_bytes)
    md5_tensor = bytes_to_tensor(hash_value)
    return md5_tensor


def validate_md5(md5_tensor_src, md5_tensor, log_prefix):
    if torch.equal(md5_tensor_src, md5_tensor):
        logging.info(f"{log_prefix} md5 validate Hash: The weights of the two models match.")
    else:
        logging.info(f"{log_prefix} md5 validate Hash: The weights of the two models do not match.")


def is_fake_tp_param(name, moe_tp_extended_ep):
    return 'mlp.experts.weight' in name and moe_tp_extended_ep
