# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
_ParallelState = None


def set_parallel_state(parallel_state):
    global _ParallelState
    _ParallelState = parallel_state


def get_parallel_state():
    global _ParallelState
    return _ParallelState


def get_pipeline_model_parallel_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_pipeline_model_parallel_group().rank_in_group
    else:
        return mpu.get_pipeline_model_parallel_rank()


def get_pipeline_model_parallel_src_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_pipeline_model_parallel_group().first_rank
    else:
        return mpu.get_pipeline_model_parallel_first_rank()


def get_pipeline_model_parallel_group(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_pipeline_model_parallel_group().device_group
    else:
        return mpu.get_pipeline_model_parallel_group()


def is_pipeline_last_stage(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_pipeline_model_parallel_group().is_last_rank
    else:
        return mpu.is_pipeline_last_stage()


def get_tensor_model_parallel_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tensor_model_parallel_rank()
    else:
        return mpu.get_tensor_model_parallel_rank()


def get_tensor_model_parallel_src_rank(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tensor_model_parallel_group().first_rank
    else:
        return mpu.get_tensor_model_parallel_src_rank()


def get_tensor_model_parallel_group(mpu, use_vllm=False):
    if use_vllm:
        from vllm.distributed import parallel_state as vpu
        return vpu.get_tensor_model_parallel_group().device_group
    else:
        return mpu.get_tensor_model_parallel_group()