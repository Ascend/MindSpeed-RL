#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
# 获取脚本的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH

GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6555
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
echo "start test_vllm_engine st"

torchrun $DISTRIBUTED_ARGS  $SCRIPT_DIR/test_vllm_engine.py --distribute-backend nccl
torchrun $DISTRIBUTED_ARGS  $SCRIPT_DIR/test_vllm_engine_multistep_decode.py --distribute-backend nccl