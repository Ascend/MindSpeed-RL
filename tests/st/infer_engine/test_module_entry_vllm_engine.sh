#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$PWD/../..:$PYTHONPATH
GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6555
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS ./test_vllm_engine.py --distribute-backend nccl