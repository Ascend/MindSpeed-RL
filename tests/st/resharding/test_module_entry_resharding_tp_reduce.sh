#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH
export VLLM_DP_SIZE=1
export HCCL_BUFFSIZE=256
export VLLM_USE_V1=1
export VLLM_VERSION=0.9.0
export VLLM_ENABLE_GRAPH_MODE=0
export VLLM_ENABLE_MC2=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ENABLE_TOPK_OPTIMZE=1

GPUS_PER_NODE=8
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
PYTHON_ARGS="
    --model-path "/data/for_dt/weights/Qwen2.5-7B-mg" \
    --tokenizer-path "/data/for_dt/weights/Qwen2.5-7B" \
    --train-tp 4 \
    --train-pp 2 \
    --train-ep 1 \
    --infer-tp 2 \
    --infer-pp 1 \
    --infer-ep 1
"

echo "start test_resharding st: tp reduce"

torchrun $DISTRIBUTED_ARGS $SCRIPT_DIR/test_resharding.py $PYTHON_ARGS