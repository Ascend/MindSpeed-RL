#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH
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
PYTHON_ARGS_new="
    --model-path "/data/for_dt/weights/Qwen2.5-7B-tp2pp2" \
    --tokenizer-path "/data/for_dt/weights/Qwen2.5-7B" \
    --train-tp 2 \
    --train-pp 2 \
    --train-ep 1 \
    --infer-tp 4 \
    --infer-pp 1 \
    --infer-ep 1
"

echo "start test_resharding st"

torchrun $DISTRIBUTED_ARGS $SCRIPT_DIR/test_resharding.py $PYTHON_ARGS

torchrun $DISTRIBUTED_ARGS $SCRIPT_DIR/test_resharding.py $PYTHON_ARGS_new