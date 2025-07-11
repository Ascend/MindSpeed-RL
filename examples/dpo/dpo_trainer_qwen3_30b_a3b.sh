#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CLOSE_MATMUL_K_SHIFT=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6005
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS cli/train_dpo.py \
    --config-name dpo_qwen3_30b_a3b_A2 \
    | tee logs/RL_dpo_qwen3_30b_a3b_rank${NODE_RANK}.log