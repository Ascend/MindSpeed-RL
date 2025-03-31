#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME= "Your SOCKET IFNAME"
export TP_SOCKET_IFNAME= "Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME= "Your SOCKET IFNAME"
export HYDRA_FULL_ERROR=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6005
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

torchrun $DISTRIBUTED_ARGS cli/train_sft.py \
    --config-name sft_qwen25_7b \
    | tee logs/sft_qwen25_7b_rank${NODE_RANK}.log
