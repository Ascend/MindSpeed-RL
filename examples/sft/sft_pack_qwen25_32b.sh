#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME= "Your SOCKET IFNAME"
export TP_SOCKET_IFNAME= "Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME= "Your SOCKET IFNAME"
export HYDRA_FULL_ERROR=1

MASTER_ADDR="localhost"
MASTER_PORT="6060"
NNODES=2
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS cli/train_sft.py \
    --config-name sft_pack_qwen25_32b \
    | tee logs/sft_pack_qwen25_32b_rank${NODE_RANK}.log
