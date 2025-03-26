#!/bin/bash

export GLOO_SOCKET_IFNAME="Your SOCKET IFNAME"
export TP_SOCKET_IFNAME="Your SOCKET IFNAME"
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR="host ip"
MASTER_PORT=6001
NNODES=4
NODE_RANK="node rank"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

INFER_ARGS="
    --tokenizer-name-or-path 'your huggingface config path' \
    --load-format megatron \
    --load 'megatron weight path' \
    --tensor-parallel-size 32 \
    --task chat \
"

torchrun $DISTRIBUTED_ARGS cli/infer_vllm.py \
    $INFER_ARGS \
    --query "Write an essay about the importance of higher education." \
    --distributed-backend nccl
