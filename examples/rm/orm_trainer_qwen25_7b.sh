#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

Project_Path=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
export PYTHONPATH="$Project_Path:$PYTHONPATH"
echo $PYTHONPATH

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6021
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
export HYDRA_FULL_ERROR=1

torchrun $DISTRIBUTED_ARGS cli/train_orm.py \
    --config-name orm_trainer_qwen25_7b \
    | tee logs/train_orm_qwen25_7b_RL.log