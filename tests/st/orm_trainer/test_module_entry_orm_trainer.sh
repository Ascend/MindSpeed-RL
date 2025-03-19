export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6004
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

torchrun $DISTRIBUTED_ARGS $SCRIPT_DIR/test_orm_trainer.py \
    | tee orm_trainer_qwen25_7b.log
