export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../../..

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6004
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS "$PROJECT_PATH"/cli/train_orm.py --config-dir="$PROJECT_PATH"/tests/st/configs \
--config-name=test_orm_trainer_qwen25_7b | tee orm_trainer_qwen25_7b.log
