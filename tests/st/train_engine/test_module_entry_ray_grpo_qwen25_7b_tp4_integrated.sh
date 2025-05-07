#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../../..

python "$PROJECT_PATH"/cli/train_grpo.py --config-dir="$PROJECT_PATH"/tests/st/configs --config-name=test_grpo_trainer_qwen25_7b_integrated