#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../../..
PROFILER_DATA_PATH=$PROJECT_PATH/ci/profiler_data
rm -rf "$PROFILER_DATA_PATH" # 清理环境可能存在的 profiler 数据

python "$PROJECT_PATH"/cli/train_grpo.py --config-dir="$PROJECT_PATH"/tests/st/configs --config-name=test_grpo_trainer_qwen25_7b_integrated

python "$SCRIPT_DIR/../profiler/check_and_clean_profiler_output.py" --profiler-dir="$PROFILER_DATA_PATH"
