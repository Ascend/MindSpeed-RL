#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True
export HCCL_HOST_SOCKET_PORT_RANGE="60000-60050"
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../../..
PROFILER_DATA_PATH=$PROJECT_PATH/ci/profiler_data
rm -rf "$PROFILER_DATA_PATH" # 清理环境可能存在的 profiler 数据

MSPROBE_DATA_PATH=$PROJECT_PATH/ci/msprobe_dump
rm -rf "$MSPROBE_DATA_PATH" # 清理环境可能存在的 msprobe 数据

python "$PROJECT_PATH"/cli/train_ppo.py --config-dir="$PROJECT_PATH"/tests/st/configs --config-name=test_ppo_trainer_qwen25_7b_integrated

python "$SCRIPT_DIR/../mindstudio/check_and_clean_mindstudio_output.py" --profiler-dir="$PROFILER_DATA_PATH" --msprobe-dir="$MSPROBE_DATA_PATH" --stage="ppo"

