#!/bin/bash
# /Ms/tests/st
BASE_DIR=$(dirname "$(readlink -f "$0")")
# /Ms/tests
EXEC_PY_DIR=$(dirname "$BASE_DIR")
# define dir
GENERATE_LOG_DIR=/data/verl_gate/run_logs
GENERATE_JSON_DIR=/data/verl_gate/run_jsons
SCRIPT_DIR="$BASE_DIR/train_engine"
BASELINE_DIR="$BASE_DIR/baseline_results"

# 清理日志和json
rm -rf $GENERATE_LOG_DIR/*
rm -rf $GENERATE_JSON_DIR/*

# 查找所有名为 *.sh 的文件
test_scripts=$(find "$SCRIPT_DIR" -name "*.sh")

for script in $test_scripts; do
    echo "正在执行脚本: $script"
    file_name_prefix=$(basename "${script%.*}")
    bash "$script" 2>&1 | tee "$GENERATE_LOG_DIR/$file_name_prefix.log"
    SCRIPT_EXITCODE=${PIPESTATUS[0]}
    if [ $SCRIPT_EXITCODE -ne 0 ]; then
        echo "Script has failed. Exit!"
        exit 1
    fi
    pytest -x $EXEC_PY_DIR/test_tools/test_ci_st.py \
        --baseline-json $BASELINE_DIR/$file_name_prefix.json \
        --generate-log $GENERATE_LOG_DIR/$file_name_prefix.log \
        --generate-json $GENERATE_JSON_DIR/$file_name_prefix.json
    PYTEST_EXITCODE=$?
    if [ $PYTEST_EXITCODE -ne 0 ]; then
        echo "$file_name_prefix compare to baseline has failed, check it!"
        exit 1
    else
        echo "Pretrain $file_name_prefix execution success."
    fi

    echo "任务执行完成": $script
    rm -rf ./ckpt
    ray stop --force
    ps -ef | grep torchrun | grep -v grep | awk '{print $2}' | xargs -r kill -9
    echo "计算资源清理完成"

done