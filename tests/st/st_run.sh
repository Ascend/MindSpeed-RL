#!/bin/bash

# 查找所有名为 test_module_entry*.sh 的文件
# 使用 -print0 处理路径中的空格
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test_scripts=$(find "$SCRIPT_DIR" -name "test_module_entry_*.sh")

for script in $test_scripts; do
    echo "正在执行脚本: $script"

    # 执行脚本，并捕获错误
    if ! bash "$script"; then
        echo "脚本执行失败: $script"
        exit 1
    fi
done