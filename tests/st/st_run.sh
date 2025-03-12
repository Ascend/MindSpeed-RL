#!/bin/bash

echo "st start"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test_scripts=$(find "$SCRIPT_DIR" -name "test_module_entry*.sh")

# 遍历找到的脚本并执行
for script in $test_scripts; do
    echo "正在执行脚本: $script"
    # 执行脚本
    sh "$script"

done