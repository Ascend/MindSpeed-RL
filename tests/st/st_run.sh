#!/bin/bash


# 查找 ST 目录下所有名为 test_module_entry*.sh 的文件
test_scripts=$(find ./ -name "test_module_entry*.sh")

# 遍历找到的脚本并执行
for script in $test_scripts; do
    echo "正在执行脚本: $script"    
    # 执行脚本
    sh $script
done