#!/bin/bash

# 使用指南：
# 1. 执行行覆盖率测试：./run_coverage.sh
# 2. 执行分支覆盖率测试：./run_coverage.sh --branch
# 生成的覆盖率报告将保存在当前目录下的 htmlcov 文件夹中，并且会生成 coverage.xml 文件。

set -e

# 默认配置
ENABLE_BRANCH="False"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --branch)
            ENABLE_BRANCH="True"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --branch       启用分支覆盖率跟踪"
            echo "  -h, --help     显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 '$0 --help' 查看可用选项"
            exit 1
            ;;
    esac
done

# 定义基础目录
BASE_DIR=$(dirname "$(readlink -f "$0")")/..
SOURCE_DIR="$BASE_DIR/mindspeed_rl"
UT_DIR="$BASE_DIR/tests/ut"
ST_DIR="$BASE_DIR/tests/st"

# 移除现有的覆盖率文件
rm -f .coverage
rm -f .coverage.*
rm -rf htmlcov

# 创建覆盖率配置文件
cat > ".coveragerc" << EOF
[run]
branch = $ENABLE_BRANCH
parallel = True
source = $SOURCE_DIR

[report]
show_missing = True
skip_covered = False
EOF

# 记录原始文件内容，以便恢复
backup_files() {
    for file in "$@"; do
        if [ -f "$file" ]; then
            cp "$file" "${file}.bak"
        else
            echo "警告: 文件 '$file' 不存在，跳过备份"
        fi
    done
}

# 恢复原始文件
restore_files() {
    for file in "$@"; do
        if [ -f "${file}.bak" ]; then
            mv "${file}.bak" "$file"
        else
            echo "警告: 备份文件 '${file}.bak' 不存在，无法恢复"
        fi
    done
}

# 添加覆盖率追踪代码
add_coverage() {
    for file in "$@"; do
        if [ ! -f "$file" ]; then
            echo "警告: 文件 '$file' 不存在，跳过处理"
            continue
        fi
        
        # 检查文件是否已经添加了覆盖率代码
        if grep -q "import coverage" "$file"; then
            echo "警告: 文件 '$file' 已经添加了覆盖率代码，跳过处理"
            continue
        fi
        
        # 生成唯一的覆盖率数据后缀
        SUFFIX=$(date +%s%N | sha256sum | base64 | head -c 8)
        
        # 添加覆盖率代码
        sed -i "1a\import random" "$file"
        sed -i "2a\import time" "$file"
        sed -i "3a\import coverage" "$file"
        sed -i "4a\cov = coverage.Coverage(data_suffix=\"$SUFFIX\")" "$file"
        sed -i "5a\cov.start()" "$file"
        
        if grep -q "    main()" "$file"; then
            sed -i "/    main()/a\    cov.stop()" "$file"
            sed -i "/    cov.stop()/a\    cov.save()" "$file"
        else
            echo "警告: 在文件 '$file' 中未找到 'main()' 函数，覆盖率数据可能无法正确保存"
        fi
    done
}

# 移除覆盖率追踪代码
remove_coverage() {
    restore_files "$@"
}

# 执行单元测试覆盖率
run_unit_tests_coverage() {
    echo "执行单元测试覆盖率..."
    local coverage_files=()
    
    # 收集所有单元测试目录
    find "$UT_DIR" -mindepth 1 -maxdepth 2 -type d | while read -r dir; do
        if [ -d "$dir" ]; then
            # 收集当前目录下的所有Python文件
            local python_files=()
            while IFS= read -r -d '' file; do
                python_files+=("$file")
            done < <(find "$dir" -type f -name "*.py" -print0)
            
            # 如果有Python文件，则执行覆盖率测试
            if [ ${#python_files[@]} -gt 0 ]; then
                echo "测试目录: $dir"
                for file in "${python_files[@]}"; do
                    echo "  测试文件: $file"
                    coverage run -p --source="$SOURCE_DIR" "$file" || echo "  警告: 文件 '$file' 测试失败"
                done
            fi
        fi
    done
    
    echo "单元测试覆盖率执行完成"
}

# 执行系统测试覆盖率
run_system_tests_coverage() {
    echo "执行系统测试覆盖率..."
    
    # 需要添加覆盖率的文件列表
    local file_list=(
        "$BASE_DIR/cli/preprocess_data.py"
        "$BASE_DIR/cli/convert_ckpt.py"
        "$BASE_DIR/cli/train_grpo.py"
        "$BASE_DIR/tests/st/mindstudio/check_and_clean_mindstudio_output.py"
        "$BASE_DIR/tests/st/resharding/test_resharding.py"
    )
    
    # 备份文件
    backup_files "${file_list[@]}"
    
    # 添加覆盖率代码
    add_coverage "${file_list[@]}"
    
    # 执行系统测试
    echo "执行系统测试脚本..."
    bash "$ST_DIR/st_run.sh" || {
        echo "系统测试脚本执行失败，但继续生成覆盖率报告"
    }
    
    # 移除覆盖率代码
    remove_coverage "${file_list[@]}"
    
    echo "系统测试覆盖率执行完成"
}

# 生成覆盖率报告
generate_coverage_report() {
    echo "生成覆盖率报告..."
    
    # 合并覆盖率数据
    coverage combine || {
        echo "警告: 合并覆盖率数据失败"
        return 1
    }
    
    # 生成HTML报告
    coverage html || {
        echo "警告: 生成HTML覆盖率报告失败"
        return 1
    }
    
    # 生成XML报告
    coverage xml || {
        echo "警告: 生成XML覆盖率报告失败"
        return 1
    }
    
    # 显示终端报告
    coverage report
    
    echo "覆盖率报告生成完成，HTML报告位于 htmlcov/index.html"
}

# 清理覆盖率文件
cleanup() {
    echo "清理覆盖率文件..."
    rm -f .coverage
    rm -f .coverage.*
    rm -f .coveragerc
    
    # 检查并删除备份文件
    find . -name "*.py.bak" -delete
    
    echo "清理完成"
}

# 主函数
main() {
    echo "开始执行覆盖率测试..."
    echo "分支覆盖率跟踪: $ENABLE_BRANCH"
    
    # 执行单元测试覆盖率
    run_unit_tests_coverage
    
    # 执行系统测试覆盖率
    run_system_tests_coverage
    
    # 生成覆盖率报告
    generate_coverage_report
    
    echo "覆盖率测试完成"
}

# 确保清理工作总是执行
trap cleanup EXIT

# 执行主函数
main
