#!/bin/bash

# 使用指南：
# 1. 执行行覆盖率测试：./run_coverage.sh
# 2. 执行分支覆盖率测试：./run_coverage.sh --branch
# 3. 执行单个文件测试：./run_coverage.sh --file <文件路径>
# 生成的覆盖率报告将保存在当前目录下的 htmlcov 文件夹中，并且会生成 coverage.xml 文件。

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 默认配置
ENABLE_BRANCH="False"
SINGLE_FILE=""
START_TIME=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --branch)
            ENABLE_BRANCH="True"
            shift
            ;;
        --file)
            SINGLE_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --branch       启用分支覆盖率跟踪"
            echo "  --file <路径>  执行单个文件的测试"
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

# 开始计时
start_timer() {
    START_TIME=$(date +%s)
    print_color "${BLUE}" "开始时间: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S')"
}

# 结束计时并显示运行时间
end_timer() {
    local end_time=$(date +%s)
    local elapsed_time=$((end_time - START_TIME))
    local hours=$((elapsed_time / 3600))
    local minutes=$(((elapsed_time % 3600) / 60))
    local seconds=$((elapsed_time % 60))
    
    print_color "${BLUE}" "结束时间: $(date -d @$end_time '+%Y-%m-%d %H:%M:%S')"
    print_color "${GREEN}" "总运行时间: ${hours}小时 ${minutes}分钟 ${seconds}秒"
}

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
omit = 
    $SOURCE_DIR/workers/eplb/*
    $SOURCE_DIR/models/rollout/*
    $SOURCE_DIR/trainer/*_trainer_hybrid.py


[report]
show_missing = True
skip_covered = False
EOF

# 记录原始文件内容，以便恢复
backup_files() {
    for file in "$@"; do
        if [ -f "$file" ]; then
            cp "$file" "${file}.bak"
            print_color "${CYAN}" "已备份文件: $file"
        else
            print_color "${YELLOW}" "警告: 文件 '$file' 不存在，跳过备份"
        fi
    done
}

# 恢复原始文件
restore_files() {
    for file in "$@"; do
        if [ -f "${file}.bak" ]; then
            mv "${file}.bak" "$file"
            print_color "${CYAN}" "已恢复文件: $file"
        else
            print_color "${YELLOW}" "警告: 备份文件 '${file}.bak' 不存在，无法恢复"
        fi
    done
}

# 添加覆盖率追踪代码
add_coverage() {
    for file in "$@"; do
        if [ ! -f "$file" ]; then
            print_color "${YELLOW}" "警告: 文件 '$file' 不存在，跳过处理"
            continue
        fi
        
        # 检查文件是否已经添加了覆盖率代码
        if grep -q "import coverage" "$file"; then
            print_color "${YELLOW}" "警告: 文件 '$file' 已经添加了覆盖率代码，跳过处理"
            continue
        fi
        
        # 在文件开头添加指定代码
        sed -i "1a\import random" "$file"
        sed -i "2a\import time" "$file"
        sed -i "3a\import hashlib" "$file"
        sed -i "4a\import base64" "$file"
        sed -i "5a\import os" "$file"
        sed -i "6a\import sys" "$file"
        sed -i "7a\if int(os.environ.get('RANK', '0')) == 0:" "$file"
        sed -i "8a\    import coverage" "$file"
        sed -i "9a\    current_time = str(time.time_ns())" "$file"
        sed -i "10a\    sha256_hash = hashlib.sha256(current_time.encode()).hexdigest()" "$file"
        sed -i "11a\    result = base64.b64encode(sha256_hash.encode()).decode()[:8]" "$file"
        sed -i "12a\    cov = coverage.Coverage(data_suffix=result)" "$file"
        sed -i "13a\    cov.start()" "$file"
        
        # 在文件末尾添加指定代码
        if grep -q "    main()" "$file"; then
            sed -i "/    main()/a\    if int(os.environ.get('RANK', '0')) == 0 and 'coverage' in sys.modules:" "$file"
            sed -i "/    if int(os.environ.get('RANK', '0')) == 0 and 'coverage' in sys.modules:/a\        cov.stop()" "$file"
            sed -i "/        cov.stop()/a\        cov.save()" "$file"
        else
            print_color "${YELLOW}" "警告: 在文件 '$file' 中未找到 'main()' 函数，覆盖率数据可能无法正确保存"
        fi

        print_color "${CYAN}" "已为文件添加覆盖率追踪: $file"
    done
}

# 移除覆盖率追踪代码
remove_coverage() {
    restore_files "$@"
}

# 执行单元测试覆盖率
run_unit_tests_coverage() {
    print_color "${GREEN}" "执行单元测试覆盖率..."
    local coverage_files=()
    
    # 如果指定了单个文件，只运行该文件
    if [ -n "$SINGLE_FILE" ]; then
        if [ -f "$SINGLE_FILE" ]; then
            print_color "${BLUE}" "测试单个文件: $SINGLE_FILE"
            coverage run -p --source="$SOURCE_DIR" -m pytest --log-cli-level=INFO "$SINGLE_FILE" || {
                print_color "${RED}" "错误: 文件 '$SINGLE_FILE' 测试失败"
            }
        else
            print_color "${RED}" "错误: 文件 '$SINGLE_FILE' 不存在"
            exit 1
        fi
    else
        # 收集所有单元测试文件
        all_python_files=()
        while IFS= read -r -d '' file; do
            all_python_files+=("$file")
        done < <(find "$UT_DIR" -type f -name "*.py" -print0)

        # 如果有Python文件，则执行覆盖率测试
        if [ ${#all_python_files[@]} -gt 0 ]; then
            # 将数组转换为空格分隔的字符串 (用于显示)
            all_files_str="${all_python_files[*]}"
            print_color "${BLUE}" "找到 ${#all_python_files[@]} 个测试文件"
            print_color "${CYAN}" "测试文件列表: $all_files_str"
            
            # 一次性运行所有测试文件
            coverage run -p --source="$SOURCE_DIR" -m pytest --log-cli-level=INFO "${all_python_files[@]}" || \
            print_color "${YELLOW}" "警告: 部分测试失败"
        else
            print_color "${YELLOW}" "未找到任何Python测试文件"
        fi
    fi
    
    print_color "${GREEN}" "单元测试覆盖率执行完成"
}

# 执行系统测试覆盖率
run_system_tests_coverage() {
    # 如果指定了单个文件，跳过系统测试
    if [ -n "$SINGLE_FILE" ]; then
        print_color "${YELLOW}" "跳过系统测试（单个文件模式）"
        return 0
    fi
    
    print_color "${GREEN}" "执行系统测试覆盖率..."
    
    # 需要添加覆盖率的文件列表
    local file_list=(
        "$BASE_DIR/tests/st/datasets/test_preprocess_data.py"
        "$BASE_DIR/cli/train_ppo.py"
        "$BASE_DIR/cli/train_dpo.py"
        "$BASE_DIR/cli/train_grpo.py"
        "$BASE_DIR/cli/train_dapo.py"
        "$BASE_DIR/tests/st/mindstudio/check_and_clean_mindstudio_output.py"
        "$BASE_DIR/tests/st/resharding/test_resharding.py"
    )
    
    # 备份文件
    backup_files "${file_list[@]}"
    
    # 添加覆盖率代码
    add_coverage "${file_list[@]}"
    
    # 执行系统测试
    print_color "${BLUE}" "执行系统测试脚本..."
    bash "$ST_DIR/st_run.sh" || {
        print_color "${YELLOW}" "系统测试脚本执行失败，但继续生成覆盖率报告"
    }
    
    # 移除覆盖率代码
    remove_coverage "${file_list[@]}"
    
    print_color "${GREEN}" "系统测试覆盖率执行完成"
}

# 生成覆盖率报告
generate_coverage_report() {
    print_color "${GREEN}" "生成覆盖率报告..."
    
    # 合并覆盖率数据
    coverage combine || {
        print_color "${YELLOW}" "警告: 合并覆盖率数据失败"
        return 1
    }
    
    # 生成HTML报告
    coverage html || {
        print_color "${YELLOW}" "警告: 生成HTML覆盖率报告失败"
        return 1
    }
    
    # 生成XML报告
    coverage xml || {
        print_color "${YELLOW}" "警告: 生成XML覆盖率报告失败"
        return 1
    }
    
    # 显示终端报告
    coverage report
    
    print_color "${GREEN}" "覆盖率报告生成完成，HTML报告位于 htmlcov/index.html"
}

# 清理覆盖率文件
cleanup() {
    print_color "${BLUE}" "清理覆盖率文件..."
    rm -f .coverage
    rm -f .coverage.*
    rm -f .coveragerc
    
    # 检查并删除备份文件
    find . -name "*.py.bak" -delete
    
    print_color "${BLUE}" "清理完成"
}

# 主函数
main() {
    start_timer
    print_color "${PURPLE}" "开始执行覆盖率测试..."
    print_color "${CYAN}" "分支覆盖率跟踪: $ENABLE_BRANCH"
    
    if [ -n "$SINGLE_FILE" ]; then
        print_color "${CYAN}" "单文件模式: $SINGLE_FILE"
    fi
    
    # 执行单元测试覆盖率
    run_unit_tests_coverage
    
    # 执行系统测试覆盖率
    run_system_tests_coverage
    
    # 生成覆盖率报告
    generate_coverage_report
    
    print_color "${PURPLE}" "覆盖率测试完成"
    end_timer
}

# 确保清理工作总是执行
trap cleanup EXIT

# 执行主函数
main