#!/bin/bash
set -e   # 开启错误检测，当脚本中的任何命令返回非零退出状态时，立即退出脚本的执行。
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../../..
PROFILER_DATA_PATH=$PROJECT_PATH/ci/profiler_data
rm -rf "$PROFILER_DATA_PATH" # 清理环境可能存在的 profiler 数据

MSPROBE_DATA_PATH=$PROJECT_PATH/ci/msprobe_dump
rm -rf "$MSPROBE_DATA_PATH" # 清理环境可能存在的 msprobe 数据


# MM依赖mindspeed/megatron0.12，而LLM依赖0.80，因此这里先将跑LLM依赖的0.80备份。
mv "$PROJECT_PATH"/megatron "$PROJECT_PATH"/megatron_bk
mv "$PROJECT_PATH"/mindspeed "$PROJECT_PATH"/mindspeed_bk
# 构建MM及MM依赖的mindspeed/megatron
mkdir tmp
cd tmp
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron "$PROJECT_PATH"/
cd ..
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 6d63944cb2470a0bebc38dfb65299b91329b8d92
cp -r mindspeed "$PROJECT_PATH"/
cd ..
git clone https://gitee.com/ascend/MindSpeed-MM.git
cd ./MindSpeed-MM
# 注意这里MM固定了commitID，避免MM的PR合入打断RL的CI
git checkout cf8acd9301ed109218ca4fffe2e878790f6b0d7f
cp -r mindspeed_mm "$PROJECT_PATH"
# MMRL的入口脚本在MM仓，RL仓跑ST时需拷贝
cp -r posttrain_vlm_grpo.py "$PROJECT_PATH"/
# 入口脚本中的model_provider是从pretrain_vlm中导入而非mindspeed_mm命名空间，需拷贝
cp -r pretrain_vlm.py "$PROJECT_PATH"/
# MMRL的模型配置在MM仓，RL仓跑ST时需拷贝
cp examples/rl/model/qwen2.5vl_3b.json "$PROJECT_PATH"/
# MMRL的runtime_env在MM仓，RL仓跑ST时需拷贝
cp examples/rl/envs/runtime_env.yaml "$PROJECT_PATH"/
# MMRL依赖mindspeed0.12，当前commitID需进行两处修改，带mindspeed修复后可以去掉
cp examples/rl/code/build_tokenizer.py "$PROJECT_PATH"/mindspeed/features_manager/tokenizer/build_tokenizer.py
cp examples/rl/code/dot_product_attention.py "$PROJECT_PATH"/megatron/core/transformer/dot_product_attention.py
# MM运行需要的额外依赖
pip install -r examples/rl/requirements.txt
cd ..
cd ..
rm -rf tmp
cd ..
ls -l
python "$PROJECT_PATH"/posttrain_vlm_grpo.py --config-path="$PROJECT_PATH"/tests/st/configs --config-name=test_grpo_trainer_qwen25vl_3b_integrated


# 恢复LLM的mindspeed/megatron环境
rm -rf "$PROJECT_PATH"/megatron
rm -rf "$PROJECT_PATH"/mindspeed
rm -rf "$PROJECT_PATH"/mindspeed_mm
mv "$PROJECT_PATH"/megatron_bk "$PROJECT_PATH"/megatron
mv "$PROJECT_PATH"/mindspeed_bk "$PROJECT_PATH"/mindspeed
