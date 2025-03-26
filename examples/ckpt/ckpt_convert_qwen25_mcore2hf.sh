# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python cli/convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --load-dir ./ckpt/ \
    --save-dir ./ckpt/qwen25-7B  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen2.5_7b_hf/mg2hg/