GPT-oss-20b 是由OpenAI于2025年8月5日发布的开放权重AI模型，总参数210亿，每token激活36亿参数，专为低延迟、本地化场景设计，可在16GB内存的边缘设备运行。该模型采用混合专家（MoE）架构，基于Transformer框架，结合密集注意力和局部带状稀疏注意力机制，支持128,000token的上下文长度。

# 环境依赖

## 
| MindSpeed RL版本 | PyTorch版本 | torch_npu版本 | CANN版本  | Python版本 |
| ---------------- | ------------ |-----------| ---------- | ---------- |
| master（主线）   | 2.7.1     | 2.7.1       | 8.5.0 | Python3.10 |

## 1、安装 vllm 和 vllm-ascend
```bash
# vllm==0.16.0
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 89a77b10846fd96273cce78d86d2556ea582d26e
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e.
# 此处的build安装的是torch以及torch-npu==2.8.0，需要更改成torch以及torch-npu==2.7.1，以下给出参考命令，vllm-ascend同
# pip uninstall torch
# pip uninstall torch-npu
# pip install torch==2.7.1
# pip uninstall torch-npu==2.7.1
cd ..

# vllm-ascend==0.15.0
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 3cc8bf15da7c182f05fdadb3d2cb071812d7ac67
pip install -r requirements.txt
pip install -v -e .
cd ..

# 源码安装transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 8365f70e925
pip install -v -e .
```

## 2、安装 MindSpeed 与 Megatron
```bash
# MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 1cdd0abd75e40936ad31721c092f57c695dd72c4
pip install -v -e .
cd ..

# Megatron
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.1
```

## 3、安装 verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -v -e .
cd ..
```

## 4、安装插件
```bash
# 请确保 vllm 已正确安装并且之后不会做覆盖
git clone https://gitcode.com/Ascend/MindSpeed-RL.git
cd MindSpeed-RL/verl_npu
pip install -v -e .
cd ../..
```


# 启动训练

安装成功后，将参考配置脚本 `MindSpeed-RL\tests\verl_examples\configs\test_grpo_gptoss_20b_fsdp_A2.sh` 放入 verl 目录下，执行：

```bash
bash test_grpo_gptoss_20b_fsdp_A2.sh
```
