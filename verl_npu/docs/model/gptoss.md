注：vllm-ascend 暂未支持该模型，为运行该模型，需要将 vllm 和 vllm-ascend 切换至以下指定的 commit_id， `import verl_npu` 将对该分支添加 patch。由于系统的换行符差异，自动 patch 的过程中，部分文件可能会 patch 失败导致程序中断，如果遇到，可以从 `./MindSpeed-RL/verl_npu/verl_npu/patch/vllm_ascend/941d54a2c/` 文件夹下移除对应 `*.patch` 文件，并根据文件里的内容手动修改对应代码文件。

# 环境依赖

## 
| MindSpeed RL版本 | PyTorch版本 | torch_npu版本 | CANN版本  | Python版本 |
| ---------------- | ------------ |-----------| ---------- | ---------- |
| master（主线）   | 2.7.1     | 2.7.1       | 8.5.0.B100 | Python3.10 |

## 1、安装 vllm 和 vllm-ascend
```bash
# vllm==0.11.1
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 2918c1b49c88c29783c86f78d2c4221cb9622379
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v .
# 此处的build安装的是torch以及torch-npu==2.8.0，需要更改成torch以及torch-npu==2.7.0，以下给出参考命令，vllm-ascend同
# pip uninstall torch
# pip uninstall torch-npu
# pip install torch==2.7.0
# pip uninstall torch-npu==2.7.0
cd ..

# vllm-ascend==0.11.0
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 941d54a2ce1ce387e4bf5d80003c098ff6d44841
pip install -r requirements.txt
pip install -e .
cd ..

# 源码安装transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 8365f70e925
pip install -e .
```

## 2、安装 MindSpeed 与 Megatron
```bash
# MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 1cdd0abd75e40936ad31721c092f57c695dd72c4
pip install -e .
cd ..

# Megatron
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.1
```

## 3、安装 verl
```bash
# verl==0.6.1
git clone https://github.com/volcengine/verl.git
cd verl
git checkout d62da4950573d7a4b7ef2362337952e7ab59e78d
pip install -e .
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
