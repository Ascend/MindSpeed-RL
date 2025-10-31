



# 环境依赖

## 
| MindSpeed RL版本 | PyTorch版本 | torch_npu版本 | CANN版本  | Python版本 |
| ---------------- | ------------ |-----------| ---------- | ---------- |
| master（主线）   | 2.7.1     | 2.7.1       | 8.2.RC1 | Python3.10 |

## 1、安装 vllm 和 vllm-ascend
```bash
# vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 38217877aa70041c0115ee367b75197af9cbc5ad
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 1de16ead8eecfec8903ec1b330b27a4fa2593c35
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
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
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 796871d7d092f7cbc6a64e7f4a3796f7a2217f5e
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

**注意**：安装插件前需要保证verl源码安装，否则插件不能生效。如果无法源码安装verl，需要指定verl源码路径：

```bash
VERL_PATH=path_to_verl pip install -e .
```

## 5、内存管理优化（可选）

安装 jemalloc 以优化内存管理

可通过源码编译安装，前往官方仓库获取最新稳定版本

安装步骤：
``` python
    tar -xvf jemalloc-{version}.tar.bz2
    cd jemalloc-{version}
    ./configure --prefix=/usr/local
    make
    make install
```

安装完成后设置环境变量(假设安装路径为 `/usr/local/lib/libjemalloc.so.2`): 
``` python
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
```

# 启动训练

安装成功后，将 `MindSpeed-RL/tests/verl_examples` 下提供的参考配置脚本放入 verl 目录下，具体为：

`configs` 目录提供具体模型及算法配置

`dapo`及`grpo`目录提供与 `configs` 对应的执行脚本，运行时配置好该脚本中的 `DEFAULT_SH` 即可拉起

# verl_npu开发

## verl_npu功能特性

verl_npu提供下述功能特性来实现极简易用的NPU集成：

1. **Patch 注入** - 使用git原生patch程序对Patch文件注入

2. **Patch Summary** - 对Patch注入结果进行概述

3. **Patch 版本管理** - 对当前依赖库多个版本进行统一

4. **Patch 结果探测** - 探测Patch注入结果

## 开发指南

### 快速开始

如果您需要为新的NPU硬件或新的verl版本开发插件，请参考：

- **[快速开始指南](docs/quick_start.md)** - 快速了解框架概览

### 验证开发结果

verl_npu开发完成后，按照上述安装步骤重新安装插件.