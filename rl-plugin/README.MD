



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
git clone https://gitcode.com/Ascend/MindSpeed-RL.git -b 2.2.0
cd MindSpeed-RL/rl-plugin
pip install -v -e .
cd ../..
```

**注意**：安装插件前需要保证verl源码安装，否则插件不能生效。如果无法源码安装verl，需要指定verl源码路径：

```bash
VERL_PATH=path_to_verl pip install -e .
```

# 启动训练

安装成功后，将 `MindSpeed-RL/tests/verl_examples` 下提供的参考配置脚本放入 verl 目录下，具体为：

`configs` 目录提供具体模型及算法配置

`dapo`及`grpo`目录提供与 `configs` 对应的执行脚本，运行时配置好该脚本中的 `DEFAULT_SH` 即可拉起


# 插件开发

## 插件框架特性

本插件框架提供下述特性来实现无侵入式的NPU集成：

1. **Module Patch 特性** - 动态扩展现有类和模块的功能

2. **Variable Patch 特性** - 运行时修改外部模块中的字典和列表变量

3. **Module Alias 特性** - 注入模块别名，实现透明的模块替换

## 开发指南

### 快速开始

如果您需要为新的NPU硬件或新的verl版本开发插件，请参考：

- **[快速开始指南](docs/quick_start.md)** - 快速了解框架概览

- **[Module Patch特性](docs/module_patch_feature.md)** - 详细的模块扩展修改特性文档

- **[Variable Patch特性](docs/variable_patch_feature.md)** - 详细的变量修改特性文档

- **[Module Alias特性](docs/module_alias_feature.md)** - 详细的模块别名修改特性文档

### 验证开发结果

插件开发完成后，按照上述安装步骤重新安装插件.
