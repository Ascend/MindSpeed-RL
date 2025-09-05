



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
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
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
git clone https://gitee.com/ascend/MindSpeed-RL.git
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