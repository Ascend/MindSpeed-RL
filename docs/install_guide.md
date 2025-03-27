## 安装指导

请参考首页[安装教程](../README.md)选择下载对应依赖版本。

### 驱动固件安装

```shell
bash Ascend-hdk-*-npu-firmware_*.run --full
bash Ascend-hdk-*-npu-driver_*.run --full
```

### CANN安装

```shell
bash Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run --install
bash Ascend-cann-kernels-*_8.1.RC1_linux-aarch64.run --install
bash Ascend-cann-nnal_8.1.RC1_linux-aarch64.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/rc1b0202/nnal/atb/set_env.sh
```

### PTA安装

```shell
# 安装torch和torch_npu
pip install torch-2.5.1-cp310-cp310-*.whl
pip install torch_npu-2.5.1.*.manylinux2014_aarch64.whl

# apex for Ascend 构建参考 https://gitee.com/ascend/apex
pip install apex-0.1.dev*.whl
```

### vllm及相关依赖安装：
（注：环境中需要安装git，因为vllm的安装过程依赖git）
```shell
git clone -b v0.7.3 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install .
```

### vllm_ascend安装
```shell
git clone -b v0.7.3-dev https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 0713836e95fe993feefe334945b5b273e4add1f1
pip install -e .
```

### 准备源码
```shell
git clone https://gitee.com/ascend/MindSpeed-RL.git 

git clone https://gitee.com/ascend/MindSpeed.git 
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df  # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt 
cp -r mindspeed ../MindSpeed-RL/
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git  # Megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-RL/
cd ..

git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout 421ef7bcb83fb31844a1efb688cde71705c0526e
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

cd ./MindSpeed-RL
pip install -r requirements.txt 
```