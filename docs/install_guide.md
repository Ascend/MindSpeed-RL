# 安装指南

##  版本配套表

MindSpeed RL支持Atlas 800T A2等昇腾训练硬件形态。软件版本配套表如下：

| MindSpeed RL版本 | Megatron版本 | PyTorch版本 | torch_npu版本 | CANN版本  | Python版本 |
| ---------------- | ------------ |-----------|-------------|---------| ---------- |
| master（主线）   | Core 0.8.0   | 2.5.1     | 2.5.1       | 8.2.RC1 | Python3.10 |
| 2.1.0（商分）    | Core 0.8.0   | 2.5.1     | 2.5.1       | 8.2.RC1 | Python3.10 |
| 2.0.0（预览）    | Core 0.8.0   | 2.5.1     | 2.5.1       | 8.1.RC1 | Python3.10 |

[昇腾辅助软件](https://gitcode.com/Ascend/pytorch/tree/master#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中有更多关于PyTorch和CANN的版本信息。

## 安装依赖的软件

在安装MindSpeed RL之前，请参考[版本配套表](#版本配套表)，安装配套的昇腾软件栈，软件列表如下：

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">25.2.0</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">8.2.RC1</td>
  </tr>
  <tr>
    <td>Kernels（算子包）</td>
  </tr>
  <tr>
    <td>NNAL神经网络加速库（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>torch</td>
    <td rowspan="2">2.5.1</td>
  </tr>
  <tr>
    <td>torch_npu</td>
  </tr>
  <tr>
    <td>apex</td>
    <td rowspan="1">0.1</td>
  </tr>
  <tr>
    <td>ray</td>
    <td>2.42.1</td>
  </tr>
  <tr>
    <td>vllm</td>
    <td>main</td>
  </tr>
</table>

### 驱动固件安装

```shell
bash Ascend-hdk-*-npu-firmware_*.run --full
bash Ascend-hdk-*-npu-driver_*.run --full
```

### CANN安装

```shell
bash Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run --install
bash Atlas-A3-cann-kernels_8.2.RC1_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash Ascend-cann-nnal_8.2.RC1_linux-aarch64.run --install
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### vllm及相关依赖安装：
（注：环境中需要安装git，因为vllm的安装过程依赖git）
```shell
git clone -b releases/v0.9.1 https://github.com/vllm-project/vllm.git
cd vllm
git checkout b6553be1bc75f046b00046a4ad7576364d03c835
VLLM_TARGET_DEVICE=empty pip install .
cd ..
```

### vllm_ascend安装
（注：若机器为x86架构，需要先进行[PyTorch框架安装](#PyTorch框架安装)中的torch和torch_npu安装，安装vllm_ascend时，注释requirements.txt和pyproject.toml中torch和torch_npu的安装）
```shell
git clone -b v0.9.1-dev https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 8c7bc45
pip install -r requirements.txt
pip install -e .
```

### PyTorch框架安装
（注：可从[PTA官方代码仓](https://gitcode.com/Ascend/pytorch/releases)获取对应版本torch和torch_npu的whl包）
```shell
# 安装torch和torch_npu
pip install torch-2.5.1-cp310-cp310-*.whl
pip install torch_npu-2.5.1.*.manylinux2014_*.whl

# apex for Ascend 构建参考 https://gitcode.com/Ascend/apex
pip install apex-0.1.dev*.whl
```

### 高性能内存库 jemalloc 安装
为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。
#### Ubuntu 操作系统
通过操作系统源安装jemalloc（注意： 要求ubuntu版本>=20.04）：
```shell
sudo apt install libjemalloc2
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
# arm64架构(可通过 find /usr -name libjemalloc.so.2 确认文件是否存在) 
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# x86_64架构
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

#### OpenEuler 操作系统

执行如下命令重操作系统源安装jemalloc
```shell
yum install jemalloc
```
如果上述方法无法正常安装，可以通过源码编译安装
前往jamalloc官网下载最新稳定版本，官网地址:https://github.com/jemalloc/jemalloc/releases/
```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
#根据实际安装路径设置环境变量，例如安装路径为:/usr/local/lib/libjemalloc.so.2,可通过以下命令来设置环境变量(可通过 find /usr -name libjemalloc.so.2 确认文件是否存在)
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
```

> 如以上安装过程出现错误，可以通过提出issue获得更多解决建议。

## 准备源码
```shell
git clone https://gitcode.com/Ascend/MindSpeed-RL.git

git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout ca70c1338f1b3d1ce46a0ea426e5779ae1312e2e  # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt 
cp -r mindspeed ../MindSpeed-RL/
cd ..

# Megatron从github下载，请确保网络能访问
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-RL/
cd ..

git clone https://gitcode.com/Ascend/MindSpeed-LLM.git -b 2.1.0
cd MindSpeed-LLM
git checkout bf1e61f
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

cd ./MindSpeed-RL
pip install -r requirements.txt
pip install antlr4-python3-runtime==4.7.2 --no-deps 
```