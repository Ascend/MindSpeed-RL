# 安装指南

##  版本配套表

MindSpeed RL支持Atlas 800T A2等昇腾训练硬件形态。软件版本配套表如下：

| MindSpeed RL版本 | Megatron版本 | PyTorch版本 | torch_npu版本 | CANN版本 | Python版本 |
| ---------------- | ------------ | ----------- | ------------- | -------- | ---------- |
| master（主线）   | Core 0.8.0   | 2.5.1       | 2.5.1         | 8.1.RC1  | Python3.10 |
| 2.0.0（预览）    | Core 0.8.0   | 2.5.1       | 2.5.1         | 8.1.RC1  | Python3.10 |

[昇腾辅助软件](https://gitee.com/ascend/pytorch#昇腾辅助软件)中有更多关于PyTorch和CANN的版本信息。

## 安装依赖的软件

在安装MindSpeed RL之前，请参考[版本配套表](#版本配套表)，安装配套的昇腾软件栈，软件列表如下：

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">25.0.RC1</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">8.1.RC1</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
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
    <td>0.7.3</td>
  </tr>
</table>

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
source /usr/local/Ascend/nnal/atb/set_env.sh
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
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.7.3
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


### 高性能内存库 jemalloc 安装
为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。
#### Ubuntu 操作系统
通过操作系统源安装jemalloc（注意： 要求ubuntu版本>=20.04）：
```shell
sudo apt install libjemalloc2
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
# arm64架构
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 $LD_PRELOAD
# x86_64架构
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 $LD_PRELOAD
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
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 $LD_PRELOAD
```

> 如以上安装过程出现错误，可以通过提出issue获得更多解决建议。

## 准备源码
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
pip install antlr4-python3-runtime==4.7.2 --no-deps 
```