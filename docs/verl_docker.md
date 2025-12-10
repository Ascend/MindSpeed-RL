# VeRL镜像使用文档

## 1.镜像环境导入

本镜像中已集成 CANN、PyTorch、PyTorch_npu、VeRL、MindSpeed、Megatron-LM、MindSpeed-RL/verl_npu等配套的依赖软件，仅需激活镜像后再导入**数据集和模型权重**即可使用。已按照[仓上安装文档](https://gitcode.com/Ascend/MindSpeed-RL/blob/2.2.0/rl-plugin/README.MD)完成了VeRL的NPU适配。

### 1.1 关键依赖软件版本

| 软件| 版本 |
|------| ----- |
|昇腾NPU固件与驱动| 25.3.RC1 |
| CANN | 8.3.RC1 |
| Python |  3.10.0 |
| Pytorch |  2.7.1 |
| Transformers | commit-id 836570e925|
| vLLM | commit-id 3821787aa7 |
| vLLM-ascend | commit-id 1de16ead8e |
| VeRL | commit-id 796871d7d0 |
| MindSpeed | commit-id 1cdd0abd75 |
| Megatron-LM | core_v0.12.1 |
| MindSpeedRL/rl-plugin | 2.2.0 |

### 1.2 导入镜像并构造容器

**下载镜像**
镜像归档在昇腾社区的[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/2a71f71cb92643baa95e527b39088e0e)，命令行下载方式如下
```
910B：docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:a2-arm
AtlasA3：docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:a3-arm
A+X：docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:ax-x86
```

查看docker是否建立成功，成功时会在REPOSITORY栏显示出swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3
```
docker images
```

**构造容器**

根据指定的swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:a2/a3-arm镜像构建一个名为verl_rc3的容器。

针对于单机8卡的机器，如Ascend910B系列机器，构造容器命令如下：

```
docker run -dit --ipc=host --network host --name 'verl_rc3'  \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /home/:/home/ \
    -v /data/:/data/ \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/bin/msnpureport:/usr/bin/msnpureport \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:a2-arm \
    /bin/bash
```

针对于单机16卡的机器，如Atlas A3系列机器，构造容器命令如下：

```
docker run -dit --ipc=host --network host --name 'verl_rc3'  \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci8 \
    --device=/dev/davinci9 \
    --device=/dev/davinci10 \
    --device=/dev/davinci11 \
    --device=/dev/davinci12 \
    --device=/dev/davinci13 \
    --device=/dev/davinci14 \
    --device=/dev/davinci15 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /home/:/home/ \
    -v /data/:/data/ \
    -v /mnt/:/mnt/ \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/bin/msnpureport:/usr/bin/msnpureport \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:a3-arm \
    /bin/bash
```

这条命令根据-v对文件进行映射，映射文件夹如下，也可以根据自己的需求添加更多映射文件夹：

● /usr/local/Ascend/driver ；
● /usr/local/Ascend/firmware ；
● /usr/local/sbin/ ；
● /home ；
● /data ;

查看容器是否建立成功，成功时会在IMAGE栏显示出swr.cn-south-1.myhuaweicloud.com/ascendhub/verl_pt27_25rc3:a3-arm，NAMES栏为verl_rc3的容器：

```
docker ps -a
```

**进入容器**

```
docker exec -it verl_rc3 bash
```

### 1.3 安全风险提示

在使用 Docker 容器运行VeRL 时，需要注意以下安全风险：

* ​**使用 root 用户运行**​：容器默认以 root 用户身份运行，可能带来安全隐患。建议在生产环境中创建非特权用户来运行应用程序。

> 在生产环境部署时，请根据实际安全要求调整容器配置，确保系统安全性。

### 1.4 激活CANN

cann相关包已经安装在`/usr/local/Ascend/cann/ascend-toolkit/`和`/usr/local/Ascend/cann/nnal`文件夹下：

```shell
source /usr/local/Ascend/cann/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/nnal/atb/set_env.sh
```

### 1.5 激活conda VeRL环境

执行以下命令，查看conda环境。

```shell
conda env list
```

```
# conda environments:
#
base                  *  /root/miniconda3
verl_pt27_25rc3          /root/miniconda3/envs/verl_pt27_25rc3
```

其中verl_pt27_25rc3为模型训练环境，包含VeRL强化学习框架及依赖，同时集成了评测工具aisbench运行环境。

执行以下命令，激活VeRL训练环境

```
conda activate verl_pt27_25rc3
```

注：适用A+X型号机器的镜像中无需该步骤

## 2. 模型训练

**以下将基于 veRL 仓库中的示例脚本，使用 Qwen3-32B 模型及 dapo-math-17k 数学领域数据集，详细介绍整体的强化学习训练流程，训练代码位于`/examples/verl`中**

### 2.1 训练数据集准备

执行以下命令，进入`/examples/datasets`文件夹，完成DAPO-Math-17k和Aime2024数据集下载

在huggingface下载数据集

```shell
#!/usr/bin/env bash
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"/examples"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/datasets/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/datasets/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${VERL_HOME}/data"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
fi

```
或在魔塔社区下载dapo-math-17k数据集
```
cd /examples/datasets
git clone https://www.modelscope.cn/datasets/AI-ModelScope/DAPO-Math-17k.git
cp DAPO-Math-17k/data/dapo-math-17k.parquet ./
rm -rf DAPO-Math-17k
```

### 2.2 模型下载

**执行以下命令，即可开始下载 Qwen3-32B 模型。**

```shell
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen3-32B --local-dir /path/to/local_dir
```

● **--local-dir：**模型保存路径。

### 2.3 启动训练

这里以**四台A3机器训练Qwen3-32B模型**为例，​如果需要使用A2机器启动训练，需修改yaml文件和训练脚本里的机器数量​。

​**主节点和从节点配置步骤一致**​，准备步骤如下：

**2.3.1 修改启动脚本**

打开 **`/examples/verl/start.sh`** 文件，该脚本文件用于配置**各类环境变量以及ray节点拉起**，进行如下修改：

- 环境变量：修改为RAY和通信等相关环境变
- DEFAULT_SH：修改为训练所用配置sh路径
- NNODES和NPUS_PER_NODE：修改为使用节点和每个节点NPU数量
- MASTER_ADDR：修改为对应主节点 IP。即所有节点的MASTER_ADDR 应该相同。
- SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME：修改为对应通信网卡（也可指定特定网卡），通信网卡可以通过以下命令获取。

```shell
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

```shell
# ……
# 修改为当前需要跑的用例路径
DEFAULT_SH="./test_dapo_qwen3_32b_fsdp2_A3.sh"
# ……
NNODES=4
NPUS_PER_NODE=16
# 修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"
# 修改为当前节点的通信网卡
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
# ……

```

**2.3.2 修改训练配置**

使用多机训练时，训练配置要相同，包括模型路径，数据集路径等。打开`/examples/verl/test_dapo_qwen3_32b_fsdp2_A3.sh`，脚本展示如下，需要修改的参数：​**NNODES、N_GPUS_PER_NODE、MODEL_PATH、CKPTS_DIR**​，其余参数可自定义：

- NNODES：修改为使用节点数量，与启动脚本保持一致
- N_GPUS_PER_NODE：修改为每个节点NPU数量，与启动脚本保持一致
- MODEL_PATH：修改为训练Qwen3-32b权重路径，需要下载
- CKPTS_DIR：修改为保存权重路径
- TRAIN_FILE、TEST_FILE：修改为训练所需的数据集路径

```shell
#!/usr/bin/env bash
# ……
# Ray
NNODES=4
N_GPUS_PER_NODE=16
# Paths
MODEL_PATH=# 修改为训练Qwen3-32b权重路径，需要下载
CKPTS_DIR=./ckpt/Qwen3-32B-save # 保存权重的路径
TRAIN_FILE=/examples/datasets/dapo-math-17k.parquet # 修改为训练数据集路径
TEST_FILE=/examples/datasets/dapo-math-17k.parquet  # 修改为测试数据集路径
# ……
```

### 2.4 分析训练日志

在训练模型的过程中，可以通过输出的日志，分析模型的训练过程，输出日志的格式如下所示。

```shell
step:1 - actor/grad_norm:0.04106094129383564 - 
……
critic/rewards/mean:-0.4141845703125 -
response_length/mean:7385.140625 - 
……
prompt_length/mean:152.890625 - 
……
timing_s/generate_sequences:1081.803466796875 - 
timing_s/gen:1315.2451746799998 - 
……
timing_s/update_actor:303.92059642999993 - 
timing_s/step:1738.0274970199998 - 
……
timing_per_token_ms/gen:0.17391938503519522 - 
……
perf/throughput:69.3938963605546 
……
```

在理想情况下，critic/rewards/mean 指标大体趋势应随着训练的进行逐步提升（允许在一定范围内有些许波动）。

## 3. AIS-Benchmark模型评测

**跑通以下模型评测流程，单机 A3 \* 16卡即可。评测文件夹位于镜像的/examples/benchmark中**

### 3.1 FSDP模型合并

在之前的模型训练中，使用了 FSDP 后端训练模型，因此需要将各个卡上碎片化的权重，重新合并为完整的模型权重。执行以下命令，即可调用 veRL 仓库中提供的 FSDP 模型合并脚本，获取完整的模型权重。

```shell
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/global_step_{num}/actor \
    --target_dir /path/to/qwen3-32b-after-rl
```

**backend：** 训练后端，选择 `fsdp`。
**local_dir：** FSDP 模型权重本地路径，可以在训练配置的保存路径中选择对应 step 的目录。
**target_dir：** 合并后模型权重本地保存路径。

### 3.2 数据集下载

在`/examples/benchmark/ais_bench/datasets/aime` 中下载aime数据集

```shell
# linux服务器内，处于工具根路径下
cd /examples/benchmark/ais_bench/datasets
mkdir aime/
cd aime/
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
unzip aime.zip
rm aime.zip
```

在{工具根路径}/ais_bench/datasets目录下执行tree aime/查看目录结构，若目录结构如下所示，则说明数据集部署成功。

```shell
# /examples/benchmark/ais_bench/datasets
aime
└── aime.jsonl
```

### 3.3 启动评测任务

**3.3.1 启动vllm serve推理服务端**

通过以下命令拉起NPU服务端，需要修改的参数：model和tensor-parallel-size。

- model：保存训练后权重转换完的huggingface模型地址；
- tensor-parallel-size：张量并行副本数，TP建议和训练时infer的配置保持一致；
- data-parallel-size：数据并行副本数，DP建议和训练时infer的配置保持一致，默认为1；
- port：可任意设置空闲端口；

```python
python -m vllm.entrypoints.openai.api_server \
       --model="path/to/Qwen3-32B/" \
       --served-model-name auto \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 24 \
       --max-model-len 22528 \
       --max-num-batched-tokens 22528 \
       --enforce-eager \
       --trust-remote-code \
       --distributed_executor_backend=mp \
       --tensor-parallel-size 8 \
       --data-parallel-size 1 \
       --generation-config vllm \
       --port 6380
```

**3.3.2 修改aisbench推理配置**

修改评测配置文件：

```
vim /examples/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py
```

python文件内容如下，**host_port需与服务端的port一致**，根据模型配置修改max_seq_len和max_out_len，推理示例设置为2k推20k：

```python
# vim /examples/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr='vllm-api-general',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 6380,
        max_seq_len = 2048,
        max_out_len = 20480,
        batch_size=48,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]
```

**3.3.3 启动aisbench推理客户端**

另起一个窗口进行评测，开启评测命令：

```
cd /examples/benchmark
ais_bench --models vllm_api_general --datasets aime2024_gen
```

**多轮评测脚本（可选）**

提供脚本以供多轮评测取平均值

```shell
#!/bin/bash
# 评测任务下发脚本 run.sh 循环多次执行评测任务
COMMAND="ais_bench --models vllm_api_general --datasets aime2024_gen"

# 循环执行16次
LOOP_TIMES=16
for ((i=1; i<=$LOOP_TIMES; i++)); do
    echo "第 $i 次执行："
    eval $COMMAND
done
```

### 3.4 评测结果

**训练前：**

| dataset  | version | metric   | mode | vllm-api-general |
| -------- | ------- | -------- | ---- | ---------------- |
| aime2024 | 187240  | accuracy | gen  | 38.99            |

**训练100步后：**

| dataset  | version | metric   | mode | vllm-api-general |
| -------- | ------- | -------- | ---- | ---------------- |
| aime2024 | 187240  | accuracy | gen  | 45.33            |

● **经过 DAPO 100步训练后模型在 AIME2024 数据集上的准确率提升了6.34%。**

● **可以在 /examples/benchmark/outputs/default/ 目录下查看推理结果和评测结果。**

## 4. 更多模型训练支持

进入 **`/examples/verl`** 文件，该路径下存在Qwen3-8b、Qwen3-30b-A3b、Qwen3-235b训练配置文件

```
test_dapo_qwen3_235b_megatron_A3.sh
test_dapo_qwen3_30b_fsdp_A3_6k.sh
test_dapo_qwen3_8b_fsdp.sh
test_dapo_qwen3_8b_megatron.sh
```

修改以上配置文件中的**权重和数据集路径**，打开其中一个训练配置文件如下

```shell
#!/usr/bin/env bash
# ……
# Ray
NNODES=2
N_GPUS_PER_NODE=16
# Paths
MODEL_PATH=# 修改为训练Qwen3-30b-A3b权重路径，需要下载
CKPTS_DIR=./ckpt/Qwen3-30B-save # 保存权重的路径
TRAIN_FILE=${TRAIN_FILE:-"/examples/datasets/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"/examples/datasets/aime-2024.parquet"}
# ……
```

**修改启动脚本中使用的训练用例路径以及节点与各节点NPU数量如下**

```
# ……
# 修改为当前需要跑的用例路径
DEFAULT_SH="./test_dapo_qwen3_30b_fsdp_A3_6k.sh"
# ……
NNODES=2
NPUS_PER_NODE=16
# ……
```

## 5. 注意事项

镜像中verl文件夹路径为/examples/verl，已经打好了相应的patch。
若需要本地拉取verl文件，使用以下命令：

```
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 796871d7d092f7cbc6a64e7f4a3796f7a2217f5e
pip install -e .
cd ..
```

此时verl文件夹中并没有打上相应的patch，需要在verl/verl/\_\_init\_\_.py中第65行加入以下命令：

```
if is_npu_available:
    import verl_npu
    print("NPU_acceleration enabled for verl")
```
