## 1. 镜像环境导入

### 1.1. 关键依赖软件版本

本镜像中已集成 CANN、PyTorch、PyTorch_npu、MindSpeed-RL、MindSpeed、Megatron-LM和MindSpeed-LLM等配套的依赖软件，仅需激活镜像后再导入**数据集和模型权重**，即可使用。

| 软件              | 版本                                               |
| ----------------- | -------------------------------------------------- |
| 昇腾NPU固件与驱动 | 25.3.RC1                                           |
| CANN              | 8.3.RC1                                            |
| Python            | 3.10                                               |
| torch             | 2.5.1                                              |
| torch_npu         | 2.5.1                                              |
| MindSpeed-RL      | v2.2.0                                             |
| MindSpeed         | commit id ca70c1338f1b3d1ce46a0ea426e5779ae1312e2e |
| Megatron-LM       | core_r0.8.0                                        |
| MindSpeed-LLM     | commit id bf1e61f                                  |
| vLLM              | commit id b6553be1bc75f046b00046a4ad7576364d03c835 |
| vLLM-ascend       | commit id 8c7bc45                                  |

### 1.2. 导入镜像并构造容器

**下载并导入镜像**

准备一台计算机，要求安装的docker版本必须为1.11.2及以上，以root登录docker所在的虚拟机，从昇腾社区下载并导入镜像：

```shell
# 昇腾 A2 系列服务器
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed_rl_pt25_25rc3:2.2.0-A2-ARM

# 昇腾 A3 系列服务器
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed_rl_pt25_25rc3:2.2.0-A3-ARM

# 昇腾 A+X 系列服务器（昇腾 AI 加速卡搭配 X86 架构 CPU的服务器）
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed_rl_pt25_25rc3:2.2.0-AX-X86
```

查看docker是否建立成功，成功时会在REPOSITORY栏显示swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed_rl_pt25_25rc3：

```shell
docker images
```

**构造容器**

根据指定的“镜像名称：镜像标签”构建容器。

针对于单机8卡的机器，如Ascend910B系列机器，构造容器命令如下：
```shell
docker run -dit --ipc=host --network host --name '容器名' --device=/dev/davinci0 \
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
    镜像名称：镜像标签 /bin/bash
```

针对于单机16卡的机器，如Ascend910系列机器，构造容器命令如下：
```shell
docker run -dit --ipc=host --network host --name '容器名' --device=/dev/davinci0 \
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
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/bin/msnpureport:/usr/bin/msnpureport \
    镜像名称：镜像标签 /bin/bash
```

这条命令根据-v对文件进行映射，映射文件夹如下，也可以根据自己的需求添加更多映射文件夹：

- /usr/local/Ascend/driver ；
- /usr/local/Ascend/firmware ；
- /usr/local/sbin/ ；
- ......

查看容器是否建立成功，成功时会在IMAGE栏显示出容器名：

```shell
docker ps
```

**进入容器**

```shell
docker exec -it 容器名 bash
```

注：容器使用tips：

```auto
# 激活容器
docker start <容器ID或容器名>

# 停止运行中的容器（若有）
docker stop <容器ID或容器名>

# 删除容器（无论是运行还是停止状态）
docker rm <容器ID或容器名>
```
### 1.3. 安全风险提示

在使用 Docker 容器运行 MindSpeed-RL 时，需要注意以下安全风险：

- **使用 root 用户运行**：容器默认以 root 用户身份运行，可能带来安全隐患。建议在生产环境中创建非特权用户来运行应用程序。

> 在生产环境部署时，请根据实际安全要求调整容器配置，确保系统安全性。

### 1.4. 激活CANN

CANN相关包，包括Toolkit（开发套件）、Kernels（算子包）和NNAL神经网络加速库已经安装在`/usr/local/Ascend/ascend-toolkit/`和`/usr/local/Ascend/nnal/atb/`文件夹下。

通过下列命令激活CANN（仅在当前会话终端生效）：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 1.5. 激活conda mindspeed_rl环境

镜像环境里已经安装好了可以运行 mindspeed_rl 2.2.0 版本的 conda 环境。

查看当前镜像里的conda环境：

```shell
conda env list
```

其中mindspeed_rl_2.2.0是在镜像里配置好的模型训练环境，包含MindSpeed-RL 2.2.0版本的强化学习框架及依赖，同时集成了评测工具aisbench运行环境 。

执行以下命令，激活MindSpeed-RL训练环境：

```
conda activate mindspeed_rl_2.2.0
```

## 2. 模型训练与评估

[MindSpeed-RL](https://gitcode.com/Ascend/MindSpeed-RL/blob/2.2.0/README.md)：基于昇腾生态的强化学习加速框架，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 生态合作伙伴提供端到端的RL训推解决方案，支持超大昇腾集群训推共卡/分离部署、多模型异步流水调度、训推异构切分通信等核心加速能力。

以下将基于 MindSpeed-RL仓库中的示例脚本，使用 **Qwen3-32B** 模型及 **dapo-math-17k** 数学领域数据集，详细介绍整体的强化学习训练流程，并基于**AISBenchmark**评估套件，得到强化学习训练后在 [aime-2024 ](https://www.modelscope.cn/datasets/AI-ModelScope/AIME_2024) 测试数据集上的效果。

代码位于镜像的/env中。

**注意：这里Qwen3-32B模型需要双机协同训练，所以镜像导入、数据集准备、权重转换，以及模型脚本文件修改需要在两个A3环境上都进行**。

### 2.1. Qwen3-32B 模型准备

可以从Huggingface [Qwen3-32B：huggingface](https://huggingface.co/Qwen/Qwen3-32B) 或者ModelScope [Qwen3-32B：modelscope](https://www.modelscope.cn/models/Qwen/Qwen3-32B) 等网站下载开源模型权重。

权重可以基于网页直接下载，也可以基于命令行下载，这里示例是保存在/env/MindSpeed-RL/model_from_hf/Qwen3-32B/路径里，也可以存在/env/MindSpeed-LLM/model_from_hf/Qwen3-32B/文件夹里。命令行下载示例如下：

```
cd /env/MindSpeed-RL/model_from_hf/
mkdir Qwen3-32B/
cd Qwen3-32B/

wget https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json
wget https://huggingface.co/Qwen/Qwen3-32B/resolve/main/tokenizer_config.json
wget https://huggingface.co/Qwen/Qwen3-32B/resolve/main/tokenizer.json
wget https://huggingface.co/Qwen/Qwen3-32B/resolve/main/model-00001-of-000017.safetensors
wget https://huggingface.co/Qwen/Qwen3-32B/resolve/main/model-00002-of-000017.safetensors
# ......
```

### 2.2. DAPO-Math-17k 数据集准备

数据集下载：在/env/MindSpeed-RL/dataset中执行以下命令，即可完成 DAPO-Math-17k 数据集下载。

```shell
cd /env/MindSpeed-RL/dataset
mkdir math-17k
cd math-17k
wget https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet --no-check-certificate
```

数据预处理的yaml配置文件放置于/env/MindSpeed-RL/configs/datasets文件夹下：[math-17k yaml配置文件](https://gitcode.com/Ascend/MindSpeed-RL/blob/2.2.0/configs/datasets/math_17k.yaml)

查看并修改 math_17k.yaml 文件相关：

```yaml
cd /env/MindSpeed-RL
vim configs/datasets/math_17k.yaml
```

yaml脚本展示如下，需要修改的参数：**input、tokenizer_name_or_path和output_prefix**：

- input：数据集的路径，需指定具体文件；
- tokenizer_name_or_path：指定分词器的名称或路径，路径具体到分词器所在目录即可；
- output_prefix：预处理后的数据保存路径，生成的文件前缀为processed。

```yaml
input: "./dataset/math-17k/dapo-math-17k.parquet"
tokenizer_name_or_path: ./model_from_hf/Qwen3-32B/
output_prefix: ./dataset/math-17k/processed
handler_name: Math17kAlpacaStyleInstructionHandler
tokenizer_type: HuggingFaceTokenizer
workers: 8
log_interval: 1000
prompt_type: empty
dataset_additional_keys: [labels]
map_keys:  {"prompt":"prompt", "query":"", "response": "reward_model", "system":""}
```

通过以下命令进行数据集预处理:

```python
cd /env/MindSpeed-RL
bash examples/data/preprocess_data.sh math_17k
```

### 2.3. 权重转换：将下载的huggingface模型格式转换成Megatron-LM格式

**权重转换需要用到MindSpeed-LLM环境（镜像中已集成），如需详细信息，可查阅文档**[Huggingface权重转换到Megatron-LM格式](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/solutions/checkpoint_convert.md#21-huggingface权重转换到megatron-lm格式)。

查看并修改权重转换脚本文件：

```
cd /env/MindSpeed-LLM
vim examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

脚本展示如下，需要修改的参数：**target-tensor-parallel-size、target-pipeline-parallel-size、load-dir、save-dir和tokenizer-model**。

注1：target-tensor-parallel-size、target-pipeline-parallel-size 需要修改为训练脚本对应的yaml文件（详情见 2.4 启动训练章节）里 tensor_model_parallel_size 和 pipeline_model_parallel_size 值。

注2：如果训练脚本对应的yaml文件里有参数 **--num-layer-list** ，则需要在脚本里额外添加。

```yaml
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 2 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /env/MindSpeed-RL/model_from_hf/Qwen3-32B/ \
    --save-dir /env/MindSpeed-RL/model_weights/Qwen3-32B_mcore/ \
    --tokenizer-model /env/MindSpeed-RL/model_from_hf/Qwen3-32B/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3 # --num-layer-list 16,17,16,15 参数根据需要添加
```

执行权重转换脚本：

```
bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

上述步骤后，转换后的权重文件直接存储到了/env/MindSpeed-RL/model_weights/Qwen3-32B_mcore/里。

### 2.4. 启动训练

这里以A3机器为例，**如果需要使用A2机器启动训练，需修改yaml文件和训练脚本里的机器数量**。

**主节点和从节点配置步骤一致**，准备步骤如下：

#### **1. 训练配置yaml文件修改**：

在启动训练之前，需要修改[ Qwen3-32B训练yaml文件](https://gitcode.com/Ascend/MindSpeed-RL/blob/2.2.0/configs/dapo_qwen3_32b_A3.yaml)的配置：

```
cd /env/MindSpeed-RL
vim configs/dapo_qwen3_32b_A3.yaml
```

脚本展示如下，需要修改的参数：**tokenizer_name_or_path、data_path、load、save和num_npus**，其余参数可自定义：

- tokenizer_name_or_path：原始权重路径；
- data_path：预处理后的数据路径，包含前缀；
- train_iters: 训练的总迭代次数；
- save_interval: 定义 checkpoint 的保存间隔（按迭代次数）；
- load：转换后的权重路径，保存在/env/MindSpeed-RL/model_weights/Qwen3-32B_mcore/里；
- save：训练完后保存的权重路径，最好和load路径区分开；
- num_npus ：实际训练的卡数，因为A3双机协同训练，所以修改为总的npu数量=16\*2 （如果是A2机器，则为8\*2）。

```yaml
defaults:
  - model:
      - qwen3_32b

megatron_training:
  model: qwen3_32b
# ......
  tokenizer_name_or_path: ./model_from_hf/Qwen3-32B/
# ......
  save_interval: 100
  train_iters: 200
# ......
  data_path: ./dataset/math-17k/processed
  split: 100,0,0
  no_shuffle: true
  full_shuffle_instruction_dataset: false

actor_config:
  model: qwen3_32b
  micro_batch_size: 1
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 2
# ......
  load: ./model_weights/Qwen3-32B_mcore/
  save: ./model_weights/Qwen3-32B_mcore_save_weight

# ......
  actor_resource:
    num_npus: 32
```

#### 2. 训练脚本修改：

修改训练脚本相关配置，训练脚本参考：[dapo_trainer_qwen3_32b.sh](https://gitcode.com/Ascend/MindSpeed-RL/blob/2.2.0/examples/dapo/dapo_trainer_qwen3_32b.sh)。

```
vim examples/dapo/dapo_trainer_qwen3_32b.sh
```

脚本展示如下，需要修改的参数：**DEFAULT_YAML、LD_PRELOAD、NNODES、NPUS_PER_NODE、MASTER_ADDR、SOCKET_IFNAME、HCCL_SOCKET_IFNAME和GLOO_SOCKET_IFNAME**，其余参数可自定义：

- DEFAULT_YAML：为步骤一指定的 yaml文件地址；
- LD_PRELOAD：通过环境变量导入jemalloc，请确认该文件存在(可通过 find /usr -name libjemalloc.so.2 确认)；
- NNODES：使用节点数，此处为双机，数量为2；
- NPUS_PER_NODE：每个节点NPU数量，此处为A3，单机卡的数量为16（如果是A2机器，改为8）；
- MASTER_ADDR：修改为对应双机中的主节点IP，从节点脚本和主节点脚本对应的主节点IP一致；
- SOCKET_IFNAME：通信网卡，通过ifconfig获取；
- HCCL_SOCKET_IFNAME：华为昇腾 AI 处理器的集合通信库 HCCL 使用的环境变量，与SOCKET_IFNAME变量值相同；
- GLOO_SOCKET_IFNAME：指定 GLOO 绑定的网卡接口，与SOCKET_IFNAME变量值相同。

```python
pkill -9 python
ray stop --force
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

DEFAULT_YAML="dapo_qwen3_32b_A3"
YAML=${1:-$DEFAULT_YAML}
echo "Use $YAML"

# ......
# 如果是x86_64架构的机器，则修改为export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2

# ......
NNODES=2
NPUS_PER_NODE=16
MASTER_ADDR="xxx.xxx.xxx.xxx"
SOCKET_IFNAME="xxxxxxxxx"
export HCCL_SOCKET_IFNAME=xxxxxxxxx
export GLOO_SOCKET_IFNAME=xxxxxxxxx
```

启动训练脚本：

```
bash examples/dapo/dapo_trainer_qwen3_32b.sh
```

### 2.5.  AISBenchmark评估模型

#### 2.5.1. 数据集准备

为AISBenchmark评测准备aime数据集，保存在```/env/benchmark/ais_bench/datasets```里，操作步骤如下：

```
cd /env/benchmark/ais_bench/datasets
mkdir aime/
cd aime/
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
unzip aime.zip
rm aime.zip
```

下载成功后，aime文件夹下会存在```aime.jsonl```文件。

#### 2.5.2. 权重转换：将训练完后的Megatron-LM模型格式转换成huggingface格式

评估之前，需要把训练完后Megatron-LM格式的模型权重转回hf格式。

权重转换需要用到MindSpeed-LLM环境，详细参考文档[Megatron-LM权重转换到Huggingface格式](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/solutions/checkpoint_convert.md#22-megatron-lm权重转换到huggingface格式)。

查看并修改权重转换脚本文件相关：

```
cd /env/MindSpeed-LLM
vim examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh
```

脚本展示如下，需要修改的参数：load-dir和save-dir。

- load-dir：上述步骤训练完后保存的Megatron-LM模型地址，为/env/MindSpeed-RL/model_weights/ Qwen3-32B_mcore_save_weight，里面应具有checkpoint文件；
- save-dir：需要填入原始huggingface模型路径，新权重会存于```/env/MindSpeed-RL/model_from_hf/Qwen3-32B/mg2hf/```。

```yaml
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /env/MindSpeed-RL/model_weights/Qwen3-32B_mcore_save_weight \
    --save-dir /env/MindSpeed-RL/model_from_hf/Qwen3-32B/ \
    --params-dtype bf16 \
    --model-type-hf qwen3
```

执行权重转换脚本：

```
bash examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh
```

由于模型权重转换后使用的分词器文件为原始模型路径文件夹里所保存，此时需要在新权重路径里新保存一份：

```
cd /env/MindSpeed-RL/model_from_hf/Qwen3-32B/
cp vocab.json tokenizer_config.json tokenizer.json ./mg2hf
```

#### 2.5.3. 服务端拉起vllm

**单机评测**

通过以下命令拉起NPU服务端，需要修改的参数：model和tensor-parallel-size。

- /path/to/Qwen3-32B/mg2hf/：保存训练后权重转换完的huggingface模型地址；
- tensor-parallel-size：张量并行副本数，TP建议和训练时infer的配置保持一致；
- data-parallel-size：数据并行副本数，DP建议和训练时infer的配置保持一致，默认为1；
- port：可任意设置空闲端口；

```python
cd /env/vllm
vllm serve /path/to/Qwen3-32B/mg2hf/ \
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

**多机评测**

如果是Qwen3-235b之类的大模型，需要多机环境评测。给出双机评测的主节点和从节点脚本示例如下，**其中的所有参数均可按需修改**：

- 如果还有集群规模，可继续增加从节点脚本，脚本里需要给定主节点IP;
- data-parallel-size-local：当前节点上要启动的本地数据并行进程数，即当前机器内属于数据并行的进程数量，需要 ≤ 全局 --data-parallel-size，且所有节点的 size-local 之和 = 全局 data-parallel-size
- data-parallel-start-rank：数据并行起始rank（多节点分布式），需保证不同节点的 Rank 范围不重叠，比如示例脚本里主节点的data-parallel-start-rank为0，data-parallel-size-local为2，负责 Rank 0、1，所以从节点的data-parallel-start-rank为2。

主节点脚本：
```python
# 多级环境要设置以下的IP和网卡信息
# 环境IP
export HCCL_IF_IP=xx.xx.xx.xx

# 下面三条是网卡的名称，需要通过ifconfig查询出来
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"
#####--------------------------##########
# export OMP_PROC_BIND=false
# export OMP_NUM_THREADS=10
export ASCEND_LAUNCH_BLOCKING=1
export VLLM_USE_V1=1
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export LD_PRELOAD=/home/vllm_auto/libjemalloc.so:$LD_PRELOAD
#图模式--torchair

vllm serve /path/to/Qwen3-235B-A22B \
    --host 0.0.0.0 \
    --port 20002 \
    --data-parallel-size 4 \
    --data-parallel-rpc-port 13399 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank 0 \
    --data-parallel-address xx.xx.xx.xx \
    --no-enable-prefix-caching \
    --max-num-seqs 16 \
    --tensor-parallel-size 4 \
    --served-model-name dsv3 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 \
    --enforce-eager
```

从节点脚本：

```python
#下面的三条是网卡的名称，需要通过ifconfig查询出来
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"

#####--------------------------##########
# export OMP_PROC_BIND=false
# export OMP_NUM_THREADS=10
export ASCEND_LAUNCH_BLOCKING=1
export VLLM_USE_V1=1
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export LD_PRELOAD=/home/vllm_auto/libjemalloc.so:$LD_PRELOAD
#图模式--torchair

vllm serve /path/to/Qwen3-235B-A22B \
    --host 0.0.0.0 \
    --port 20002 \
    --headless \
    --data-parallel-size 4 \
    --data-parallel-rpc-port 13399 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address xx.xx.xx.xx \
    --no-enable-prefix-caching \
    --max-num-seqs 16 \
    --tensor-parallel-size 4 \
    --served-model-name dsv3 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 \
    --enforce-eager
```

**启动成功**

如果启动成功，则会话中显示如下：

```
INFO:     Started server process [237697]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

#### 2.5.4. 客户端

**另起一个会话**，使用ais-benchmark工具进行常用数据集下的评测。

注：另起会话后需要重新激活CANN，因为CANN只在当前会话被激活生效。

aisbench评测文件夹位于镜像的/env/benchmark/中，aime数据集保存在 **/env/benchmark/ais_bench/datasets/aime** 中。

修改评测脚本：

```
cd /env/benchmark/
vim ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py
```

脚本展示如下，host_port需与服务端的port一致，根据模型配置修改max_seq_len和max_out_len，推理示例设置为2k推20k：

```python
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

开启评测命令：

```
cd /env/benchmark
ais_bench --models vllm_api_general --datasets aime2024_gen
```

命令行推理结果示例展示：

```
| dataset | version | metric | mode | vllm-api-general |
|----- | ----- | ----- | ----- | -----|
| aime2024 | 187240 | accuracy | gen | 33.33 |
```

**可以在/env/benchmark/outputs/default/目录下查看更完整的推理结果和评估结果。**

#### 2.5.5. 多轮评测（可选）

如需多轮评测取平均值，提供脚本如下：

```
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

## 3. 更多模型训练

镜像中提供更多模型的训练脚本，包括qwen3-8b、qwen3-30b-a3b和qwen3-235b-a22b等。

model配置文件位于：```/env/MindSpeed-RL/configs/model```，yaml配置文件位于：```/env/MindSpeed-RL/configs```，模型训练脚本文件位于：```/env/MindSpeed-RL/examples```的dapo和grpo等文件夹下。

其余模型训练流程与本文示例qwen3-32b训练流程相似，请按照**实际使用的参数、机器数和卡数**进行配置。
