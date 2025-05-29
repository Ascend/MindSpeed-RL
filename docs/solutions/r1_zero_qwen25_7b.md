# DeepSeek-R1-Zero-Qwen25-7B
R1-Zero模型是使用base模型，基于GPRO+规则奖励打分进行训练，本篇工作使用Qwen25-7B模型复现DeepSeek-R1-Zero在Math领域的工作。

## 整体流程示意图

![](../../sources/images/r1_zero/r1_zero_roadmap.png)


## 复现效果
### 训练细节

我们使用Qwen-2.5-7B base模型在deepscaler数据集上训练，使用标准的格式奖励和准确性奖励，训练超参如下：

|  迭代  | 学习率 |  gbs  |  采样数 | 温度 |  kl-coef | 输入长度 | 输出长度 | 规则奖励 | 奖励模型 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 800 | 1e-6 (constant) |  32  |  8  |  1.0  |  0.001  |  2048  |  2048  |  base_acc  | - |

**训练过程记录如下：**

**Reward曲线：**

![](../../sources/images/r1_zero/7b-reward-curses.png)

**Response length曲线：**

![](../../sources/images/r1_zero/7b-response-curses.png)


### 评测结果
|              Model              | MATH 500 (Pass@1) | AIME24  (Avg@24) | GPQA Diamond (Avg@4) |
|:-------------------------------:|:-----:|:-----:|:-----:|
| Qwen-2.5-7B  | 62.8 | 9.1 | 27.9 |
| **MindSpeed-RL/竞品 200iter** | **77**/75.4 | **13.7**/12 | **35.8**/26.2 |
| **MindSpeed-RL/竞品 400iter** | **76.2**/75.6 | **12.9**/12.6 | **40.9**/36.3 |
| **MindSpeed-RL/竞品 600iter** | **76.6**/78.4 | **14.5**/13.7 | **37.3**/35.3 |
| **MindSpeed-RL/竞品 800iter** | **76.4**/75.8 | **16.2**/12.9 | **36.3**/34.3 |


### Aha-Moment
我们观察到经过30iter后模型就能出现Aha-Moment，取典型样例如下：

- **训练前**

  ![](../../sources/images/r1_zero/normal_answer.png)
- **训练后**

  ![](../../sources/images/r1_zero/aha_moment.png)


## 环境配置
配置MindSpeed-RL基础环境以及准备代码，参考[安装指南](../install_guide.md)

## 模型选择
* Qwen2.5-7B [[**下载**]](https://huggingface.co/Qwen/Qwen2.5-7B)
该模型指令遵从度高，有一定概率能引导模型输出`<think>...</think><answer>...$\boxed{}</answer>`格式回复，训练曲线符合预期，在评测集上提升较大。

### 权重转换
在进行RL训练之前，模型需要从HuggingFace权重转换为megatron权重，可参考[**权重转换部分**](../algorithms/grpo.md)

```bash
# 设置需要的权重转换参数
# 全共卡方案actor和ref model使用TP4PP1，将脚本里改成TP4PP1配置
bash examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh

# 训练完后如需要转回HF格式
bash examples/ckpt/ckpt_convert_qwen25_mcore2hf.sh
```

## 模板构造

* R1-Zero复现需要在数据处理时加上prompt模板激发`<think>...</think><answer>...$\boxed{}</answer>`
  ```
  <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>Put your final answer within \\boxed{}.\n{你真正的问题}<|im_end|>\n<|im_start|>assistant\n{模型真正的回答}
  ```

* 以上为默认的qwen_r1模板，根据模型和数据的不同，用户可以在`configs/model/templates.json`添加自己的**自定义模板**


## 数据集
对于7B模型我们使用DeepScaler 40K来训练

* [**DeepScaler**](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/tree/main)

### 数据预处理
需要先配置数据处理的yaml文件(configs/datasets/deepscaler.yaml)
自定义数据集需要设置--map-keys映射，或重写自定义handler；具体参考[**数据集处理部分**](../algorithms/grpo.md)


**Qwen2.5-7B**
* 处理的时候默认使用qwen_r1的模板

  ```shell
  # 启动转换
  bash examples/data/preprocess_data.sh deepscaler
  ```

## 打分器
DeepSeek-R1-Zero训练的过程中仅使用了基于程序的打分器而没有使用ORM，我们在数学领域上的打分逻辑分为以下几个部分：

![](../../sources/images/r1_zero/rule_reward.png)

## 训练
### 背景

传统的PPO中需要一个通过广义优势估计（Generalized Advantage Estimation）计算得到的advantage，并依赖于和reward model同结构的需要同步训练的critic model计算得到价值函数(V)

GRPO通过分组采样n个输出，利用组内的平均奖励作为基线计算每个输出在组内的相对奖励，并基于相对奖励计算优势值，从而避免了引入额外的价值网络（critic model）

![](../../sources/images/r1_zero/grpo.png)

DeepSeek-R1-Zero的训练过程使用GRPO算法，将ORM（结果奖励模型）替换为基于规则的打分器。

### 配置准备

模型结构的配置文件位于configs/model下，训练配置文件位于configs/目录下，我们以qwen2.5-7b的A3配置为例[grpo_qwen25_7b_A3.yaml]，该配置用到了16die，为了进一步加速可以不断增加推理DP的数量。[**参数配置具体含义参考**](../features/grpo_yaml.md)

### 手动启动训练
与基于ray的其他强化训练一样，我们多机需要先在主节点初始化ray：

```shell
# 创建一个集群，端口6344，dashboard端口8260
ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260
```

随后，在其他节点加入主节点的集群：
```shell
# IP_ADDRESS 处填写主节点 IP 地址
ray start --address="IP_ADDRESS:6344"
```

最后，在主节点上启动训练：
```shell
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

python cli/train_grpo.py --config-name grpo_qwen25_7b_A3.yaml | tee logs/r1_zero_qwen25_7b_full.log
```

### 脚本启动训练

[**参数配置具体含义参考**](../algorithms/grpo.md)
```shell
# 主节点 在脚本文件中修改节点数、每个节点的卡数以及主节点的IP地址
bash examples/grpo/grpo_trainer_qwen25_7b.sh
```

```shell
# 其余子节点 在脚本文件中修改节点数、每个节点的卡数以及主节点的IP地址
bash examples/grpo/grpo_trainer_qwen25_7b.sh
```


***注意：所有节点的代码、权重、数据等路径的层级要保持一致，且启动ray的时候都位于MindSpeed-RL目录下***
