# 结果奖励模型训练

## 介绍
奖励模型（Reward Model）主要用于在基于人类反馈的强化学习（RLHF）过程中对大语言模型的回答进行评估，以使得模型生成的结果更加符合人类的喜好。结果奖励模型（Outcome Reward Model, ORM）仅对最终的结果进行评分，而不关注推理的过程。ORM 的输入是一个问题及回答，输出是该回答与问题匹配程度的评分。


ORM 结构以预训练模型为基础，将最后的 unembedding layer 替换为一个输出特征数为1的线性层 value head，将模型的输出映射为一个标量评分。

ORM 训练过程中，loss 采用以下公式计算得出：

$$
loss(\theta) = -E_{(x, y_c, y_r) \sim D} \log(\sigma(r_\theta(x, y_c) - r_\theta(x, y_r)))
$$

其中，$r_\theta(x,y)$ 表示 ORM 对该 “问题$x$-回答$y$” 的评分，$y_c$ 表示符合人类偏好的回答（chosen），$y_r$ 表示不符合人类偏好的回答（reject），$D$ 表示人工排序的 Pairwise 数据集。

## 数据集

ORM 训练使用 Pairwise 数据集，每条数据包含一个问题及配对的两条回答，一条是相对符合人类偏好的回答（chosen），一条是相对不符合人类偏好的回答（reject）。
常用Pairwise数据集有：

- [orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)
- [alpaca_messages_2k_dpo_test](https://huggingface.co/datasets/fozziethebeat/alpaca_messages_2k_dpo_test)

`Pairwise` 数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：
```shell
cd dataset/
wget https://huggingface.co/datasets/Intel/orca_dpo_pairs/blob/main/orca_rlhf.jsonl
cd ..
```

使用MindSpeed-RL仓库中的preprocess_data.sh对数据进行预处理

```shell
bash examples/data/preprocess_data.sh alpaca_reward
```

数据预处理的文件在configs/datasets目录下，一般只需要配置输入和输出参数
- input 需配置为数据集目录或具体文件。如果是目录，则处理全部文件, 支持.parquet/.csv/.json/.jsonl/.txt/.arrow 格式
- output_prefix 输出文件名的路径和前缀

其他常用可配置参数含义如下
- handler_name 数据处理器的名称，该处理器负责执行某些特定的任务
- log_interval 日志间隔，表示每处理多少个样本后记录一次日志信息
- tokenizer_name_or_path 指定分词器（tokenizer）的名称或路径
- tokenizer_type 分词器类型，当前仅支持'HuggingFaceTokenizer'
- prompt_type 用于指定模型模板，能够让base模型微调后能具备更好的对话能力

## 模型准备

目前仓上已包含Qwen2.5-7B/32B的ORM模型训练脚本，相应模型的权重可以从huggingface获得
- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)

Reward 模型使用奖励模型训练后的模型进行初始化。模型权重使用Megatron-mcore格式，其他格式的权重需要进行模型权重转换。

将 Huggingface 格式转化为 Megatron 格式
```shell
bash examples/ckpt/ckpt_convert_qwen25_hf2mcore_orm.sh
```

将 Megatron 格式转化为 Huggingface 格式
```shell
bash examples/ckpt/ckpt_convert_qwen25_hf2mcore_orm.sh
```

在脚本内需要根据真实环境配置
- --load-dir 需配置为待转换的权重路径 
- --save-dir 需配置为转换后保存权重的路径
- --tokenizer-model 对应的分词器路径
- --target-tensor-parallel-size 转换目标的张量并行数
- --target-pipeline-parallel-size 转换目标的流水线并行数

## 启动训练方法

### 单机

参考[配置](#配置)，根据真实环境填写路径。进入项目目录后通过 [examples/rm/orm_trainer_qwen25_7b.sh](../examples/rm/orm_trainer_qwen25_7b.sh) 启动7B模型训练（单机）

### 多机

参考[配置](#配置)，根据真实环境填写路径。 进入项目目录后通过 [examples/rm/orm_trainer_qwen25_32b.sh](../examples/rm/orm_trainer_qwen25_32b.sh) 启动32B模型训练（多机）
在运行脚本前需要根据真实环境配置脚本中的环境变量

- MASTER_ADDR 主节点的IP
- MASTER_PORT 主节点的端口
- NNODES 参与训练的节点数
- NODE_RANK 该节点在集群内对应的RANK
- GLOO_SOCKET_IFNAME 可以通过 ifconfig 命令，找到本机IP对应的网卡名
- TP_SOCKET_IFNAME，HCCL_SOCKET_IFNAME 可与 GLOO_SOCKET_IFNAME 配置成一样的

在所有需要启动的机器内配置好脚本，在命令行统一运行即可启动多机训练。

### 配置

脚本使用的是configs下的orm_trainer_qwen25_7b.yaml，orm_trainer_qwen25_32b.yaml配置文件

在文件内需要根据真实环境配置
- tokenizer_name_or_path 需配置为 tokenizer对应路径
- data_path 需配置为 \<path\>/\<data prefix\> 的形式，需要保证加载的 bin 文件的文件名为`<data_prefix>_packed_(.*)_document(.*)`
- tensor_model_parallel_size 张量并行数
- pipeline_model_parallel_size 流水线并行数
- 需要保证并行数乘积能整除总卡数
- 并行配置要与[模型准备](#模型准备)中转换目标的并行配置一致

## 参考文献

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) 