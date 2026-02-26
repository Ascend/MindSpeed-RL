# DPO

## 简介
MindSpeed RL 仓库现已支持 [ Direct Preference Optimization (DPO) ](https://arxiv.org/abs/2305.18290) 算法。训练开始前需要完成代码仓、环境、数据集以及权重等准备工作，再按照说明中的启动方式启动训练，以下为具体的操作说明。

## 环境配置
配置 MindSpeed RL 基础环境以及准备代码: 参考 [安装指南](../install_guide.md)。

## 数据预处理
配置好环境后，需要对数据集进行预处理。

以 [**orca_dpo_pairs**](https://huggingface.co/datasets/Intel/orca_dpo_pairs/tree/main) 为例。

```bash
# 读取orca_dpo_pairs数据集
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/Intel/orca_dpo_pairs/resolve/main/orca_rlhf.jsonl
cd ..
```

数据预处理的yaml配置文件放置于configs/datasets文件夹下，通过以下命令进行数据集预处理：
[示例yaml配置文件](../../configs/datasets/orca_rlhf.yaml)
```bash
# 读取configs/datasets/orca_rlhf.yaml文件 
bash examples/data/preprocess_data.sh orca_rlhf
```

数据集处理配置可以根据需求自行配置，以下是数据集处理的yaml文件中基础参数的介绍：
| 参数 | 说明 |
|------|------|
| `input` | 数据集的路径，需指定具体文件，例如 ./datasets/orca_rlhf.json |
| `tokenizer_type` | 指定分词器的类型，例如 HuggingFaceTokenizer 使用 HuggingFace 库提供的分词器来对文本进行分词处理 |
| `tokenizer_name_or_path` | 指定分词器的名称或路径 |
| `output_prefix` | 输出结果的前缀路径，例如 ./dataset/data |
| `workers` | 设置处理数据时使用的 worker 数 |
| `prompt_type` | 用于指定对话模板，能够让 base 模型微调后能具备更好的对话能力，`prompt_type` 的可选项可以在 [configs/model/templates.json](../../configs/model/templates.json) 文件内查看关键词"name" |
| `log_interval` | 设置日志记录的间隔，每处理多少条数据时记录一次日志，用于监控数据处理的进度和状态 |
| `handler_name` | 指定处理数据的处理器名称 |
| `seq_length` | 设置数据预处理最大序列长度，超过了会过滤掉 |
| `map_keys` | 指定数据处理时使用的映射字典，用于将原始数据中的字段映射到目标字段中：<br>- prompt：主指令/题目文本（Alpaca 格式里的 instruction）。例如把原始样本的 "problem" 作为指令<br>- query：可选的补充输入/上下文（Alpaca 格式里的 input）。没有就设为空串 ""<br>- response：目标答案/参考输出（训练时作为监督标签）。这里映射到原始样本的 "answer"<br>- system：可选的系统提示（chat 模板的 system 角色，用于全局行为设定）。没有就设为空串 "" |

## 模型权重转换

根据 DPO 算法原理，Actor 和 Reference 模型可以均为同一模型。DPO 算法模型权重均使用 Megatron-mcore 格式，其他格式的权重需要进行模型权重转换。
```bash
#注意，MSRL里跑MoE模型默认设置moe_tp_extend_ep为true，转换权重时要加上下面的配置（不加会导致推理乱码）
--moe-tp-extend-ep
```
### 环境要求
**权重转换需要安装MindSpeed-LLM，建议在新建虚拟环境中安装，避免和MindSpeed RL 出现依赖冲突。**
如果环境里已有驱动和CANN，具体安装方法参考[“PTA”和“MindSpeed-LLM及相关依赖”安装指南](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/install_guide.md#pta%E5%AE%89%E8%A3%85)。

接下来，以 Qwen3-30B-A3B 模型的权重转换脚本为参考，相应的权重转换步骤如下:

### 获取权重文件
权重文件可以从 Huggingface 网站上获取，可以根据模型的使用场景灵活选择，在这里以
[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)  为参考。
### hf 转 mcore
在训练前，需要将 HuggingFace 权重转换成Mcore格式，具体权重转换方式可见[安装指南](../install_guide.md)中对应 commit id 的[MindSpeed-LLM 权重转换部分](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/solutions/checkpoint_convert.md)。

### mcore 转 hf（可选）
训练结束后，如果需要将生成的mcore格式权重转换回 HuggingFace 格式，具体权重转换方式可见[安装指南](../install_guide.md)中对应 commit id 的[MindSpeed-LLM 权重转换部分](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/solutions/checkpoint_convert.md)。

## 启动训练

以 Qwen3-30B 模型为例，在启动训练之前，需要修改[ 启动脚本 ](../../examples/dpo/dpo_trainer_qwen3_30b_a3b.sh)的环境变量的配置：
1. 根据使用机器的情况，修改 NNODES 、NPUS_PER_NODE 配置， 例如单机 <term> Atlas A3</term> 训练系列产品可设置 NNODES 为 1 （双机 <term> Atlas A3</term> 训练系列产品可设置 NNODES 为2）、NPUS_PER_NODE 为16；单机 <term> Atlas A2</term> 训练系列产品可设置 NNODES 为 1 （双机 <term> Atlas A2</term> 训练系列产品可设置 NNODES 为2）、NPUS_PER_NODE 为8；
2. 如果是单机，需要保证 MASTER_ADDR 与 CURRENT_IP 一致，如果为多机，需要保证各个机器的 MASTER_ADDR 一致，CURRENT_IP 为各个节点的 IP；
```bash
#上述注意点修改完毕后，可启动脚本开启训练
bash examples/dpo/dpo_trainer_qwen3_30b_a3b.sh
```

***注意：***

***1、所有节点的代码、权重、数据等路径的层级要保持一致，且启动ray的时候都位于MindSpeed RL目录下***

***2、Moe模型开启mc2暂不支持EP跨超节点***

## 断点续训
进行断点续训时，需要注意配置以下参数：
  ```yaml
megatron_training:
    finetune: false       <------- 断点续训时 finetune 参数设置为 false
    load: ./ckpt-30b      <------- 断点续训时 load 路径应为之前保存的权重路径
    save: ./ckpt
    no_load_optim: false  <------- 断点续训时 no_load_optim 应为 false
    no_load_rng: false    <------- 断点续训时 no_load_rng 应为 false
  ```

## 性能数据
| 模型 | 机器型号     | GBS | 集群 | 方案 | 序列 | 性能             | 
|---|----------|---|---|---|---|----------------| 
| Qwen3-30B-A3B | Atlas 900 A2 PODc | 64 | 2x8 | 全参 | dynamic | 2.78 samples/s |
| Qwen3-30B-A3B | Atlas 900 A3 SuperPoD | 64 | 2x8 | 全参 | dynamic | 7.19 samples/s |