# 后训练方法 Ray GRPO

## 简介
[Group Relative Policy Optimization (GRPO) ](https://arxiv.org/pdf/2402.03300)是 Deepseek-Math中提出的训练方法，它移除了 PPO 中对 Critic 模型的依赖，而是通过计算同一prompt多次重复采样输出的相对奖励来估计优势函数，这一创新大大减少了显存占用，提高了算法在强化学习任务中的效率。

在 GRPO 方法中包含了三个关键模型：Actor，Reference，Reward。其中 Actor 和 Reference 模型是通过SFT后得到的策略模型，而 Reward 模型则是通过训练构建的奖励评估模型。GRPO 的核心训练目标是优化 Actor 模型的策略，使其在执行强化学习任务时能够产生更优的动作序列，更符合任务目标的预期。

# 使用说明
通过 MindSpeed-RL 仓库复现 GRPO 训练方法，前期需要完成代码仓及环境、数据集以及权重等准备工作，再按照说明中的启动方式启动训练，以下为具体的操作说明。

## 环境配置
配置 MindSpeed-RL 基础环境以及准备代码: 参考[安装指南](../install_guide.md)

## 数据预处理
配置好环境后，需要对数据集进行预处理。

以 [pe-nlp 数据集](https://huggingface.co/datasets/pe-nlp/Big-Math-RL-Verified-Filtered) 为例。

```shell
cd dataset/
wget https://huggingface.co/datasets/pe-nlp/Big-Math-RL-Verified-Filtered/resolve/main/data/train-00000-of-00001.parquet --no-check
cd ..
```

数据预处理的yaml配置文件放置于configs/datasets文件夹下，通过以下命令进行数据集预处理：
[示例yaml配置文件](../../configs/datasets/grpo_pe_nlp.yaml)
```shell
bash examples/data/preprocess_data.sh grpo_pe_nlp 
#读取configs/datasets/grpo_pe_nlp.yaml文件 
```
### 参数介绍
数据集处理配置可以根据需求自行配置，以下是数据集处理的yaml文件中基础参数的介绍：
* `input`：数据集的路径，需指定具体文件，例如/datasets/pe-nlp/train-00000-of-00001.parquet
* `tokenizer_type`：指定分词器的类型，例如 HuggingFaceTokenizer 使用 Hugging Face 库提供的分词器来对文本进行分词处理;
* `tokenizer_not_use_fast`：选择是否使用 fast 分词器版本,设定为 True 时不使用。fast 分词器通常在处理速度上有优势，但可能在某些情况下不适用或存在兼容性问题;
* `tokenizer_name_or_path`：指定分词器的名称或路径;
* `output_prefix`：输出结果的前缀路径;
* `workers`：设置处理数据时使用的 worker 数;
* `prompt_type`: 用于指定模型模板，能够让 base 模型微调后能具备更好的对话能力;
* `log_interval`：设置日志记录的间隔，每处理多少条数据时记录一次日志，用于监控数据处理的进度和状态;
* `handler_name`：指定处理数据的处理器名称;
* `seq_length`：设置序列长度;

## 模型权重转换

根据 GRPO 算法要求，Actor 和 Reference 模型应该使用 SFT 微调后的模型进行初始化，Reward 模型应该使用奖励模型训练后的模型进行初始化。GRPO 算法模型权重均使用 Megatron-mcore 格式，其他格式的权重需要进行模型权重转换。

接下来，以 Qwen25-7B 模型的权重转换脚本为参考，相应的权重转换步骤如下:

### 获取权重文件
权重文件可以从 Huggingface 网站上获取，可以根据模型的使用场景灵活选择，在这里以
[ Qwen2.5-7B-Instruct 权重](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)和
[Qwen2.5-Math-7B 权重](https://huggingface.co/Qwen/Qwen2.5-Math-7B)为参考。
### hf 转 mcore
在训练前，需要将 Hugging Face 权重转换成Mcore格式。

注：这里会调用到 mindspeed_llm 仓，进行权重转换时注意按照安装手册中的环境准备步骤，将 mindspeed_llm 放入 MindSpeed-RL 目录下。

脚本启动命令可以用bash启动，可根据真实情况配置脚本，[示例脚本](../../examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh)启动命令和配置参数如下：
```bash
# 路径按照真实情况配置
bash examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh
```
配置参数介绍
* `use-mcore-models`：启用 MCore 模型；
* `model-type`：指定模型类型，如 GPT;
* `load-model-type`：指定加载模型的类型，如 hf（Hugging Face）;
* `save-model-type`：指定保存模型的类型，如 mg;
* `target-tensor-parallel-size`：设置目标张量并行大小；
* `target-pipeline-parallel-size`：设置目标流水线并行大小；
* `add-qkv-bias`：是否进行 QKV 偏置；
* `load-dir`：加载 Hugging Face 权重的路径；
* `save-dir`：保存转换后权重的路径；
* `tokenizer-model`：分词器模型文件的路径；
* `model-type-hf`：指定 Hugging Face 模型类型，如 llama2;
* `params-dtype`：指定参数的数据类型，如 bf16。

### mcore 转 hf（可选）
训练结束后，如果需要将生成的mcore格式权重转换回 Hugging Face 格式，可以参照以下[示例脚本](../../examples/ckpt/ckpt_convert_qwen25_mcore2hf.sh)命令及脚本参数：

```bash
# 路径按照真实情况配置
bash examples/ckpt/ckpt_convert_qwen25_mcore2hf.sh
```
配置参数介绍

这里的参数与上文基本一致，注意以下几个事项即可：
1. 权重转换转回 Hugging Face 格式时，tp 和 pp 配置需配置为1；
2. load-model-type 参数配置为 mg，save-model-type 参数配置为 hf ;
3. save-dir 路径需要填入原始 HF 模型路径，新权重会存于 HF 原始权重文件下的 mg2hg 目录下，如/qwen2.5_7b_hf/mg2hg/
## 训练启动方式

### 单机

通过 --config-name 传递选取的 config 文件名（不添加.yaml后缀），可以通过下列命令直接启动训练（Qwen25 7B 模型可单机运行）。
目前已支持的配置文件放置在 configs/ 文件夹下。配置文件的具体说明见下文。

以 Qwen25 7B 模型为例,单机启动命令示例如下：
```bash
bash examples/grpo/grpo_trainer_qwen25_7b.sh
```

### 多机

以[ DeepSeekR1-ZERO-Qwen2.5-32B 复现](../solutions/r1_zero_qwen25_32b.md) 为例，多机启动步骤如下：

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
python cli/train_grpo.py --config-name r1_zero_qwen25_32b | tee logs/r1_zero_qwen25_32b_full.log
```

### 脚本启动训练


```shell
# 主节点
bash examples/r1/qwen25/r1_zero_qwen25_32b_master.sh r1_zero_qwen25_32b
```

```shell
# 其余子节点
bash examples/r1/qwen25/r1_zero_qwen25_32b_worker.sh
```


***注意：所有节点的代码、权重、数据等路径的层级要保持一致，且启动ray的时候都位于MindSpeed-RL目录下***

## 配置文件

由于 GRPO 训练过程中涉及 3 个模型，通过将模型参数和训练配置解耦的层级化参数配置，来简化 GRPO 训练的参数配置过程。RLXF 训练涉及到的所有配置文件均存储在 configs/rlxf 路径下，其中 model 文件夹下存储了模型结构相关的配置文件，GRPO 训练相关的模型参数文件以 grpo_{模型名}.yaml方式命名。

在每个 grpo_trainer 配置文件中，需要包含 defaults、megatron_training、rl_config、generate_config等字段的参数配置以及  GRPO 训练过程中涉及到的 3 个角色 actor，reward，ref 的配置。

1. defaults 负责引入模型配置文件，在 defaults 中应列举本配置文件中所需要用到的所有模型配置，模型配置可以在下方3个角色的具体配置中通过 model 字段进行选择。
2. megatron_training 字段设置的参数为所有 3 个角色通用的默认参数，这些参数可以在下方进一步被角色的单独配置所覆盖。
3. actor_config、ref_config 以及 reward_config：三个角色的训练配置。
4. rl_config: 在 GRPO 训练中的特性参数，以及 actor，reward，ref 模型的资源配置。
5. generate_config: 包含 tokenizer 相关配置、推理并行配置、vllm 模型相关设置以及样本采样参数配置。

### 参数解析

相较于普通模型训练，GRPO 增加一些特殊参数，以下将给出部分参数的意义解析。具体的参数配置格式请参照示例[配置文件](../../configs/grpo_trainer_qwen25_7b.yaml)。

### `defaults:`
引入模型配置(网络结构需要定义在model目录的yaml文件下)：
* `model`: qwen25_7b
### `megatron_training:`

* `stage`：用于指定训练算法，使用 Ray GRPO 训练须设置为`ray_grpo`；
* `global_batch_size`: 经过多少样本后 actor-train 和 rollout 权重同步；
* `data_path`: 数据集路径配置，例如 /dataset/data ；
* `tokenizer_name_or_path`: 分词器路径配置，可以配置为 Hugging Face 权重文件的文件夹路径，例如 /ckpt/qwen2.5_7b_hf/ ;
* `其余参数`: 其余参数为Megatron训练中的特性配置；

### `actor_config、ref_config 以及 reward_config：`
配置 GRPO 训练中 Actor 模型、Reference 模型和 Reward 模型的配置参数；当前支持不开启 Reward 模型，开启规则奖励进行打分，开启参数详见rl_config中的rule_reward参数。
* `tensor_model_parallel_size`：TP 并行策略数;
* `pipeline_model_parallel_size`：PP 并行策略数;
* `micro_batch_size`：mbs 数量;
* `lr`：学习率；
* `lr_decay_style`：学习率衰减配置；
* `min_lr`：最小学习率；
* `weight_decay`：权重衰减，用于防止模型过拟合；
* `lr_warmup_fraction`：学习率预热比例，在训练初期逐渐增大学习率的比例；
* `load`：模型加载的路径；
* `save`：模型保存的路径；
* `no_load_optim`：续训加载优化器状态；
* `no_load_rng`：续训加载数据随机数生成器；
* `no_save_optim`：保存优化器状态；
* `no_save_rng`：保存数据随机数生成器；

### `rl_config:`
* `blocking`：是否开启异步，默认为 False；
* `n_samples_per_prompt`：每条prompt的重用次数，一条 prompt 输入能输出 n 条 responese；
* `max_prompt_length`：GRPO 训练中最大 prompt 长度，默认为512；
* `clip_ratio`：Actor 模型训练计算损失函数时的 clip 比例，默认为0.2 一般取值范围 [0.1，0.3] 最大取值范围[0，1] 该数值越大允许策略更新的幅度越大，反之不然；
* `shuffle_mini_batch`：Actor 训练时是否对 minibatch 进行 shuffle，默认为 False；
* `actor_resource` ：分配给 Actor 模型的显卡数量；
* `reference_resource` ：分配给 Reference 模型的显卡数量；
* `reward_resource` ：分配给 Reward 模型的显卡数量；

    显卡资源配置格式为 :
    ```
    actor_resource:
        num_npus: 4
    ```
开启规则奖励开关后，不用分配资源给 reward_resource 参数，规则奖励参数配置如下：
* `rule_reward`: 开启后，使用规则奖励进行打分；
* `verifier_function`: 选择使用的规则奖励模型方法，例如["acc", "strict_format"] ；
* `verifier_weight`: 配置规则奖励模型权重，例如[1.0, 1.0]；

日志配置参数也在 rl_config 中进行配置，当前支持 wandb/tensorboard 日志输出：

tensorboard开关（若use_tensorboard和use_wandb同时为True，则tensorboard不生效）:
* `use_tensorboard`: 配置为 True 时打开 tensorboard；     

wandb开关:
* `use_wandb`: 配置为 True 时打开 wandb；            
* `wandb_project`:  project 名称配置；        
* `wandb_exp_name`: 实验名称配置；   
* `wandb_save_dir`: 本地存储 wandb 路径；


### `generate_config:`
#### tokenizer相关配置
* `micro_batch_size`：mbs 大小，推理时每次处理的样本数量；
#### 推理时的并行配置
* `infer_tensor_parallel_size`：TP并行策略数；
* `infer_pipeline_parallel_size`：PP并行策略数；
* `infer_expert_parallel_size`：EP并行策略数；
#### resharding 相关配置
* `offload_train_optimizer`：卸载训练节点优化器；
* `offload_train_grad`：卸载训练节点梯度；
* `offload_train_param`：卸载模型权重；
#### vllm 模型相关设置
vllm 模型参数 可以参照[vllm官网参数介绍](https://docs.vllm.ai/en/latest/serving/engine_args.html)：
* `max_num_seqs`：vllm 推理并发最大样本限制；
* `max_num_batched_tokens`：vllm 推理并发最大token限制；
* `gpu_memory_utilization`：GPU 内存利用率，指定推理时使用 GPU 内存的比例；
#### 采样配置
* `logprobs`：是否生成logprobs；
* `max_tokens`：单条response最大生成token数量；
* `temperature`：采样时的随机性参数；
* `detokenize`：是否将输出token重新转为文本；


## 复现效果
当前已成功复现DeepSeekR1-ZERO训练流程以及训练效果，详细的复现流程以及效果图展示在以下文档：

[DeepSeekR1-ZERO-Qwen2.5-7B](../solutions/r1_zero_qwen25_7b.md)

[DeepSeekR1-ZERO-Qwen2.5-32B](../solutions/r1_zero_qwen25_32b.md)
