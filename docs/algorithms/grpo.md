# 后训练方法 Ray GRPO

## 简介
[Group Relative Policy Optimization (GRPO) ](https://arxiv.org/pdf/2402.03300)是 Deepseek-Math中提出的训练方法，它移除了 PPO 中对 Critic 模型的依赖，而是通过计算同一prompt多次重复采样输出的相对奖励来估计优势函数，这一创新大大减少了显存占用，提高了算法在强化学习任务中的效率。MindSpeed RL 仓库复现 GRPO 训练方法，前期需要完成代码仓、环境、数据集以及权重等准备工作，再按照说明中的启动方式启动训练，以下为具体的操作说明。

## 环境配置
配置 MindSpeed RL 基础环境以及准备代码: 参考 [安装指南](../install_guide.md)

## 数据预处理
配置好环境后，需要对数据集进行预处理。

以 [**DeepScaler**](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/tree/main) 为例。

```bash
# 读取deepscaler数据集
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/main/deepscaler.json --no-check
cd ..
```

数据预处理的yaml配置文件放置于configs/datasets文件夹下，通过以下命令进行数据集预处理：
[示例yaml配置文件](../../configs/datasets/deepscaler.yaml)
```bash
# 读取configs/datasets/deepscaler.yaml文件 
bash examples/data/preprocess_data.sh deepscaler
```

数据集处理配置可以根据需求自行配置，以下是数据集处理的yaml文件中基础参数的介绍：
* `input`：数据集的路径，需指定具体文件，例如/datasets/deepscaler.json
* `tokenizer_type`：指定分词器的类型，例如 HuggingFaceTokenizer 使用 Hugging Face 库提供的分词器来对文本进行分词处理;
* `tokenizer_name_or_path`：指定分词器的名称或路径;
* `output_prefix`：输出结果的前缀路径，例如 /datasets/data;
* `workers`：设置处理数据时使用的 worker 数;
* `prompt_type`: 用于指定对话模板，能够让 base 模型微调后能具备更好的对话能力，`prompt-type` 的可选项可以在 `configs/model/templates.json` 文件内查看;
* `log_interval`：设置日志记录的间隔，每处理多少条数据时记录一次日志，用于监控数据处理的进度和状态;
* `handler_name`：指定处理数据的处理器名称；
* `seq_length`：设置数据预处理最大序列长度，超过了会过滤掉;

## 模型权重转换

根据 GRPO 算法要求，Actor 和 Reference 模型应该使用 SFT 微调后的模型进行初始化，Reward 模型应该使用规则奖励。GRPO 算法模型权重均使用 Megatron-mcore 格式，其他格式的权重需要进行模型权重转换。

接下来，以 Qwen25-7B 模型的权重转换脚本为参考，相应的权重转换步骤如下:

### 获取权重文件
权重文件可以从 Huggingface 网站上获取，可以根据模型的使用场景灵活选择，在这里以
[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)  为参考。
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

这里的参数与上文一致，注意以下几个事项即可：
1. 权重转换转回 Hugging Face 格式时，tp 和 pp 配置需配置为1；
2. load-model-type 参数配置为 mg，save-model-type 参数配置为 hf ;
3. save-dir 路径需要填入原始 HF 模型路径，新权重会存于 HF 原始权重文件下的 mg2hg 目录下，如/qwen2.5_7b_hf/mg2hg/
## 启动训练

以 Qwen25 7B 模型为例,在启动训练之前，需要修改[ 启动脚本 ](../../examples/grpo/grpo_trainer_qwen25_7b.sh)的配置：
1. 修改 DEFAULT_YAML 为指定的 yaml，目前已支持的配置文件放置在 configs / 文件夹下，具体参数说明可见 [配置文件参数介绍](../features/grpo_yaml.md)；
2. 根据使用机器的情况，修改 NNODES 、NPUS_PER_NODE 配置， 例如单机 A3 可设置 NNODES 为 1 、NPUS_PER_NODE 为16；
3. 如果是单机，需要保证 MASTER_ADDR 与 CURRENT_IP 一致，如果为多机，需要保证各个机器的 MASTER_ADDR 一致，CURRENT_IP 为各个节点的 IP；
```bash
#上述注意点修改完毕后，可启动脚本开启训练
bash examples/grpo/grpo_trainer_qwen25_7b.sh
```

***注意：所有节点的代码、权重、数据等路径的层级要保持一致，且启动ray的时候都位于MindSpeed-RL目录下***


## 实践效果
当前已成功复现DeepSeekR1-ZERO训练流程以及训练效果，详细的复现流程以及效果图展示在以下文档：

[DeepSeekR1-ZERO-Qwen2.5-7B](../solutions/r1_zero_qwen25_7b.md)

[DeepSeekR1-ZERO-Qwen2.5-32B](../solutions/r1_zero_qwen25_32b.md)
