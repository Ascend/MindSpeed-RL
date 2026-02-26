# GRPO

## 简介
以 MindSpeed RL 仓库复现 [Group Relative Policy Optimization (GRPO) ](https://arxiv.org/pdf/2402.03300) 后训练方法为例来帮助用户快速入门，前期需要完成代码仓、环境、数据集以及权重等准备工作，再按照说明中的启动方式启动训练，以下为具体的操作说明。

## 环境配置
配置 MindSpeed RL 基础环境以及准备代码: 参考 [安装指南](../install_guide.md)。

## 数据预处理
配置好环境后，需要对数据集进行预处理。
目前支持DeepScaler和Math-17k数据集，对于能力较弱的模型（如qwen2.5_7b），建议使用Math-17k数据集来训练，打分器配套使用math_17k_acc。

如果使用 [**Math-17k**](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)，数据预处理操作如下：

```bash
# 读取math-17k数据集
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet
cd ..
```

数据预处理的yaml配置文件放置于configs/datasets文件夹下，通过以下命令进行数据集预处理：
[示例yaml配置文件](../../configs/datasets/math_17k.yaml)
```bash
#读取configs/datasets/math_17k.yaml文件 
bash examples/data/preprocess_data.sh math_17k
```

如果使用 [**DeepScaler**](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/tree/main) 数据集，数据预处理操作如下：

```bash
# 读取deepscaler数据集
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/main/deepscaler.json
cd ..
```

数据预处理的yaml配置文件放置于configs/datasets文件夹下，通过以下命令进行数据集预处理：
[示例yaml配置文件](../../configs/datasets/deepscaler.yaml)
```bash
# 读取configs/datasets/deepscaler.yaml文件 
bash examples/data/preprocess_data.sh deepscaler
```

数据集处理配置可以根据需求自行配置，以下是数据集处理的yaml文件中基础参数的介绍：
| 参数 | 说明 |
|------|------|
| `input` | 数据集的路径，需指定具体文件，例如 /datasets/deepscaler.json |
| `tokenizer_type` | 指定分词器的类型，例如 HuggingFaceTokenizer 使用 HuggingFace 库提供的分词器来对文本进行分词处理 |
| `tokenizer_name_or_path` | 指定分词器的名称或路径，路径具体到分词器所在目录即可 |
| `output_prefix` | 输出结果的前缀路径，例如 /dataset/data |
| `workers` | 设置处理数据时使用的 worker 数 |
| `prompt_type` | 用于指定对话模板，能够让 base 模型微调后能具备更好的对话能力，`prompt_type` 的可选项可以在 [configs/model/templates.json](../../configs/model/templates.json) 文件内查看关键词"name" |
| `log_interval` | 设置日志记录的间隔，每处理多少条数据时记录一次日志，用于监控数据处理的进度和状态 |
| `handler_name` | 指定处理数据的处理器名称 |
| `map_keys` | 指定数据处理时使用的映射字典，用于将原始数据中的字段映射到目标字段中：<br>- prompt：主指令/题目文本（Alpaca 格式里的 instruction）。例如把原始样本的 "problem" 作为指令<br>- query：可选的补充输入/上下文（Alpaca 格式里的 input）。没有就设为空串 ""<br>- response：目标答案/参考输出（训练时作为监督标签）。这里映射到原始样本的 "answer"<br>- system：可选的系统提示（chat 模板的 system 角色，用于全局行为设定）。没有就设为空串 "" |
| `dataset_additional_keys` | 指定在数据处理后需要保留的原始数据集中的额外字段，示例：`["labels"]` |

## 模型权重转换

根据 GRPO 算法要求，Actor 和 Reference 模型应该使用 SFT 微调后的模型进行初始化，Reward 模型应该使用规则奖励。GRPO 算法模型权重均使用 Megatron-mcore 格式，其他格式的权重需要进行模型权重转换。
```bash
#注意，MSRL里跑MoE模型默认设置moe_tp_extend_ep为true，转换权重时要加上下面的配置（不加会导致推理乱码）
--moe-tp-extend-ep
```
### 环境要求
**权重转换需要安装MindSpeed-LLM，建议在新建虚拟环境中安装，避免和MindSpeed RL 出现依赖冲突。**
如果环境里已有驱动和CANN，具体安装方法参考[“PTA”和“MindSpeed-LLM及相关依赖”安装指南](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/install_guide.md#pta%E5%AE%89%E8%A3%85)。

接下来，以 Qwen2.5-7B 模型的权重转换脚本为参考，相应的权重转换步骤如下:

### 获取权重文件
权重文件可以从 Huggingface 网站上获取，可以根据模型的使用场景灵活选择，在这里以
[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)  为参考。
### hf 转 mcore
在训练前，需要将 HuggingFace 权重转换成 Mcore 格式，具体权重转换方式可见[安装指南](../install_guide.md)中对应 commit id 的[MindSpeed-LLM 权重转换部分](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/solutions/checkpoint_convert.md)。


### mcore 转 hf（可选）
训练结束后，如果需要将生成的 Mcore 格式权重转换回 HuggingFace 格式,具体权重转换方式可见[安装指南](../install_guide.md)中对应 commit id 的[MindSpeed-LLM 权重转换部分](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/solutions/checkpoint_convert.md)。

## 启动训练

以 Qwen25 7B 模型为例,在启动训练之前，需要修改[ 启动脚本 ](../../examples/grpo/grpo_trainer_qwen25_7b.sh)的配置：
1. 根据实际安装路径设置 jemalloc 环境变量，用于更好管理内存，避免长跑过程中内存 OOM ，例如：export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 
2. 修改 DEFAULT_YAML 为指定的 yaml，目前已支持的配置文件放置在 configs / 文件夹下，同时需要对[ 环境变量配置文件 ](../../configs/envs/runtime_env.yaml) 中的 VLLM_DP_SIZE 及 HCCL_SOCKET_IFNAME 等参数进行配置，具体参数说明可见 [配置文件参数介绍](../features/grpo_yaml.md)；
3. 根据使用机器的情况，修改 NNODES 、NPUS_PER_NODE 配置， 例如单机 <term> Atlas A3</term> 训练系列产品可设置 NNODES 为 1 （双机 <term> Atlas A3</term> 训练系列产品可设置 NNODES 为2）、NPUS_PER_NODE 为16；单机 <term> Atlas A2</term> 训练系列产品可设置 NNODES 为 1 （双机 <term> Atlas A2</term> 训练系列产品可设置 NNODES 为2）、NPUS_PER_NODE 为8；
4. 如果是单机，需要保证 MASTER_ADDR 与 CURRENT_IP 一致，如果为多机，需要保证各个机器的 MASTER_ADDR 一致，CURRENT_IP 为各个节点的 IP (需要注意的是MASTER_ADDR 与 CURRENT_IP 不能设置为 localhost)；
5. 启动脚本中的 SOCKET_IFNAME 需要设置为 CURRENT_IP 所对应的通信网卡名；
```bash
#上述注意点修改完毕后，可启动脚本开启训练
bash examples/grpo/grpo_trainer_qwen25_7b.sh
```

***注意：***

***1、所有节点的代码、权重、数据等路径的层级要保持一致，且启动ray的时候都位于MindSpeed RL目录下***

***2、Moe模型开启mc2暂不支持EP跨超节点***

## 断点续训
进行断点续训时，需要注意配置以下参数：
  ```yaml
actor_config:
    finetune: false       <------- 断点续训时 finetune 参数设置为 false
    load: ./ckpt-32b      <------- 断点续训时 load 路径应为之前保存的权重路径
    save: ./ckpt
    no_load_optim: false  <------- 断点续训时 no_load_optim 应为 false
    no_load_rng: false    <------- 断点续训时 no_load_rng 应为 false
  
rl_config:
    integrated_mode_config:
      ref_model_load_path: ./Qwen2.5-7B-tp4 <------- 断点续训时，应在 ref_model_load_path 中配置原始模型权重路径，供 reference model 加载
  ```

## 时间分布和性能计算方式

日志打点具体内容可以参考[ 日志打点指标说明 ](../features/log_metrics.md) 中的详细说明。

* 全共卡方案下总时间分布

`timing/all` >= `timing/rollout` +`timing/old_log_p` + `timing/update`  +  `timing/reference_model` + `timing/reshard_to_train` + `timing/reshard_to_infer`  + `max(timing/non_overlap_rule_reward, timing/non_overlap_reference_model)`


* e2e_tps计算方式

$$
(\text{response\_length\_mean} + \text{prompt\_length\_mean}) \times \text{global\_batch\_size} \times \text{n\_samples\_per\_prompt} / \text{world\_size} \ / \text{time\_all}
$$

* update_tps计算方式

$$
(\text{response\_length\_mean} + \text{prompt\_length\_mean}) \times \text{global\_batch\_size} \times \text{n\_samples\_per\_prompt} / \text{world\_size} \ / \text{time\_update}
$$

* vllm_tps计算方式

$$
(\text{response\_length\_mean} + \text{prompt\_length\_mean}) \times \text{global\_batch\_size} \times \text{n\_samples\_per\_prompt} / \text{world\_size} \ / \text{time\_rollout}
$$

***注意：***

***以上计算公式中 ` time_all`、`time_update`、`time_rollout`、`response_length_mean` 和 `prompt_length_mean` 即分别对应于[ 日志打点指标说明 ](../features/log_metrics.md)里的`timing/all`、`timing/update`、`timing/rollout`、`response_length/mean`和`prompt_length/mean`，此处名字修改是为了区别于公式里的`/`计算符号；***

## 性能数据

| 模型                  | 机器型号     | GBS | n_samples | max_prompt_length | max_tokens | 端到端 tps | 
|---------------------|----------|-----|-----------|-------------------|------------|---------| 
| Qwen25-7B           | Atlas 900 A3 SuperPoD | 32  | 8         | 2048              | 2048       | 220     | 
| Qwen25-32B          | Atlas 900 A3 SuperPoD | 64  | 16        | 1024              | 2048       | 220     | 
| Qwen25-32B          | Atlas 900 A2 PODc | 64  | 16        | 1024              | 1024       | 100     |
| Qwen3-8B            | Atlas 900 A3 SuperPoD | 32  | 8         | 2048              | 8192       | 252     |
| DeepSeek-R1-671B    | Atlas 900 A3 SuperPoD | 384 | 32        | 1024              | 2048       | 210     |


***注意：***

***模型 token/p/s 性能数据会打印在日志中, 当前计算公式下，A3单卡性能需要将日志打印的token/p/s性能指数*2。***

## FAQ
Q：math-17k数据预处理过程中如果出现报错：
```shell
RuntimeError: Failed to import transformers.data.data_collator because of the following error (look up to see its traceback):
/usr/local/python3.10/lin/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0: cannot allocate memory in static TLS block
```

A：这是由于在ARM架构或某些Linux环境中，当程序尝试加载libgomp库时，可能会遇到静态TLS (线程局部存储) 内存分配失败。libgomp库在初始化的时候会占用静态TLS空间，但如果库加载顺序不当（例如，其他库先于libgomp加载并占用了TLS空间），会导致内存分配失败，进而引发以来该库的模块（如sklearn）无法导入。解决方法（仅当前终端生效）：
```shell
export LD_PRELOAD=${LD_PRELOAD}:$(find /usr/ -name libgomp-947d5fa1.so.1.0.0 | grep scikit_learn)
```