# DAPO

## 简介
MindSpeed RL 仓库复现 [Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) ](https://arxiv.org/abs/2503.14476) 后训练方法.前期需要完成代码仓、环境、数据集以及权重等准备工作，再按照说明中的启动方式启动训练，以下为具体的操作说明。

## 环境配置
配置 MindSpeed RL 基础环境以及准备代码: 参考 [安装指南](../install_guide.md)

## 数据预处理
配置好环境后，需要对数据集进行预处理。

以 [**Math 17K**](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) 为例。

```bash
# 读取 Math 17K 数据集
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet --no-check
cd ..
```

数据预处理的yaml配置文件放置于configs/datasets文件夹下，通过以下命令进行数据集预处理：
[示例yaml配置文件](../../configs/datasets/math_17k.yaml)
```bash
# 读取configs/datasets/math_17k.yaml文件 
bash examples/data/preprocess_data.sh math_17k
```

数据集处理配置可以根据需求自行配置，以下是数据集处理的yaml文件中基础参数的介绍：
* `input`：数据集的路径，需指定具体文件，例如/datasets/dapo-math-17k.parquet
* `tokenizer_type`：指定分词器的类型，例如 HuggingFaceTokenizer 使用 Hugging Face 库提供的分词器来对文本进行分词处理;
* `tokenizer_name_or_path`：指定分词器的名称或路径;
* `output_prefix`：输出结果的前缀路径，例如 /datasets/data;
* `workers`：设置处理数据时使用的 worker 数;
* `prompt_type`: 用于指定对话模板，能够让 base 模型微调后能具备更好的对话能力，`prompt-type` 的可选项可以在 `configs/model/templates.json` 文件内查看;
* `log_interval`：设置日志记录的间隔，每处理多少条数据时记录一次日志，用于监控数据处理的进度和状态;
* `handler_name`：指定处理数据的处理器名称；
* `seq_length`：设置数据预处理最大序列长度，超过了会过滤掉;

## 模型权重转换

根据 DAPO 算法要求，Actor 模型应该使用 SFT 微调后的模型进行初始化，Reward 模型应该使用规则奖励。DAPO 算法模型权重均使用 Megatron-mcore 格式，其他格式的权重需要进行模型权重转换。

接下来，以 Qwen25-32B 模型的权重转换脚本为参考，相应的权重转换步骤如下:

### 获取权重文件
权重文件可以从 Huggingface 网站上获取，可以根据模型的使用场景灵活选择，在这里以
[Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B/tree/main)  为参考。
### hf 转 mcore
在训练前，需要将 Hugging Face 权重转换成 Mcore 格式，具体权重转换方式可见[安装指南](../install_guide.md)中对应 commit id 的 [MindSpeed-LLM](https://gitee.com/ascend/MindSpeed-LLM) 权重转换部分 。

### mcore 转 hf（可选）
训练结束后，如果需要将生成的 Mcore 格式权重转换回 Hugging Face 格式,具体权重转换方式可见[安装指南](../install_guide.md)中对应 commit id 的 [MindSpeed-LLM](https://gitee.com/ascend/MindSpeed-LLM) 权重转换部分 。

## 启动训练

以 Qwen25 32B 模型为例,在启动训练之前，需要修改[ 启动脚本 ](../../examples/dapo/dapo_trainer_qwen25_32b.sh)的配置：
1. 根据实际安装路径设置 jemalloc 环境变量，用于更好管理内存，避免长跑过程中内存 OOM ，例如：export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 
2. 修改 DEFAULT_YAML 为指定的 yaml，目前已支持的配置文件放置在 configs / 文件夹下，具体参数说明可见 [配置文件参数介绍](../features/grpo_yaml.md), DAPO 在 GRPO 的参数下，引入了几个特性参数：
    ```yaml
    #token level loss策略
    token_level_loss: true                                <------- 默认开启

    #clip Higher策略
    clip_higher_enable: true                              <------- 默认关闭
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28

    #过长response惩罚措施
    overlong_buffer_enable: true                          <------- 默认关闭
    rollout_max_tokens : 2048                             <------- response最大长度
    overlong_buffer: 512                                  <------- 超长惩罚缓冲区大小
    overlong_buffer_penalty_factor: 1.0                   <------- 超长惩罚系数

    #动态采样过滤措施
    filter_groups_enable: true                            <------- 默认关闭
    filter_groups_metric: grpo/acc_for_dapo_rewards/mean  <------- 指定用于过滤的 metric，其中'acc_for_dapo'需要与verifier_function参数中的值保持一致
    filter_groups_max_batches: -1                         <------- 设置过滤的最大次数，-1 代表不限制最大次数
    filter_groups_train_batch_size: 32                    <------- 制定需要筛选出多少条数据才停止采样，建议与gbs值一致，或者是gbs值的二分之一

3. 根据使用机器的情况，修改 NNODES 、NPUS_PER_NODE 配置， 例如单机 A3 可设置 NNODES 为 1 、NPUS_PER_NODE 为16；
4. 如果是单机，需要保证 MASTER_ADDR 与 CURRENT_IP 一致，如果为多机，需要保证各个机器的 MASTER_ADDR 一致，CURRENT_IP 为各个节点的 IP；
```bash
#上述注意点修改完毕后，可启动脚本开启训练
bash examples/dapo/dapo_trainer_qwen25_32b.sh
```

***注意：所有节点的代码、权重、数据等路径的层级要保持一致，且启动ray的时候都位于MindSpeed-RL目录下***

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
      ref_model_load_path: ./Qwen2.5-32-tp8 <------- 断点续训时，应在 ref_model_load_path 中配置原始模型权重路径，供 reference model 加载
  ```

