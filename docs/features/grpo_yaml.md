## GRPO配置参数简介
MindSpeed RL 通过将模型参数和训练配置解耦的层级化参数配置，来简化 GRPO 训练的参数配置过程。强化学习训练涉及到的所有配置文件均存储在 configs/ 路径下，其中 model 文件夹下存储了模型结构相关的配置文件，GRPO 训练相关的模型参数文件以 grpo_trainer_模型名_模型大小_机器型号.yaml方式命名。

在 grpo_trainer 配置文件中，主要包含 defaults、megatron_training、actor_config、rl_config、generate_config 字段的参数配置:

| 参数名 | 说明 |
|--------|------|
| `defaults` | 引入需要使用的模型配置文件，其下的模型配置（如`model`）<br> 可在 `megatron_training`、`actor_config` 等具体配置中被选择使用 |
| `megatron_training` | 训练引擎的通用默认参数配置 |
| `actor_config` | Actor 模型和 Reference 模型的训练配置参数 |
| `rl_config` | GRPO 训练中的特有参数，以及相关模型的资源配置 |
| `generate_config` | 包含分词器设置、推理并行配置、vLLM 引擎参数及生成采样参数等 |

## 参数解析

相较于普通模型训练，GRPO 增加一些特殊参数，以下将给出部分参数的意义解析。具体的参数配置格式请参照示例 [配置文件](../../configs/grpo_qwen25_7b_A3.yaml)。

### `defaults:`
引入模型配置(网络结构需要定义在model目录的yaml文件下)：
* `model`: qwen25_7b
### `megatron_training:`

| 参数名 | 说明 |
|--------|------|
| `stage` | 用于指定训练算法，使用 Ray GRPO 训练须设置为 `ray_grpo` | `ray_grpo` |
| `global_batch_size` | 经过多少样本后 actor-train 和 rollout 权重同步 |
| `data_path` | 数据集路径配置，例如 `/dataset/data`，注意带前缀 |
| `tokenizer_name_or_path` | 分词器路径配置，可以配置为 Hugging Face 权重文件的文件夹路径，例如 `/ckpt/qwen2.5_7b_hf/` |
| `其余参数` | 其余参数为 Megatron 训练中的特性配置 |

#### 全量重计算

| 参数名 | 说明 | 默认值 |
|--------|------|--------|
| `recompute_granularity` | 对于内存非常有限的情况，全量重计算只保存 Transformer 层或层组的输入激活值，其他部分全部重新计算 |
| `recompute_num_layers` | 指定重计算分组层数或重计算层数 |
| `recompute_method` | 全量重计算的方法选择：<br>• `uniform`：将 Transformer 层均匀划分组（每组大小由 `recompute_num_layers` 指定），按组存储输入和激活值<br>• `block`：将前 `recompute_num_layers` 个 Transformer 层重计算，剩余层不进行重计算 |

### `actor_config：`
配置 GRPO 训练中 Actor 模型、Reference 模型和 Reward 模型的配置参数；当前支持不开启 Reward 模型，开启规则奖励进行打分，开启参数详见rl_config中的rule_reward参数。

| 参数名 | 说明 |
|--------|------|
| `micro_batch_size` | 梯度累积的 mbs 大小 |
| `tensor_model_parallel_size` | TP 并行策略数 |
| `pipeline_model_parallel_size` | PP 并行策略数 |
| `lr` | 学习率 |
| `lr_decay_style` | 学习率衰减配置 |
| `min_lr` | 最小学习率 |
| `weight_decay` | 权重衰减，用于防止模型过拟合 |
| `lr_warmup_fraction` | 学习率预热比例，在训练初期逐渐增大学习率的比例 |
| `clip_grad` | 梯度裁剪系数 |
| `load` | 模型加载的路径 |
| `save` | 模型保存的路径 |
| `no_load_optim` | 续训加载优化器状态 |
| `no_load_rng` | 续训加载数据随机数生成器 |
| `no_save_optim` | 保存优化器状态 |
| `no_save_rng` | 保存数据随机数生成器 |

### `rl_config: `
配置 GRPO 训练中 Actor 模型、Reference 模型和 Reward 模型的配置参数；当前支持不开启 Reward 模型，开启规则奖励进行打分，开启参数详见rl_config中的rule_reward参数。
| 参数名 | 说明 |
|--------|------|
| `use_integrated_worker` | 是否开启全共卡模式 |
| `blocking` | 是否开启异步 |
| `gamma` | 奖励折扣因子 |
| `lam` | GAE参数 |
| `actor_forward_micro_batch_size` | actor model 前向计算 logp 的 mbs 大小 |
| `ref_forward_micro_batch_size` | ref model 前向计算 logp 的 mbs 大小 |
| `adv_estimator` | 优势计算方法 |
| `kl_penalty` | kl 散度惩罚系数 |
| `kl_ctrl_type` | kl loss 计算方法 |
| `init_kl_coef` | kl loss 所占权重 |
| `mini_batch_size` | 每 mini batch size 之后 actor 会更新一次 |
| `max_prompt_length` | GRPO 训练中最大 prompt 长度 ｜
| `clip_ratio` | Actor 模型训练计算损失函数时的 clip 比例，一般取值范围 [0.1，0.3] 最大取值范围[0，1]，该数值越大允许策略更新的幅度越大 |
| `entropy_coeff` | entropy loss 所占权重 |
| `n_samples_per_prompt` | 每条prompt的重用次数，一条 prompt 输入能输出 n 条 response |
| `guarantee_order` | 是否开启TransferDock保序 |
| `shuffle_mini_batch` | Actor 训练时是否对 minibatch 进行 shuffle |
| `log_max_throughput` | 配置tps计算时是否使用max值 |
| `num_cpus_for_local_task` | ray 进程配置的 cpu 数量 |
| `actor_resource` | 分配给 Actor 、Reference模型的显卡数量 |

#### 显卡资源配置
    ```
    actor_resource:
        num_npus: 4
    ```
#### 规则奖励配置

| 参数名 | 说明 |
|--------|------|
| `rule_reward` | 开启后，使用规则奖励进行打分 |
| `verifier_function` | 选择使用的规则奖励模型方法，例如 `["acc", "strict_format"]` |
| `verifier_weight` | 配置规则奖励模型权重，例如 `[1.0, 1.0]` |

#### 日志配置

tensorboard开关（若use_tensorboard和use_wandb同时为True，则tensorboard不生效）:
| 参数名 | 说明 |
|--------|------|
| `use_tensorboard` | 配置为 True 时打开 tensorboard（若 `use_tensorboard` 和 `use_wandb` 同时为 True，则 tensorboard 不生效） |

wandb开关:
| 参数名 | 说明 |
|--------|------|
| `use_wandb` | 配置为 True 时打开 wandb |
| `wandb_project` | wandb project 名称配置 |
| `wandb_exp_name` | wandb 实验名称配置 |
| `wandb_save_dir` | 本地存储 wandb 数据的路径 |


### `generate_config:`
#### 推理时的并行配置
| 参数名 | 说明 |
|--------|------|
| `infer_tensor_parallel_size` | TP并行策略数 |
| `infer_pipeline_parallel_size` | PP并行策略数，当前未支持该功能，设置为 '1' |
| `infer_expert_parallel_size` | EP并行策略数 |
#### resharding 相关配置
| 参数名 | 说明 |
|--------|------|
| `trust_remote_code` | 是否信任远程代码执行 |
| `offload_train_optimizer` | 卸载训练节点优化器 |
| `offload_train_grad` | 卸载训练节点梯度 |
| `offload_train_param` | 卸载模型权重 |
#### vllm 模型相关设置
vllm 模型参数 可以参照 [vllm官网参数介绍](https://docs.vllm.ai/en/latest/serving/engine_args.html)：
| 参数名 | 说明 |
|--------|------|
| `max_num_seqs` | vllm 推理并发最大样本限制 |
| `max_model_len` | vllm 能够处理的最大输入序列长度(prompt+response) |
| `max_num_batched_tokens` | vllm 单步能处理的最大 token 数量 |
| `enforce_eager` | 使能PyTorch eager模式，默认开启，仅 DeepSeek V3 开启 torchair_graph 时需要关闭 |
| `torchair_graph` | DeepSeek V3 使能 torchair 图模式 |
| `enable_expert_parallel` | MOE 模型使能专家切分，需要 MOE 模型支持 |
| `dtype` | vllm 推理所使用的数据类型 |
| `gpu_memory_utilization` | GPU 内存利用率，指定推理时使用 GPU 内存的比例 |
| `num_scheduler_steps` | 在一个完整的调度周期内，调度器会将批处理请求分成多少个子步骤来执行 |
#### 采样配置
| 参数名 | 说明 |
|--------|------|
| `logprobs` | 是否生成logprobs |
| `max_tokens` | 单条response最大生成token数量 |
| `top_p` | vllm 筛选出概率累积和达到top_p的token集合，随后只在这个集合里进行采样 |
| `top_k` | vllm 会先选出概率最高的 top_k 个 token，然后在这 top_k 个 token 范围内进行采样 |
| `min_p` | vllm 过滤掉概率低于 min_p 的词元，不参与后续的采样过程 |
| `temperature` | 采样时的随机性参数 |
| `detokenize` | 是否将输出token重新转为文本 |