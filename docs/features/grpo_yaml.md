## GRPO配置参数简介
MindSpeed RL 通过将模型参数和训练配置解耦的层级化参数配置，来简化 GRPO 训练的参数配置过程。RLXF 训练涉及到的所有配置文件均存储在 configs/ 路径下，其中 model 文件夹下存储了模型结构相关的配置文件，GRPO 训练相关的模型参数文件以 grpo_trainer_模型名_模型大小_机器型号.yaml方式命名。在每个 grpo_trainer 配置文件中，需要包含 defaults、megatron_training、actor_config、rl_config、generate_config 字段的参数配置。

* `defaults:` 负责引入模型配置文件，在 defaults 中应列举本配置文件中所需要用到的所有模型配置，模型配置可以在 megatron_training 、actor_config 具体配置中通过 model 字段进行选择。
* `megatron_training:` 字段设置的参数为训练引擎通用的默认参数。
* `actor_config：`actor 、ref 的训练配置参数。
* `rl_config: ` 在 GRPO 训练中的特性参数，以及模型的资源配置。
* `generate_config:` 包含 tokenizer 相关配置、推理并行配置、vllm 模型相关设置以及样本采样参数配置。

## 参数解析

相较于普通模型训练，GRPO 增加一些特殊参数，以下将给出部分参数的意义解析。具体的参数配置格式请参照示例 [配置文件](../../configs/grpo_qwen25_7b_A3.yaml)。

### `defaults:`
引入模型配置(网络结构需要定义在model目录的yaml文件下)：
* `model`: qwen25_7b
### `megatron_training:`

* `stage`：用于指定训练算法，使用 Ray GRPO 训练须设置为`ray_grpo`；
* `global_batch_size`: 经过多少样本后 actor-train 和 rollout 权重同步；
* `data_path`: 数据集路径配置，例如 /dataset/data，注意带前缀；
* `tokenizer_name_or_path`: 分词器路径配置，可以配置为 Hugging Face 权重文件的文件夹路径，例如 /ckpt/qwen2.5_7b_hf/ ;
* `其余参数`: 其余参数为Megatron训练中的特性配置；

### `actor_config：`
配置 GRPO 训练中 Actor 模型、Reference 模型和 Reward 模型的配置参数；当前支持不开启 Reward 模型，开启规则奖励进行打分，开启参数详见rl_config中的rule_reward参数。
* `micro_batch_size`：梯度累积的 mbs 大小;
* `tensor_model_parallel_size`：TP 并行策略数;
* `pipeline_model_parallel_size`：PP 并行策略数;
* `lr`：学习率；
* `lr_decay_style`：学习率衰减配置；
* `min_lr`：最小学习率；
* `weight_decay`：权重衰减，用于防止模型过拟合；
* `lr_warmup_fraction`：学习率预热比例，在训练初期逐渐增大学习率的比例；
* `clip_grad`：梯度裁剪系数；
* `load`：模型加载的路径；
* `save`：模型保存的路径；
* `no_load_optim`：续训加载优化器状态，默认为false；
* `no_load_rng`：续训加载数据随机数生成器，默认为false；
* `no_save_optim`：保存优化器状态，默认为false；
* `no_save_rng`：保存数据随机数生成器，默认为false；


### `rl_config: `
* `use_integrated_worker`：是否开启全共卡模式，默认为 true;
* `blocking`：是否开启异步，默认为 true;
* `actor_forward_micro_batch_size`：actor model 前向计算 logp 的 mbs 大小;
* `ref_forward_micro_batch_size`：ref model 前向计算 logp 的 mbs 大小;
* `adv_estimator`：优势计算方法;
* `kl_ctrl_type`：kl loss 计算方法;
* `init_kl_coef`：kl loss 所占权重;
* `mini_batch_size`：每 mini batch size 之后 actor 会更新一次;
* `max_prompt_length`：GRPO 训练中最大 prompt 长度，默认为512;
* `clip_ratio`：Actor 模型训练计算损失函数时的 clip 比例，默认为0.2 一般取值范围 [0.1，0.3] 最大取值范围[0，1] 该数值越大允许策略更新的幅度越大，反之不然；
* `entropy_coeff`: entropy loss 所占权重;
* `n_samples_per_prompt`：每条prompt的重用次数，一条 prompt 输入能输出 n 条 responese;
* `guarantee_order`: 是否开启TransferDock保序，默认 False;
* `shuffle_mini_batch`：Actor 训练时是否对 minibatch 进行 shuffle，默认为 False;
* `actor_resource` ：分配给 Actor 、Reference模型的显卡数量;

#### 显卡资源配置
    ```
    actor_resource:
        num_npus: 4
    ```
#### 规则奖励配置
* `rule_reward`: 开启后，使用规则奖励进行打分；
* `verifier_function`: 选择使用的规则奖励模型方法，例如["acc", "strict_format"] ；
* `verifier_weight`: 配置规则奖励模型权重，例如[1.0, 1.0]；

#### 日志配置

tensorboard开关（若use_tensorboard和use_wandb同时为True，则tensorboard不生效）:
* `use_tensorboard`: 配置为 True 时打开 tensorboard；     

wandb开关:
* `use_wandb`: 配置为 True 时打开 wandb；            
* `wandb_project`:  project 名称配置；        
* `wandb_exp_name`: 实验名称配置；   
* `wandb_save_dir`: 本地存储 wandb 路径；


### `generate_config:`
#### 推理时的并行配置
* `infer_tensor_parallel_size`：TP并行策略数；
* `infer_pipeline_parallel_size`：PP并行策略数；
* `infer_expert_parallel_size`：EP并行策略数；
#### resharding 相关配置
* `offload_train_optimizer`：卸载训练节点优化器；
* `offload_train_grad`：卸载训练节点梯度；
* `offload_train_param`：卸载模型权重；
#### vllm 模型相关设置
vllm 模型参数 可以参照 [vllm官网参数介绍](https://docs.vllm.ai/en/latest/serving/engine_args.html)：
* `max_num_seqs`：vllm 推理并发最大样本限制；
* `max_model_len`：vllm 能够处理的最大输入序列长度(prompt+response)；
* `dtype`：vllm 推理所使用的数据类型；
* `gpu_memory_utilization`：GPU 内存利用率，指定推理时使用 GPU 内存的比例；
#### 采样配置
* `logprobs`：是否生成logprobs；
* `max_tokens`：单条response最大生成token数量；
* `top_p`：vllm 筛选出概率累积和达到top_p的token集合，随后只在这个集合里进行采样；
* `top_k`：vllm 会先选出概率最高的 top_k 个 token，然后在这 top_k 个 token 范围内进行采样；
* `min_p`：vllm 过滤掉概率低于 min_p 的词元，不参与后续的采样过程；
* `temperature`：采样时的随机性参数；
* `detokenize`：是否将输出token重新转为文本；