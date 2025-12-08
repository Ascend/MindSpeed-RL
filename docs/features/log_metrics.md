## 日志打点指标说明

强化学习算法迭代打屏日志说明如下：

**时间相关指标说明**

| 指标                                 | 说明                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| `timing/all`                         | 一次迭代总时间                                        |
| `timing/update`                      | 一次迭代中actor model进行update耗时                   |
| `timing/rollout`                     | 一次迭代中actor model进行rollout耗时                  |
| `timing/old_log_p`                   | 一次迭代中actor model计算log_p耗时                     |
| `timing/reference_model`             | 一次迭代中reference model计算log_p耗时                 |
| `timing/resharding_to_train`         | 权重转到训练mode耗时                                  |
| `timing/resharding_to_infer`         | 权重转到推理mode耗时                                  |
| `timing/adv`                         | 计算advantages耗时                                    |
| `timing/non_overlap_reference_model` | reference model计算log_p耗时的未被掩盖时间               |
| `timing/non_overlap_rule_reward`     | rule_reward耗时的未被掩盖时间                         |
| `timing/non_overlap_reward_model`    | reward_model耗时的未被掩盖时间                        |
| `timing/non_overlap_adv`             | advantages计算耗时的未被掩盖时间                        |
| `timing/rule_reward`                 | rule reward打分耗时                                   |
| `timing/reward_model`                | reward model打分耗时                                  |
| `timing/ref_onload`                  | reference model计算log_p过程中，onload耗时             |
| `timing/ref_offload`                 | reference model计算log_p过程中，offload耗时               |
| `timing/critic_model`                 | 一次迭代中critic model计算values耗时               |
| `timing/update_critic`                 | 一次迭代中critic model进行update耗时              |


**算法基本指标说明**

| 指标                                 | 说明                                                         |
|------------------------------------| ------------------------------------------------------------ |
| `actor/entropy`                    | 策略熵，表示策略的随机性或探索能力                           |
| `actor/kl_loss`                    | kl散度，衡量当前策略与参考策略（如旧策略或参考模型）之间的偏离程度 |
| `actor/pg_loss`                    | pg_loss，基于优势函数的策略梯度目标函数值，表示当前策略对提升奖励的学习能力。 |
| `actor/pg_clipfrac`                | actor model裁剪机制生效的比例，反映了策略更新幅度的稳定性         |
| `actor/ppo_kl`                     | PPO算法的实际 KL 散度                                        |
| `grad_norm`                        | 梯度范数，表示当前反向传播中参数梯度的整体幅度               |
| `{verifier_function}_rewards/mean` | 规则奖励打分的平均总奖励值                                   |
| `actor/lr`                         | actor model学习率，优化器当前使用的学习率                               |
| `response_length/mean`             | 平均生成长度，模型生成回复（response）的平均 token 数        |
| `response_length/min`              | 最短生成长度，当前 batch 中生成最短的 response 长度          |
| `response_length/max`              | 最长生成长度，当前 batch 中生成最长的 response 长度          |
| `prompt_length/mean`               | 平均输入长度，输入 prompt 的平均长度                         |
| `prompt_length/max`                | 最长输入长度，当前 batch 中最长的 prompt长度                 |
| `prompt_length/min`                | 最短输入长度，当前 batch 中最长的 prompt长度                 |
| `global_batch_size`                | 每次训练迭代所处理的总prompt数量                             |
| `n_samples_per_prompt`             | 每条prompt在rollout阶段生成的response数量                   |
| `world_size`                       | 在分布式训练中集群中总的设备数量（并行训练的总进程数）         |
| `e2e_tps`                          | 端到端的tokens/p/s指标                                       |
| `update_tps`                       | 训练的tokens/p/s指标                                         |
| `vllm_tps`                         | 推理的tokens/p/s指标                                         |


**GRPO算法相关指标**

| 指标                                 | 说明                                                         |
|------------------------------------| ------------------------------------------------------------ |
| `grpo/score/mean`                  | 开启奖励模型时的reward均值                                   |
| `grpo/score/max`                   | 奖励模型及规则奖励对同一个样本的reward最大值                 |
| `grpo/score/min `                  | 奖励模型及规则奖励对同一个样本的reward最小值                 |
| `grpo/rewards/mean`                | 规则奖励的reward均值；奖励模型对样本的reward经过归一化后的均值 |
| `grpo/rewards/max`                 | 规则奖励的reward最大值；奖励模型对样本的reward经过归一化后的最大值 |
| `grpo/rewards/min`                 | 规则奖励的reward最小值；奖励模型对样本的reward经过归一化后的最小值 |


**PPO算法相关指标**
| 指标                                 | 说明                                                         |
|------------------------------------| ------------------------------------------------------------ |
| `critic/lr`                        | critic model学习率，优化器当前使用的学习率                               |
| `critic/vf_loss`                   | vf_loss，基于优势函数的策略梯度目标函数值，表示当前策略对提升奖励的学习能力。 |
| `critic/vf_clipfrac`               | PPO中critic model裁剪机制生效的比例，反映了策略更新幅度的稳定性         |
| `critic/vf_clipfrac`               | PPO中critic model裁剪机制生效的比例，反映了策略更新幅度的稳定性         |                                 |
| `critic/score/mean`                | 开启奖励模型时的reward均值                                   |
| `critic/score/max`                 | 奖励模型及规则奖励对同一个样本的reward最大值                 |
| `critic/score/min `                | 奖励模型及规则奖励对同一个样本的reward最小值                 |
| `critic/rewards/mean`              | 规则奖励的reward均值；奖励模型对样本的reward经过归一化后的均值 |
| `critic/rewards/max`               | 规则奖励的reward最大值；奖励模型对样本的reward经过归一化后的最大值 |
| `critic/rewards/min`               | 规则奖励的reward最小值；奖励模型对样本的reward经过归一化后的最小值 |
| `critic/advantages/mean`           | 优势值均值；奖励模型对样本的reward经过归一化后的均值 |
| `critic/advantages/max`            | 优势值最大值；奖励模型对样本的reward经过归一化后的最大值 |
| `critic/advantages/min`            | 优势值最小值；奖励模型对样本的reward经过归一化后的最小值 |
| `critic/returns/mean`              | 所有未来奖励的折扣和均值；奖励模型对样本的reward经过归一化后的均值 |
| `critic/returns/max`               | 所有未来奖励的折扣和最大值；奖励模型对样本的reward经过归一化后的最大值 |
| `critic/returns/min`               | 所有未来奖励的折扣和最小值；奖励模型对样本的reward经过归一化后的最小值 |
| `critic/values/mean`               | 当前状态下未来收益均值；奖励模型对样本的reward经过归一化后的均值 |
| `critic/values/max`                | 当前状态下未来收益均值最大值；奖励模型对样本的reward经过归一化后的最大值 |
| `critic/values/min`                | 当前状态下未来收益均值最小值；奖励模型对样本的reward经过归一化后的最小值 |