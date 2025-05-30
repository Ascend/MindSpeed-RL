# 后训练数据系统

## 背景介绍

在LLM后训练过程中，各个计算任务之间存在较多数据依赖。为此，提供一个数据管理系统用于管理后训练中的数据流程。

## 方案概述
本方案在LLM后训练系统中连接了推理框架与训练框架，扮演了**转运港口**的角色：

1. 数据生产者将生成数据写入到数据系统中；
2. 数据系统将数据存储至预先分配的缓存区，并更新数据状态；
3. 数据消费者向数据系统发送请求，若存在足量数据，则将对应数据组织为Batch，返回给数据消费者。

在该架构中，推理框架、训练框架中的各个实例数据均存放至数据调度模块，由其统一调度，从而避免了各个实例之间的绑定，提高了整体计算资源的利用率。


### 高并发设计

在设计过程中，对高并发场景进行了细致考虑。具体地，RL后训练过程涉及以下两种不同的并发场景：

* RL**角色间**同时读写：如generate_sequence, compute_log_prob, update等，这些RL角色会同时向数据调度模块进行读写数据请求
* RL**角色内**同时读写：对非训练过程的RL角色，其各路DP之间存在异步数据读写，导致数据争抢

针对以上并发场景，数据调度模块实现了尽可能的无阻塞数据读写。具体地，代码实现将一次访问拆解为**数据采样**与**数据读写**两个过程。

数据采样过程针对用户在外侧不指定Index的场景（例如，各类读请求），此时需依赖派生类扫描并采样出可被读写的Index。若用户直接指定了Index（例如，各类写请求需先读出数据和Index，再按Index写回），则直接进入数据读写过程。在数据读写过程中，依据派生类采样出的Index或用户指定的Index进行读写，完全消除了数据阻塞。


|并发场景|使用方式|数据采样过程|数据读写过程|
|:----:|:----:|:----:|:----:|
|RL角色间|给定Index访问|无阻塞|无阻塞|
|RL角色间|随机访问|无阻塞|无阻塞|
|RL角色内|给定Index访问|无阻塞|无阻塞|
|RL角色内|随机访问|有阻塞|无阻塞|




## 开发原理
### 初始化
数据系统在Trainer类中进行初始化，其引用作为参数传递给后训练过程中的各个Worker。该初始化过程保证了各个Worker间基于相同的数据模块实例，提供了集中式的数据管理能力。

![数据系统初始化过程](../../sources/images/data_module/td_init.png)


### Worker读写逻辑

在Worker的计算任务中，首先需指定所需要读写的列名与每次读取的数据量。之后每个循环，都将依照`self.all_consumed()`状态确定是否要继续读取数据，若本GBS仍有数据未处理完，则调用`dispatch_transfer_dock_data()`函数从数据系统中读取数据，并在完成计算任务后通过`collect_transfer_dock_data()`函数将对应结果写回数据系统。

上述交互逻辑简化了分布式计算中各个进程的数据读写操作，每个计算任务均向单一的数据源进行读写请求，避免了显式定义不同计算任务之间的数据链路，简化了编程流程。

![数据系统交互逻辑](../../sources/images/data_module/td_interaction_logic.png)

## 参数配置

本方案涉及后训练中的数据流转功能，在与数据读写相关的参数配置上提供以下参考。

在yaml配置文件中，关于各类batch_size的说明如下：

| 参数名                            | 参数位置                       | 说明                                                                                                                           |
|:-------------------------------|:---------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| global_batch_size              | megatron_training          | 每个iteration所处理的Prompt数量；对于GRPO算法，在代码中将自动与n_samples_per_prompt相乘                                                              |
| mini_batch_size                | rl_config                  | 每次更新actor update的Prompt数量；对于GRPO算法，在代码中将自动与n_samples_per_prompt相乘。该值需小于等于global_batch_size，等于时即为on-policy算法，小于时为off-policy算法 |
| actor_forward_micro_batch_size | rl_config | actor每次前向的(Prompt, Response)对数量；对于GRPO算法，设置时需指定考虑n_samples_per_prompt之后的值                                                    |
| ref_forward_micro_batch_size   | rl_config | ref每次前向的(Prompt, Response)对数量；对于GRPO算法，设置时需指定考虑n_samples_per_prompt之后的值                                                      |
| micro_batch_size             | actor_config               | 对于actor update任务每次前向+反向的(Prompt, Response)对数量；对于GRPO算法，设置时需指定考虑n_samples_per_prompt之后的值                                            |

开发者还可通过调整以下可选参数，控制每次数据读写的粒度实现性能极致调优需求：

|参数名| 参数位置                    | 说明                                                                                                                                                                                                                             |
|:----|:------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|actor_rollout_dispatch_size| config_cls/rl_config.py | 【可选参数】actor rollout的每路DP每次从TD中读出的(Prompt, Response)对的数据量；默认设置为global_batch_size * n_sample_per_prompt / actor_rollout_dp_size                                                                                                  |
|actor_logprob_dispatch_size| config_cls/rl_config.py | 【可选参数】actor logprob的每路DP每次从TD中读出的(Prompt, Response)对的数据量；默认设置为global_batch_size * n_sample_per_prompt / actor_logprob_dp_size                                                                                                  |
|actor_update_dispatch_size| config_cls/rl_config.py | 【可选参数】actor update的每路DP每次从TD中读出的(Prompt, Response)对的数据量；默认设置为global_batch_size * n_sample_per_prompt / actor_update_dp_size                                                                                                    |
|ref_dispatch_size| config_cls/rl_config.py | 【可选参数】ref logprob的每路DP每次从TD中读出的(Prompt, Response)对的数据量；默认设置为global_batch_size * n_sample_per_prompt / ref_logprob_dp_size                                                                                                      |
|reward_dispatch_size| config_cls/rl_config.py | 【可选参数】reward每路DP(若有)每次从TD中读出的(Prompt, Response)对的数据量；对于Reward Model，默认设置为global_batch_size * n_sample_per_prompt / reward_dp_size；对于规则奖励默认设置为global_batch_size * n_sample_per_prompt；手动设置时对于GRPO算法需保证为n_samples_per_prompt的整数倍 |
|adv_dispatch_size| config_cls/rl_config.py | 【可选参数】advantage每次从TD中读出的(Prompt, Response)对的数据量；默认设置为global_batch_size * n_sample_per_prompt                                                                                                                                   |



## 未来演进

当前数据调度模块采用单节点设计，各路DP均会向单一节点发送读写请求，在千卡以上大规模训练时可能成为瓶颈。未来将进一步支持分布式存储，将控制平面与数据平面分离，管理节点维护数据状态，实际数据读写过程将分布在各个存储节点中，从而缓解网络带宽瓶颈与IO瓶颈。