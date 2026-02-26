# partial rollout
 
## 简介
Partial rollout 核心思想是通过对长序列 response 推理样本做提前中断，并在下次推理过程中对当前样本进行续推，从而避免单一的长尾样本对推理过程造成资源浪费。通过该能力，我们可以降低长序列推理场景下的长尾样本对端到端性能的影响。

## 使用方法
```yaml
rl_config:
  partial_rollout_max_split: N # 设置N>1即可在N轮内完全推理完成最长序列
```
## 技术方案
### 同步推理引擎方案

核心理念：断点续推+跨迭代长尾调度避免推理资源闲置
同步引擎：数据按批处理，同时进入推理引擎、批次内所有数据完成推理后同时返回结果
关键技术点：
1.	长序列推理截断机制：根据最大推理长度和次数设置推理截断点，将截断样本放入TransferDock，当满足≥GBS个prompt已完成全部推理，则进入后续计算任务，否则则从TransferDock中取数据再次推理，达成高资源利用率。
2.	基于优先级的混合数据重排和采样技术：在下一轮推理时，优先取出被截断样本进行推理，避免影响效果和收敛性。

![img.png](../../docs/zh/figures/partial_rollout/sync.png)
 
图1 同步引擎方案示意图

![img_1.png](../../docs/zh/figures/partial_rollout/sync_1.png)
 
图2 同步引擎流程图

### 异步推理引擎方案

核心理念：断点续推+跨迭代长尾调度避免推理资源闲置
异步引擎：数据按批次进入推理引擎，可异步按样本粒度返回结果
关键技术点：
1.	实时长序列推理截断机制：实现与推理引擎交互，动态确定长尾序列被截断长度，当满足≥GBS个prompt已完成全部推理，则中断推理过程，将截断样本放入TransferDock，避免长尾序列推理拖慢整体推理时间、造成资源空置。
2.	基于优先级的混合数据重排和采样技术：在下一轮推理时，优先取出被截断样本并混合新样本进行推理。
3.	收敛性和稳定性保证：实现样本在规定的迭代轮数内完成推理。
 ![img_2.png](../../docs/zh/figures/partial_rollout/async.png)
图3 异步引擎方案示意图

![img_3.png](../../docs/zh/figures/partial_rollout/async_1.png)
 
图4 异步引擎流程图

## 验证情况
![img_4.png](../../docs/zh/figures/partial_rollout/sync_partial_compare_result.png)
图5 同步引擎验证结果

### 注意事项
目前此特性暂不支持和VPP特性同时启用。