# 长序列并行

## 背景介绍
长序列训练需求日益增加，应用场景极为广泛，如翻译场景、多模态场景等等。为解决长序列导致显存溢出的问题，本仓库提供了长序列并行（Context Parallel）的解决方案。

## 方案介绍
### Ulysses
[Ulysses](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-ulysses)是一种用于长序列训练的分布式并行技术，由微软 DeepSpeed 提出。其核心思想是将输入序列在序列维度上切分给不同的计算设备，并通过 All-to-All 通信方式确保每个计算设备能够计算不同注意力头的子集。这种方式可以降低激活显存，解决长序列场景下显存OOM的问题。

具体来说，Ulysses 将各个样本在序列维度上分割给参与的计算设备；然后，在 attention 计算之前，它对已分割的查询(Q)、键(K)和值(V)执行 all-to-all 通信操作，以使每个计算设备接收完整的序列，但仅用于注意力头的非重叠子集，这使得参与的计算设备可以并行计算不同的注意力头；最后，Ulysses 使用另一个 all-to-all 来在注意力头上收集结果，同时重新在序列维度上进行分区。


## 使用介绍

当前仓上的Context Parallel支持ulysses切分，通过如下配置可以使能：
```
actor_config:
   context_parallel_size: 2
   context_parallel_algo: ulysses_cp_algo
```

其中：

`context_parallel_size` 表示CP并行数。如果选用ulysses_cp_algo，需满足条件**模型num_attention_heads%(CP*TP)=0**

`context_parallel_algo` 表示选用的长序列并行方法，当前仅支持**ulysses_cp_algo**；如果不配置此参数，默认取**ulysses_cp_algo**。

