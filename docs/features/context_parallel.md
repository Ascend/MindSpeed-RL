# 长序列并行

## 背景介绍
长序列训练需求日益增加，应用场景极为广泛，如翻译场景、多模态场景等等。为解决长序列导致显存溢出的问题，本仓库提供了长序列并行（Context Parallel）的解决方案。

## 方案介绍
### Ulysses
[Ulysses](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-ulysses)是一种用于长序列训练的分布式并行技术，由微软 DeepSpeed 提出。其核心思想是将输入序列在序列维度上切分给不同的计算设备，并通过 All-to-All 通信方式确保每个计算设备能够计算不同注意力头的子集。这种方式可以降低激活显存，解决长序列场景下显存OOM的问题。

具体来说，Ulysses 将各个样本在序列维度上分割给参与的计算设备；然后，在 attention 计算之前，它对已分割的查询(Q)、键(K)和值(V)执行 all-to-all 通信操作，以使每个计算设备接收完整的序列，但仅用于注意力头的非重叠子集，这使得参与的计算设备可以并行计算不同的注意力头；最后，Ulysses 使用另一个 all-to-all 来在注意力头上收集结果，同时重新在序列维度上进行分区。

### Ring Attention
[Ring Attention](https://arxiv.org/pdf/2310.01889)借鉴了分块Softmax原理，在不需要获取整个序列的完整矩阵情况下进行分块attention计算，提出以分块方式执行自注意力和前馈网络计算，跨多个设备分布序列维度。具体地，该方法在进程之间构建注意力计算块的环状通信结构（Ring），每个进程具有一个切分后的本地QKV块；在计算完本地的attention后，通过向后发送和向前获取KV块，遍历进程设备环，以逐块的方式进行注意力和前馈网络计算；同时，本地的attention计算和KV块的通信理想情况下可以互相掩盖，从而消除了额外引入的通信开销。
相较于ulysses方案，此方案不需要模型的 num_attention_heads 被（cp_size*tp_size）整除，相对来说具有更高的灵活性。
如果想要使得计算和通信可以互相掩盖，理论上需要确保每个计算块分到的序列长度c≥F/B。其中F是每个device的FLOPS，B是每个device间的带宽。具体推导过程参见原文。在实践中，需要确保每个计算块分到的序列长度足够大，才能较好掩盖。

## 使用介绍
### Ulysses
当前仓上的Context Parallel支持ulysses切分，通过如下配置可以使能：
```yaml
actor_config:
   context_parallel_size: 2
   context_parallel_algo: ulysses_cp_algo

# 与remove_padding特性一起使用
rl_config:
  use_remove_padding: true

megatron_training:
   reset_position_ids: true
   variable_seq_lengths: true
```

对于直接偏好对齐（DPO）算法，通过如下配置可以使能：

```yaml
# 填写在megatron_training
megatron_training:
   variable_seq_lengths: true
   context_parallel_size: 2
   context_parallel_algo: ulysses_cp_algo
```

`context_parallel_size` 表示CP并行数。如果选用ulysses_cp_algo，需满足条件**模型num_attention_heads%(CP*TP)=0**

`context_parallel_algo` 表示选用的长序列并行方法，如果不配置此参数，默认取**ulysses_cp_algo**

特别的，此特性与**remove_padding**一起使用时，配置说明如下：
直接叠加使能remove_padding的配置即可：将use_remove_padding、reset_position_ids、variable_seq_lengths都设置为true。

### Ring Attention
```yaml
actor_config:
   context_parallel_size: 2
   context_parallel_algo: megatron_cp_algo
   cp_attention_mask_type: causal

# 与remove_padding特性一起使用
rl_config:
  use_remove_padding: true

megatron_training:
   reset_position_ids: true
   variable_seq_lengths: true
   reset_attention_mask: true
```

对于直接偏好对齐（DPO）算法，通过如下配置可以使能：

```yaml
# 填写在megatron_training
megatron_training:
   variable_seq_lengths: true
   context_parallel_size: 2
   context_parallel_algo: megatron_cp_algo
   cp_attention_mask_type: causal
```


其中：

`context_parallel_size` 表示CP并行数，ring attention 的 cp_size无模型num_attention_heads的限制

`context_parallel_algo` 表示选用的长序列并行方法，使能ring attention方案需要设置为**megatron_cp_algo**

`cp_attention_mask_type` 表示选用的attention_mask类型，默认配置**causal**，当前也只支持该类型

特别的，此特性与**remove_padding**一起使用时，配置说明如下：

除了需要将use_remove_padding、reset_position_ids、variable_seq_lengths都设置为true之外（使能remove_padding），还需要将reset_attention_mask设置为true; 否则这些配置都不要配，默认**false**。
