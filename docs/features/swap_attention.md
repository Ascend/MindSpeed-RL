# swap-attention

## 背景介绍

随着大模型（如Transformer、GPT等）规模的不断增大，训练过程中面临的内存瓶颈和计算效率问题愈发突出。大模型的参数量庞大，尤其是在深度学习训练过程中，激活值需要占用大量的内存空间，而传统的内存管理方法往往无法满足这种需求。为了避免内存溢出或设备资源不足，开发者通常会选择“重计算”策略，即在反向传播时重新计算部分前向传播的激活值，以节省内存。

然而，重计算虽然能显著减少内存占用，但却需要消耗更多的计算资源和时间，从而增加了训练过程的延迟，导致效率低下。随着模型规模的扩大，单纯依赖传统内存优化手段已经难以满足高效训练的要求，尤其在多节点分布式训练和大规模并行计算环境下，内存和计算之间的瓶颈更加突出。

## 解决方案

针对这一挑战，提出了swap-attention功能，旨在梯度反向传播的同时，从CPU内存中动态预取需要的激活值，通过优化内存使用和计算过程来减少重计算，并充分利用H2D（Host to Device）高带宽的数据传输优势，有效缓解内存瓶颈，提升每秒浮点运算数（MFU），加速大模型的训练。


![alt text](../../docs/zh/figures/swap_attention/swap_attention0.png)

## 使用场景

### a. 开启重计算，优化性能：

在需要开启全重计算的场景下，可以通过开启`swap_attention`和`recompute_num_layers:[int]`替换全重计算，以达到提升性能的目的。

开启后，将对每一层的attention层的激活值进行预取，同时，对前[int]层的全连接层进行重计算。

![alt text](../../docs/zh/figures/swap_attention/swap_attention1.png)

### b. 仅开启预取功能, 节省内存：

对于不需要重计算的场景，只开启`swap_attention`，可以在几乎不损耗性能的情况下，节省内存，以支持更大的模型的配置。

开启后，将对每一层的attention层的激活值进行预取，提高计算效率。

![alt text](../../docs/zh/figures/swap_attention/swap_attention2.png)

## 使用方法

1. （前提）开启flash attention融合算子: `use_flash_attn = True`。
2. 开启swap_attention功能：`swap_attention: True`。

可选参数:

1. `swap_modules`：参数类型为string，默认值为"input_norm,self_attention,post_attention_norm"，可根据模型自行配置module，在mcore场景下默认仅预取self_attention module。
2. `recompute_num_layers`: 参数类型为int, 默认值为None，即不开启重计算。可根据场景需要自行配置重计算的层数。

## 注意事项：

1. `recompute_num_layers [int]`中的[int]层数指的是每一个pp stage的层数。[int]的取值应该小于等于num_layers/pipeline_model_parallel_size.
2. 若出现性能波动，可能是跨NUMA内存访问引起，可尝试通过进程绑核缓解 `export CPU_AFFINITY_CONF=1,lazy_bind:0`
3. `swap_attention`暂不兼容LoRA微调。
