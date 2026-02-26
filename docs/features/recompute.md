# Megatron 重计算

## 背景与挑战

在大模型的训练流程中，传统的实践要求存储前向传播阶段产生的激活值，以供后续反向传播过程中的梯度计算使用。这一需求导致了激活值保存数量随模型深度线性增长的现象，显著加剧了对硬件内存资源的压力。

## 解决思路

为应对上述挑战，提出了重计算策略。具体而言，在前向传播与损失函数计算阶段，即时释放不再需要的激活值内存空间，仅在反向传播时根据需要重新计算激活值。此方法通过有效缩短激活值的生命周期，显著减轻了内存负担，提升了整体的资源利用效率。

## 使用场景

在显存不够的情况下可以开启重计算特性，且分为以下两种方式：

* 选择性重计算（推荐）：专注于对Transformer架构内的core_attention组件进行重计算。该策略保留了那些占用较小内存空间但重计算成本较高的激活值，同时，对占用较大内存但重计算成本相对较低的激活值执行激活重计算。此方法在保证模型性能的同时，实现了内存使用的高效管理。
* 完全重计算：适用于内存资源极为受限的极端环境。在这种模式下，除了保存输入数据外，所有激活值均在需要时重新计算，最大限度地减少了对内存的依赖。

## 使用方法

* 选择性重计算： `recompute_activations   #开启选择性重计算`。
* 完全重计算： `recompute_granularity full    #开启完全重计算`
* 重计算方案，  `recompute_method`可配置参数uniform或block，任选其一：

`recompute_method uniform`：将Transformer层均匀划分组（每组大小`recompute_num_layers`），按组存储输入和激活值。
`recompute-method block`：将前`recompute_num_layers`个Transformer层重计算，剩余层不进行重计算。

#### 说明：

* 同时配置`recompute_activations` 、`recompute_granularity full`时，生效选择性重计算。
* 当脚本配置了`recompute_method block`、`recompute_granularity full`、`num_layers_per_virtual_pipeline_stage N`参数时，用户可以通过`recompute_num_layers N`参数来配置每个vpp stage做多少层重计算，参数`enable_recompute_layers_per_pp_rank`可用于修改此情况下`recompute_num_layers N`参数的语义，新的语义表示无视vpp，按每个pp stage来配置重计算层数。
* 注意：在legacy分支下，开启`use_flash_attn`将无法使用选择性重计算。

## 使用影响

采用重计算策略，能够带来以下使用效果：

* 通过避免长时间保留大量中间计算结果，大幅降低了对内存的需求。
* 重计算激活值会带来额外的计算成本，降低性能。因此重计算的使用和配置需要综合考虑内存占用情况。