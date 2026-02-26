# MindSpeed-RL 特性文档

本文档汇总了 MindSpeed-RL 支持的核心特性，涵盖训练加速、推理加速、强化学习调度及基础功能四个方面。

## 1. 训练加速特性
本部分包含在模型训练（Actor/Critic Update）阶段，针对显存优化、通信优化及计算加速的特性。

| 特性名称 | 简介 | 发布状态 | 文档链接 |
| :--- | :--- | :--- |:--- |
| **optimizer/grad offload** | 支持将优化器状态和梯度卸载至 CPU，降低训练显存占用。| preview | [doc](./offload.md) |
| **VPP/DPP** | 支持交错式流水线并行（Interleaved Pipeline Parallelism），减少流水线气泡。| release |  [doc](./vpp.md)  |
| **Swap optimizer** | 支持优化器状态在设备与主机内存间交换，进一步优化显存峰值。| preview | [doc](./swap_optimizer.md) |
| **CP** | Context Parallel，针对长序列训练的上下文并行优化方案。| preview | [doc](./context_parallel.md) |
| **BF16参数副本复用** | 复用 BF16 参数副本，减少内存冗余占用。| preview | [doc](./reuse_fp32_param.md) |
| **重计算** | 通用重计算策略，通过以计算换显存的方式支持更大模型训练。| preview |  [doc](./recompute.md)  |
| **激活函数重计算** | 针对 GeLU/SwiGLU 等激活函数的特定重计算优化。| preview | [doc](./activation_function_recompute.md) |
| **Norm 重计算** | 针对 LayerNorm/RMSNorm 的重计算优化。| preview |  [doc](./norm_recompute.md)|

## 2. 推理加速特性
本部分包含在生成（Rollout/Make Experience）阶段，基于 vLLM 或其他推理后端的性能优化特性。

| 特性名称 | 简介 | 发布状态 | 文档链接 |
| :--- | :--- | :--- | :--- |
| **推理图模式（torchair/Aclgraph）** | 利用 torchair/Aclgraph 将推理过程编译为静态图执行，提升推理性能。| preview | [doc](./acl_graph.md) |
| **cudagraph_mode: FULL_DECODE_ONLY** | 在解码阶段启用 CUDA Graph（全图模式），减少 Kernel 启动开销。| preview | / |
| **vLLM Prefix cache** | 复用 vLLM 的 Prefix KV Cache，加速多轮对话或共享前缀场景的推理。| preview | [doc](./vLLM_prefix_cache.md) |
| **chunked prefill** | 支持分块预填充（Chunked Prefill），优化长 Prompt 下的首字延迟和吞吐平衡。| preview |  [doc](./chunked_prefill.md)  |
| **dynamic batch size** | 支持动态 Batch Size 调度，提高推理时的计算资源利用率。| preview | [doc](./remove_padding.md) |
| **remove padding** | 移除输入序列中的 Padding，减少无效计算。| preview | [doc](./remove_padding.md) |
| **swap attention** | 支持 Attention KV Cache 的 Swap 机制，处理超长上下文显存不足的情况。| preview | [doc](./swap_attention.md) |
| **推理大EP** | 支持推理阶段的大规模专家并行（Expert Parallel），适配 MoE 模型推理。| preview | [doc](./expert_parallel.md) |
|
## 3. 强化学习框架调度特性
本部分涉及强化学习算法（如 PPO/GRPO）流程中的数据流转、任务编排及特定策略支持。

| 特性名称 | 简介 | 发布状态 | 文档链接 |
| :--- | :--- | :--- | :--- |
| **Partial rollout** | 支持部分采样（Partial Rollout），允许在生成部分数据后即开始训练，优化流水线效率。| preview | [doc](./partial_rollout.md) |
| **Task Queue** | 内部任务队列管理，用于异步调度推理与训练任务，提升系统并发能力。| preview | [doc](./task_queue.md) |

## 4. 基础功能特性
MindSpeed-RL 支持的基础模型并行策略、环境配置及调试功能。

| 特性名称 | 简介 | 发布状态 | 文档链接 |
| :--- | :--- | :--- | :--- |
| **并行策略** | 包含基础并行策略集：**TP** (Tensor Parallel), **PP** (Pipeline Parallel), **DP** (Data Parallel), **SP** (Sequence Parallel), **DDP**, **CP（ring）**, **swap-optimizer**。| release | / |
| **moe-hierarchical-alltoallv** | 针对 MoE 模型的分层 AllToAllV 通信优化，提升大规模集群下的通信效率。| preview | / |
| **确定性计算** | 支持开启确定性计算模式，确保训练结果的可复现性，便于对齐与调试。| preview | [doc](./deterministic_computation.md) |
| **AIV环境变量** | 提供 AIV (Ascend Inference Virtualization) 相关环境变量配置说明，用于调优底层推理行为。| preview | / |
