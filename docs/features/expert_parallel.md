# 推理大EP（Expert Parallel）

## 背景介绍

MoE 类模型支持 ​**专家并行（Expert Parallel, EP）**​，将不同专家分别部署在不同的设备上，实现专家维度的并行计算。

当前提供两种 EP 并行模式：

* ​**`ep_level: 1`**​：基于 **AllGather** 通信的专家并行；
* ​**`ep_level: 2`**​：基于 **AllToAll 与通算融合** 的专家并行。

---

## 适用场景

* 多个专家分布在不同设备，每个 token 只激活局部专家；
* 提升 MoE 混合专家模型推理的​**并行深度**​；
* 避免因单卡显存不足而无法容纳全部专家参数。

---

## 预期效果

* 在 MoE 推理中，可提升吞吐 ​**1.5–4 倍**​；
* 专家分布至更多设备，缓解单卡显存压力。

---

## 配置方法

1. **开启专家并行**
   在 `general_config` 中设置：
   **yaml**
   
   ```
   enable_expert_parallel: true
   infer_expert_parallel_size: n  # n 为 EP 并行度
   ```
2. **同步修改运行环境**
   在 `runtime_env.yaml` 中，将 `VLLM_DP_SIZE` 设为与 `infer_expert_parallel_size` 相同的值 `n`，以保持配置一致。
3. **配置模型并行参数**
   在 `actor_config` 中设置：
   **yaml**
   
   ```
   expert_model_parallel_size: n  # 与上述 n 保持一致