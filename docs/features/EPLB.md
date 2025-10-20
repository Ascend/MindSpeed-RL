
# EPLB 背景介绍

在 Mixture-of-Experts (MoE) 架构中，不同专家所接收的输入（tokens）数量存在显著差异，这直接导致了专家计算负载的不均衡。负载差异会导致部分 GPU 的算力被过度占用，而另一些 GPU 的算力则处于闲置状态。为了实现负载均衡，提出了EPLB（Expert Parallelism Load Balancer），通过调度专家副本的负载分配，提升 MoE 模型推理与训练效率。


---
<br/>

# EPLB 使用流程

 **Step 1：MoE Token Collect**  
 收集 token 到 expert 的分配信息

 **Step 2：EPLB Map Generate**  
 基于分配信息生成冗余专家映射表

 **Step 3：EPLB Usage**  
 根据映射表重新分发专家权重

---
<br/>

## MoE Token Collect 使用说明

本功能用于在 **推理/训练过程** 中采集 Mixture-of-Experts (MoE) 的路由信息，记录每一层的 **token 到 expert 的分配情况**，方便后续分析负载均衡和专家使用率。

---

### 1. 功能描述

* 在 MoE 前向计算时，系统会拦截 `AscendUnquantizedFusedMoEMethod.apply` 调用。
* 自动统计 `topk_ids`（即每个 token 被路由到的 expert ID），并按 layer 累计。
* 所有数据会收集到 **rank0** 所在的节点上，统一写入一个 JSON 文件。
* JSON 文件格式如下：

```json
{
  "0": {
    "0": 123,
    "1": 256,
    "2": 89,
    ...
  },
  "1": {
    "0": 88,
    "1": 190,
    ...
  }
}
```

* 其中：

  * 外层 key 表示 MoE 层号
  * 内层 key 表示 expert ID
  * 值为该 expert 被分配的 token 数

---

### 2. 配置方法

在 `yaml` 配置文件的 **`generate` 节点** 下增加以下参数：

```yaml
generate:
  token_collects: true                # 是否开启 token 收集
  token_save_path: "/path/to/save"    # JSON 文件保存目录
```

参数说明：

* `token_collects`:

  * `true` → 开启 token 收集
  * `false` → 默认关闭，避免不必要的 I/O 开销
* `token_save_path`:

  * 必填，指定 JSON 文件存储目录
  * 最终只会生成 **一个汇总文件**：

    ```
    token_collects_all.json
    ```

---

### 3. 运行效果

1. 运行推理或训练后，各 rank 会先本地统计数据，再通过分布式通信汇总到 rank0。

2. rank0 会在 `token_save_path` 下生成一个 JSON 文件：

   ```bash
   $ ls /mnt/data2/MindSpeed-RL
   token_collects_all.json
   ```

3. 文件中保存了 **所有 rank 的统计结果**，包含每层每个 expert 的 token 数。

---

### 4. 注意事项

* **性能开销**：开启采集后会增加统计与 JSON I/O，建议在 profiling 或 debug 时使用。
* **多机环境**：数据会自动聚合，用户只需查看 rank0 生成的 `token_collects_all.json` 文件。
* **文件写入安全**：采用临时文件 + 原子替换，避免因进程异常退出导致 JSON 文件损坏。
* **配置优先级**：`yaml` 中的配置会覆盖代码默认值。
---

<br/>

## Eplb_map Generate 使用说明

本功能基于每一层的 **token 到 expert 的分配情况**，通过DeepSeek开源的**EPLB策略**，生成 **冗余专家映射表**，实现更合理的专家分布。

---

### 1. 功能描述

* 使用分层负载均衡策略来分配专家。
* 首先将专家均匀分配到各个节点，确保不同节点的负载保持平衡。接着，在每个节点内部复制专家。最后，将这些复制后的专家分配到各个 npu 上，以确保不同 npu 之间的负载均衡。
* 去重处理：每个 NPU 内部不重复分配专家。
* 该功能可以得到冗余专家与npu之间的映射情况，生成一个JSON文件。
* JSON文件格式如下：

```json
{
    "moe_layer_count": 1,
    "layer_list": [
        {
            "layer_id": 0,
            "device_count": 2,
            "device_list": [
                {
                    "device_id": 0,
                    "device_expert": [
                        0,
                        1,
                        2,
                        3,
                        5
                    ]
                },
                {
                    "device_id": 1,
                    "device_expert": [
                        5,
                        6,
                        7,
                        8,
                        1
                    ]
                }
            ]
        }
    ]
}
```

* 其中：

  * moe_layer_count 表示总 MOE 层数
  * layer_list 表示每层的专家映射表
  * layer_id 表示 MOE 层号
  * device_count 表示 npu 总数
  * device_list 表示该 MOE 层中每个 npu 的专家映射表
  * device_id 表示 npu rank
  * device_expert 表示该 npu 上分配的 expert ID

---

### 2. 配置方法

在 examples/eplb/eplb.sh 文件中配置以下参数：
```
python  mindspeed_rl/workers/eplb/eplb_generate_map_ds.py \
--json_folder ./json_file \
--num_replicas 40  \
--num_groups  4 \
--num_nodes  1 \
--num_gpus 8 \
--output_path mindspeed_rl/workers/eplb/expert_map.json \
```

* 其中：

  * json_folder 表示 `token_collects_all.json` 所在目录，与 `Step1 MOE Token Collect` 中的 `token_save_path` 一致
  * num_replicas 表示冗余专家总数（含原始专家）
  * num_groups 表示负载均衡策略中的专家分组数, 要求能被num_gpus整除
  * num_nodes 表示机器数
  * num_gpus 表示总npu数
  * output_path 指定生成的 JSON 文件路径与命名

---

### 3. 运行效果

1. 运行方式
    ```
    bash examples/eplb/eplb.sh
    ```
2. 生成JSON文件： `output_path` 
3. 文件中保存了 **每个层上** expert在**每个rank上**的映射情况。

---

### 4. 注意事项

* **使用限制**：推理目前只支持单实例; 每个npu上的专家数需要保持一致。
---

<br/>

## EPLB Usage 使用说明

本功能基于生成的Eplb_map映射表重新分发专家权重，实现推理阶段的负载均衡。

---

### 1. 功能描述

* 基于推理的分布式并行策略生成初始化的专家映射表，代表推理侧的初始权重存放情况。
* Step2生成的Eplb_map作为权重分配的目标映射表，将权重从初始状态重新收发，达到目标状态。
* 专家权重的收发采用p2p实现。

---

### 2. 配置方法

在 `yaml` 配置文件的 **`generate` 节点** 下关闭MoE Token Collect的参数：

```yaml
generate:
  #token_collects: true                # 是否开启 token 收集
  #token_save_path: "/path/to/save"    # JSON 文件保存目录
```

在 `grpo_deepseek_r1_671b_A3_eplb.yaml` 配置文件的 **`generate` 节点** 下增加专家映射表路径：

```yaml
expert_map_path: /file/to/save
```

参数说明：

* `expert_map_path`:专家映射表文件，与 Eplb_map Generate 中 eplb.sh 的 `output_path` 参数一致

  * 必填，指定 冗余专家映射表文件

运行脚本：
```bash
bash examples/eplb/grpo_trainer_deepseek_r1_671b_eplb.sh
```
---

### 3. 注意事项

* **性能开销**：涉及 `P2P` 通信，建议在资源充足时使用。
* **配置优先级**：`yaml` 配置优先于代码默认值
* **使用限制**：推理目前只支持单实例。

---