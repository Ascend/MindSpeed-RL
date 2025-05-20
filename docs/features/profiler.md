# 性能调优（Profiler）

## 概述

性能调优模块为强化学习训练流程提供了性能数据采集、分析能力，可帮助用户识别训练过程中的性能瓶颈并进行优化。

## 配置选项

> **注意**：当前 profiler 性能数据采集仅支持共卡模式（integrated）场景。

性能调优工具通过 YAML 配置文件中的 `profiler_config` 部分进行配置：

```yaml
profiler_config:
  integrated:
    profile: false
    mstx: false
    stage: all
    profile_save_path: ./profiler_data
    profile_export_type: text
    profile_step_start: 1
    profile_step_end: 2
    profile_level: level1
    profile_with_memory: false
    profile_record_shapes: false
    profile_with_cpu: true
    profile_with_npu: true
    profile_with_module: false
    profile_analysis: false
    profile_ranks: all
```

### 主要配置参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| profile | 性能分析开关 | true/false，所有性能数据采集均依赖该开关开启 |
| mstx | 轻量化打点采集开关 | true/false，默认值false，启用/关闭轻量化打点采集，需要查看轻量化打点性能数据时需开启 |
| stage | 性能数据采集阶段 | all(采集所有阶段性能数据)、actor_generate(采集actor模型生成阶段性能数据)、actor_compute_log_prob(采集actor模型计算log概率阶段性能数据)、reference_compute_log_prob(采集reference参考模型计算log概率阶段性能数据)、actor_update(采集模型更新阶段性能数据) |
| profile_save_path | 性能数据输出目录 | 任意有效路径，默认为"./profiler_data" |
| profile_export_type | 导出格式 | text、db(性能数据交付件为db格式，可减少约70%磁盘空间)，默认值text |
| profile_step_start | 开启采集数据的步骤 | 任意正整数，默认为1，profile_step_start从1开始 |
| profile_step_end | 结束采集数据的步骤 | 任意正整数，默认为2，实际采集步数为 profile_step_end-profile_step_start，不包含profile_step_end |
| profile_level | 采集级别 | level_none、level0、level1、level2，默认值level0 |
| profile_with_memory | 内存分析开关 | true/false，默认值false，启用/关闭内存分析 |
| profile_record_shapes | 张量形状记录开关 | true/false，默认值false，是否记录张量形状 |
| profile_with_cpu | Host侧性能数据开关 | true/false，默认值true，是否包含Host侧性能数据 |
| profile_with_npu | Device侧性能数据开关 | true/false，默认值true，是否包含NPU侧性能数据 |
| profile_with_module | Python调用栈信息开关 | true/false，默认值false，是否包含Python侧调用栈信息 |
| profile_analysis | 自动解析开关 | true/false，默认值false，是否在采集后自动解析数据 |
| profile_ranks | 采集数据的卡号 | all表示所有rank, 默认值all，可以通过列表指定，如[0, 1] |

## 性能数据采集

GRPO算法涉及多个worker模块交互，包含训练、推理等流程。在完整训练过程中，性能数据量可能较大，以下是几种优化分析效率的方法：

### 1. 减小训练参数规模

- **关键配置**:
  - 减少单条response最大生成token数量: `sampling_config.max_tokens <= 128`
  - 降低全局批次大小: `megatron_training.global_batch_size <= 8`
  - 减小模型层数
  
- **适用场景**: 调试内存问题或与批次大小/序列长度/模型层数无关的性能问题

### 2. 使用轻量化采集模式

- **关键配置**:
  ```yaml
  profile: true
  mstx: true
  profile_level: level_none
  profile_with_cpu: false
  profile_with_npu: true
  ```
  
- **适用场景**: 轻量级采集包含自定义打点和所有通信算子的内置打点，MindSpeed-RL已集成所有worker的关键计算函数、dispatch_transfer_dock_data、resharding等关键函数的自定义打点。如需查看某代码片段在timeline中的执行耗时，可通过以下两种方式在MindSpeed-RL中添加自定义打点：

  ```python
  # 方式一：使用装饰器装饰函数
  from mindspeed_rl.utils.utils import mstx_timer_decorator
  
  @mstx_timer_decorator
  def your_function():
      # 函数代码
      pass
  
  # 方式二：框住代码片段
  import torch_npu

  id = torch_npu.npu.mstx.range_start("your_tag_name")
  result = complex_operation()  # 需要记录打点时间片的代码
  torch_npu.npu.mstx.range_end(id)
  ```
  轻量级采集数据量较小，无需调整训练参数，适用于检查模块执行时间是否异常以及多机间的通信瓶颈。

### 3. 按训练阶段分段采集

- **关键配置**: 
  - 对于integrated worker，`stage`参数可选值：
    - all(采集所有阶段性能数据) 
    - actor_generate(采集actor模型生成阶段性能数据)
    - actor_compute_log_prob(采集actor模型计算log概率阶段性能数据)
    - reference_compute_log_prob(采集reference参考模型计算log概率阶段性能数据)
    - actor_update(采集模型更新阶段性能数据)

  - 采集数据时同样建议降低全局批次大小 `megatron_training.global_batch_size <= 4`，避免性能数据过大

- **适用场景**: 需要查看训练某一特定阶段的详细计算、通信profiling数据

## 性能数据解析

性能数据采集后需要进行解析才能查看，可通过以下两种方式：

### 1. 离线解析

适用于大规模集群，使用如下脚本对性能数据进行解析。

```python
import torch_npu
# 在性能数据采集完成后，可对所有性能数据执行离线解析（"./profiler_data"可包含多份性能数据，解析可并行进行）
torch_npu.profiler.profiler.analyse(profiler_path="./profiler_data")
```

### 2. 在线解析

设置`profile_analysis=true`在采集后自动解析。注意：当性能数据量较大时，解析时间可能较长。

## 结果可视化

解析后的性能数据保存在`profile_save_path`指定目录中，可通过以下工具进行可视化：

- **MindStudio Insight**：提供丰富的性能分析视图，包括时间线视图、算子分析、通信分析。详细使用指南可参考[MindStudio文档](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/index/index.html)

## 最佳实践

1. 从轻量级分析（mstx 模式）开始识别训练的瓶颈模块
2. 使用特定阶段分析聚焦于问题区域
3. 分析特定rank而非所有rank，以减少数据量

> **注意**: 请更新至最新8.1.RC1 CANN包后使用轻量化数据解析功能。
