# 精度分析（Msprobe）

## 概述

msprobe模块为强化学习训练流程提供了配置采集、关键过程数据采集比对、模型层输入输出数据采集比对的能力，帮助精度问题分析和调优。

## 特性使用

> **注意**：当前 msprobe 数据采集仅支持共卡模式（integrated）场景。

### 前置条件

安装msprobe三方库，[安装指南](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/01.installation.md)。

### 配置选项

精度分析工具通过 YAML 配置文件中的 `msprobe_config` 部分进行配置：

```yaml
msprobe_config:
  msprobe: false
  dump_path: "./msprobe_dump"
  key_data_dump: false
  configurations_dump: false
  actor_train_dump: false
  reference_dump: false
  step_start: 0
  step_end: 0
```

### 配置参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| msprobe | 是否使能msprobe | true/false，开启后，下列的采集项才会生效 |
| dump_path | 存盘路径 | str，默认值"./msprobe_dump" |
| key_data_dump | 关键过程数据采集 | true/false，默认false，是否采集关键过程数据，包括prompt、response、ref_log_prob、advantage、log_prob、kl_loss、loss的统计量信息（最大值、最小值、均值、L2norm值）和真实数据 |
| configurations_dump | 训练配置采集 | true/false，默认false，是否采集训练配置 |
| actor_train_dump | actor的训练阶段模型层输入输出 | true/false，默认false，是否采集actor_compute_log_prob、actor_update阶段的模型层数据 |
| reference_dump | reference的模型层输入输出 | true/false，默认false，是否采集reference的模型层数据 |
| step_start | 采集开始步数 | int，默认0，只对actor_train_dump、reference_dump生效 |
| step_end | 采集结束步数 | int，默认0，只对actor_train_dump、reference_dump生效。如果只想采某一步的数据，设置为跟step_start一样 |

### 落盘数据说明

```txt
msprobe_dump/
├── actor_compute_log_prob/  # actor_compute_log_prob阶段的模型层数据
├── actor_update/  # actor_update阶段的模型层数据
├── reference_compute_log_prob/  # reference的模型层数据
├── data/  # 训练过程关键数据
│   └── advantages/  
│   └── kl_loss/  
│   └── log_prob/  
│   └── loss/  
│   └── prompts/  
│   └── ref_log_prob/  
│   └── responses/  
├── configurations.json  # 训练配置文件
```

## 适用场景

### 精度对齐（例如确定性问题）

1. 训练关键数据采集。

按如下配置运行两次模型（两次需要设置不同的dump_path）
```yaml
msprobe_config:
  msprobe: true
  dump_path: "./msprobe_dump"
  key_data_dump: true
```
得到两次训练过程的关键阶段性数据，这个数据我们用来定界到模型或代码块。

2. 将采集到的数据进行比对。

复制如下训练脚本，将dump_path1和dump_path2改为前一步中两次采集设置的两个输出路径，output_path改为自己的存盘路径，执行该脚本：
```python
from msprobe.core import SingleComparator
SingleComparator.compare(
    "dump_path1", 
    "dump_path2", 
    "output_path")
```
得到一个比对结果目录，里面会包含各项关键数据比对结果表格。

3. 观察结果表格，找到首个出现差异的地方，例如responses完全一致，ref_log_prob存在差异，则可以定界到reference model计算存在确定性问题

4. 再按如下配置运行两次模型（两次需要设置不同的dump_path，step_start和step_end均设置为问题出现的步数）
```yaml
msprobe_config:
  msprobe: true
  dump_path: "./msprobe_dump"
  reference_dump: true
  step_start: 2
  step_end: 2
```
得到两次训练过程的reference模型层的输入输出数据，这个数据我们用来定位问题点。

5. 将采集到的模型层输入输出进行比对

复制如下训练脚本，将./dump_path1/step2和./dump_path2/step2改为需要比对的step层级路径（比如./msprobe_dump/reference_compute_log_prob/step2），output_path改为自己的存盘路径，执行该脚本：
```python
from msprobe.pytorch import *
compare_distributed(
    './dump_path1/step2', 
    './dump_path2/step2', 
    './output_path')
```
获得一个比对结果表格compare_result_{timestamp}.xlsx。

6. 在结果表格中找到首个差异点，这就是问题点

**以上是一个简单示例，具体可以根据问题情况不同灵活运用该特性**

### 更多功能

[关键数据比对指南（key_data_dump）](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/34.RL_collect.md#%E7%BB%93%E6%9E%9C%E6%AF%94%E5%AF%B9)

[模型层数据比对指南（actor_train_dump、reference_dump）](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/10.accuracy_compare_PyTorch.md#222-compare_distributed-%E5%87%BD%E6%95%B0)
