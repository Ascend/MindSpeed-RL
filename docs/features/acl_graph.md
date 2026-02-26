# 推理图模式

## 概述

TorchAir（Torch Ascend Intermediate Representation）是昇腾为Ascend Extension for PyTorch（torch_npu）提供的图模式能力扩展库，支持PyTorch网络在昇腾设备上进行图模式​**推理**​。TorchAir基于torch.compile提供了昇腾设备的图模式编译后端，对接PyTorch的[Dynamo特性](https://pytorch.org/docs/main/torch.compiler_dynamo_overview.html)，将PyTorch的[FX](https://pytorch.org/docs/main/fx.html)（Functionalization）计算图转换为昇腾Ascend IR（Intermediate Representation）计算图，并通过GE（Graph Engine）启动计算图编译和执行能力。

在vLLM（Very Large Language Model）推理框架中，`enforce_eager=True`是一种执行模式开关，用于强制模型以“eager execution”方式运行，而非默认的图优化执行路径。该模式下，PyTorch的操作会逐条立即执行，不进行图构建或延迟求值。

## 使用介绍

torchair_graph参数：DeepSeek V3使能torchair图模式

enforce_eager参数：使能PyTorch eager模式。当显存有余量，建议该参数设置为false，以启动图模式，提高推理性能。

## 使用方法

```
general_config:
    enforce_eager: true
    torchair_graph：false
```

注意：

1. enforce_eager: false 时，TASK_QUEUE_ENABLE需设置为1，否则设置为2。
2. DeepSeek V3开启torchair_graph时，需要关闭enforce_eager。