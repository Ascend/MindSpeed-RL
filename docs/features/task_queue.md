# TASK_QUEUE_ENABLE

## 概述

通过此环境变量可配置task_queue算子下发队列是否开启和优化等级。具体请参考[TASK\_QUEUE\_ENABLE-Ascend Extension for PyTorch7.2.0-昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_007.html)

## **使用方法**

注意：

1. [ASCEND\_LAUNCH\_BLOCKING](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_006.html)设置为“1”时，task_queue算子队列关闭，TASK_QUEUE_ENABLE设置不生效。
2. TASK_QUEUE_ENABLE配置为**“2”**时，由于内存并发，可能导致运行中NPU内存峰值上升。
3. 打开图模式(enforce eager = False )该值设置为1，否则设置为2

```
export TASK_QUEUE_ENABLE=2
```

## **使用效果**

* 对请求进行排队、批处理、排序，提高系统吞吐，避免队头阻塞
* 动态负载均衡（适用于 MoE 专家数不均衡）
 