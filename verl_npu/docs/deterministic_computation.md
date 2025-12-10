# 确定性计算

## 概述

确定性计算是指计算结果完全由输入决定，在给定相同输入的情况下总是产生相同输出的计算过程。


## 配置流程

### 数据输入一致
     
在verl的启动脚本中设置shuffle参数
      
      ```
      data.shuffle=False
      data.validation_shuffle=False
      ```
     
### 使能端到端的确定性seed

对于fsdp训练后端和megatron后端开启方式分别为在fsdp_worker.py/megatron_worker.py文件开头使能seed函数
      
      ```
        import random
        import numpy as np
        import torch
        import torch_npu
        import os

        def seed_all(seed=1234):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            os.environ['HCCL_DETERMINISTIC'] = str(True)
            os.environ['LCCL_DETERMINISTIC'] = str(1)
            os.environ['CLOSE_MATMUL_K_SHIFT'] = str(1)
            os.environ['ATB_LLM_LCOC_ENABLE'] = "0"
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch_npu.npu.manual_seed_all(seed)
            torch_npu.npu.manual_seed(seed)

        seed_all()
        

