# Module Alias 特性

Module Alias 是 NPU 插件框架的一个核心特性，提供了在运行时对注入模块别名alias替换的能力。该特性允许您将本地模块映射到外部包的命名空间中，使得上游包能够透明地访问您的本地实现，而无需修改其源代码。

## 特性概述

Module Alias 特性解决了在集成 NPU 支持时需要提供缺失模块或替换现有模块的问题。通过这个特性，您可以：

- **透明模块替换**：将本地模块注入到外部包命名空间
- **依赖解耦**：避免修改上游包的导入语句
- **动态注入**：运行时决定模块映射关系
- **批量处理**：支持批量注入多个模块别名

## 核心组件

### 模块别名注入函数

```python
from verl_npu.core import inject_module_alias

# 注入单个模块别名
success = inject_module_alias(
    source_module_name="verl_npu.workers.custom_worker",
    target_module_name="verl.workers.custom_worker"
)
```

### 批量注入函数

```python
from verl_npu.core import inject_module_aliases_batch

# 批量注入多个模块别名
module_pairs = [
    ("verl_npu.models.npu_model", "verl.models.npu_model"),
    ("verl_npu.optimizers.npu_optimizer", "verl.optimizers.npu_optimizer"),
]
inject_module_aliases_batch(module_pairs)
```

### bootstrap函数

```python
from verl_npu.core import bootstrap_default_aliases

# 自动引导默认的模块别名
bootstrap_default_aliases()
```

## 快速开始

### 基本用法 - 单个模块别名

```python
from verl_npu.core import inject_module_alias

# 场景：上游包期望 verl.workers.npu_worker 模块存在
# 但我们的实现在 verl_npu.workers.npu_worker

# 注入模块别名
success = inject_module_alias(
    source_module_name="verl_npu.workers.npu_worker",
    target_module_name="verl.workers.npu_worker"
)

if success:
    print("Module alias injected successfully")
    
    # 现在上游包可以正常导入
    from verl.workers.npu_worker import NPUWorker  # 实际来自 verl_npu
    
    worker = NPUWorker()
    worker.start()
```

### 批量模块别名注入

```python
from verl_npu.core import inject_module_aliases_batch

# 定义多个模块映射关系
module_mappings = [
    # (源模块, 目标模块)
    ("verl_npu.workers.sharding_manager.hybrid_tp_config", 
     "verl.workers.sharding_manager.hybrid_tp_config"),
    
    ("verl_npu.models.npu_models", 
     "verl.models.npu_models"),
     
    ("verl_npu.optimizers.npu_optimizers", 
     "verl.optimizers.npu_optimizers"),
     
    ("verl_npu.utils.npu_utils", 
     "verl.utils.npu_utils"),
]

# 批量注入
try:
    inject_module_aliases_batch(module_mappings)
    print("All module aliases injected successfully")
except RuntimeError as e:
    print(f"Failed to inject module aliases: {e}")
```

## 如何进行别名替换

### 在`bootstrap_default_aliases`添加

```python
# verl_npu/module_injection.py 中的实现
def bootstrap_default_aliases():
    """引导默认的模块别名"""
    
    package_root = __name__.split('.')[0]  # 'verl_npu'
    
    # 定义所有默认的模块别名对
    module_pairs = [
        # 核心工作组件
        (f"{package_root}.workers.sharding_manager.hybrid_tp_config", 
         "verl.workers.sharding_manager.hybrid_tp_config"),
        
        # NPU特定模块
        (f"{package_root}.models.npu_models",
         "verl.models.npu_models"),
         
        # 添加更多默认映射...
    ]
    
    inject_module_aliases_batch(module_pairs)
```

## 与插件系统集成

### 在`__init__`中提早初始化

```python
# verl_npu/__init__.py
from verl_npu.core import bootstrap_default_aliases

def _initialize_npu_plugin():
    """初始化NPU插件"""
    
    # 第一步：引导模块别名替换（必须最早执行）
    bootstrap_default_aliases()
    
    # 第二步：应用其他patch
    from verl_npu.plugin import apply_npu_plugin
    apply_npu_plugin()

# 模块导入时立即初始化，防止patch模块加载的时候冲突
_initialize_npu_plugin()
```

## 最佳实践

### 1. 命名约定

```python
# 保持一致的命名模式
source_pattern = "verl_npu.{category}.{module_name}"
target_pattern = "verl.{category}.{module_name}"

# 例如：
# verl_npu.models.transformer_model -> verl.models.transformer_model
# verl_npu.workers.npu_worker -> verl.workers.npu_worker
```

### 2. 模块组织

```python
# 按功能分组模块别名
def setup_feature_based_aliases():
    """按功能分组设置模块别名"""
    
    feature_groups = {
        "core": [
            ("verl_npu.core.context", "verl.core.context"),
            ("verl_npu.core.device", "verl.core.device"),
        ],
        "models": [
            ("verl_npu.models.base", "verl.models.base"),
            ("verl_npu.models.transformer", "verl.models.transformer"),
        ],
        "training": [
            ("verl_npu.training.trainer", "verl.training.trainer"),
            ("verl_npu.training.optimizer", "verl.training.optimizer"),
        ]
    }
    
    for feature, modules in feature_groups.items():
        print(f"Setting up {feature} module aliases...")
        try:
            inject_module_aliases_batch(modules)
            print(f"✓ {feature} aliases completed")
        except RuntimeError as e:
            print(f"✗ {feature} aliases failed: {e}")
```

## 故障排除

### 常见问题

1. **模块已存在错误**：
```python
# 如果目标模块已存在，inject_module_alias 会直接返回 True
# 这是正常行为，不是错误
```

2. **导入顺序问题**：
```python
# 确保在任何使用别名的导入之前注入别名
bootstrap_default_aliases()  # 必须在这之前
from verl.workers.something import Something  # 这之后
```

3. **符号不存在**：
```python
# 检查源模块是否真的包含指定的符号
import verl_npu.source_module
print(dir(verl_npu.source_module))  # 查看可用符号
```
