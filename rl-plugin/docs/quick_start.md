# NPU 插件框架 - 快速开始

## 简介

NPU 插件框架提供了一套完整的运行时修改和扩展外部模块的特性。该框架包含三个核心特性，专为 NPU 集成场景设计：

1. **Variable Patch 特性** - 运行时修改变量（字典和列表）
2. **Module Patch 特性** - 动态扩展类和模块功能  
3. **Module Alias 特性** - 注入模块别名和符号

## 框架优势

- **无侵入式集成**：无需修改外部库源码即可添加 NPU 支持
- **类型安全**：完整的类型检查和 Enum 支持
- **智能推断**：自动变量名推断和有意义的日志
- **完整追踪**：详细的操作摘要和错误处理
- **IDE友好**：完整的自动完成和类型检查支持

## 特性概览

### 1. Variable Patch 特性 - 修改变量

```python
from verl_npu.core import NPUVariablePatcher, D, L

# 定义NPU数据
npu_config = {"npu_enabled": True, "device_count": 8}
npu_models = ["qwen-npu", "deepseek-npu"]

class MyVariablePatch(NPUVariablePatcher):
    D(npu_config) >> CONFIG_DICT    # 合并字典
    L(npu_models) >> MODEL_LIST     # 扩展列表

MyVariablePatch.apply_patch()
```

### 2. Module Patch 特性 - 扩展类/模块

```python
from verl_npu.core import NPUPatchHelper
from some_library import ModelClass

class MyModulePatch(NPUPatchHelper[ModelClass]):
    # 添加NPU方法
    def enable_npu(self):
        self.npu_enabled = True
    
    @classmethod
    def create_npu_model(cls, config):
        return cls(config)

MyModulePatch.apply_patch()
```

### 3. Module Alias 特性 - 注入模块别名

```python
from verl_npu.core import inject_module_alias

# 将本地模块映射到外部包命名空间
inject_module_alias(
    source_module_name="verl_npu.workers.npu_worker",
    target_module_name="verl.workers.npu_worker"
)

# 现在可以透明导入
from verl.workers.npu_worker import NPUWorker
```

## 特性对比

| 特性 | 用途 | 目标 | 示例场景 |
|------|------|------|----------|
| Variable Patch | 修改变量内容 | 字典、列表变量 | 添加NPU配置到现有配置字典 |
| Module Patch | 扩展功能 | 类、模块 | 为模型类添加NPU训练方法 |
| Module Alias | 模块映射 | 模块命名空间 | 让上游包透明使用本地NPU实现 |

## 日志和摘要

框架会自动生成清晰的日志和摘要：

### Variable Patch 日志
```
Applying MyVariablePatch...
✓ Merge npu_config -> CONFIG_DICT
✓ Extend npu_models -> MODEL_LIST
✓ MyVariablePatch completed (2 operations)
```

### Module Patch 摘要
```
================ NPU Patch Summary ================
1. Target: some_library.ModelClass
   Patch : MyModulePatch
   Changes:
     - added    callable    enable_npu
     - added    classmethod create_npu_model
===================================================
```

## 完整集成示例

```python
# 完整的NPU插件集成示例
from verl_npu.core import NPUVariablePatcher, D, L, NPUPatchHelper, inject_module_alias

# 1. Module Alias - 注入必要的模块别名
inject_module_alias(
    "verl_npu.workers.hybrid_tp_config",
    "verl.workers.sharding_manager.hybrid_tp_config"
)

# 2. Module Patch - 扩展模型类功能
class ModelNPUPatch(NPUPatchHelper[ModelClass]):
    def enable_npu_training(self):
        self.use_npu = True
        return self

# 3. Variable Patch - 修改配置变量
class ConfigNPUPatch(NPUVariablePatcher):
    npu_config = {"npu_enabled": True, "device_count": 8}
    npu_models = ["qwen2-npu", "deepseek-npu"]
    
    D(npu_config) >> TRAINING_CONFIG
    L(npu_models) >> SUPPORTED_MODELS

# 应用所有patch
ModelNPUPatch.apply_patch()
ConfigNPUPatch.apply_patch()
```

## 目录结构

```
rl-plugin/
├── verl_npu/
│   ├── core/                        # 核心功能模块
│   │   ├── __init__.py              
│   │   ├── variable_patcher.py      # Variable Patch 特性
│   │   ├── module_patcher.py        # Module Patch 特性
│   │   ├── module_injection.py      # Module Alias 特性
│   │   └── conditional_patch.py     # 条件patch支持
│   ├── plugin.py                    # 插件集成点
└── docs/                           # 特性文档
    ├── variable_patch_feature.md   # Variable Patch 特性
    ├── module_patch_feature.md     # Module Patch 特性
    ├── module_alias_feature.md     # Module Alias 特性
    └── quick_start.md              # 本文档
```

## 下一步

### 了解各特性
1. **Variable Patch**: 阅读 `docs/variable_patch_feature.md`
2. **Module Patch**: 阅读 `docs/module_patch_feature.md`
3. **Module Alias**: 阅读 `docs/module_alias_feature.md`

### 实践和测试
1. 在 `plugin.py` 中注册你的 patch 类

### 集成到插件

```python
# verl_npu/plugin.py
def apply_npu_plugin():
    # 1. Module Alias (最早执行)
    from .module_injection import bootstrap_default_aliases
    bootstrap_default_aliases()
    
    # 2. Module Patch
    from .my_module_patches import ModelPatch
    ModelPatch.apply_patch()
    
    # 3. Variable Patch
    from .my_variable_patches import ConfigPatch
    ConfigPatch.apply_patch()
    
    # 4. 查看摘要
    from .patch_util import print_patch_summary
    print_patch_summary()
```

修改完毕后，按照`README.MD`中的指引进行`pip install -e .`源码安装插件即可生效（需要提前源码安装`verl` package），如果patch成功，执行`verl`的训练行为时会在日志中打屏幕输出对应的patch详细信息。
