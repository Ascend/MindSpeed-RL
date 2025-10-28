# Module Patch 特性

Module Patch 是 NPU 插件框架的一个核心特性，提供了一种干净的机制来动态扩展或修改现有的类和模块。该特性允许您在运行时向目标类或模块添加新的方法、属性和功能，而无需修改原始源代码。

## 特性概述

Module Patch 特性解决了在集成 NPU 支持时需要扩展外部类和模块功能的问题。通过这个特性，您可以：

- **无侵入式扩展**：向现有类添加新方法和属性，无需修改原始代码
- **条件patch**：支持类级别和方法级别的条件装饰器
- **类型安全**：使用订阅语法指定目标，编译时类型检查
- **冲突检测**：自动检测重复patch，避免意外覆盖
- **完整追踪**：详细的patch摘要和日志记录
- **灵活应用**：支持类方法、静态方法、实例方法和属性

## 核心组件

### NPUPatchHelper 基类

```python
from verl_npu.core import NPUPatchHelper

class MyPatch(NPUPatchHelper[TargetClass]):
    # 添加新的属性和方法
    new_field = "This will be added to TargetClass"
    
    def new_method(self):
        return "This method will be added to TargetClass"
    
    @classmethod
    def new_classmethod(cls):
        return "This classmethod will be added to TargetClass"
    
    @staticmethod
    def new_staticmethod():
        return "This staticmethod will be added to TargetClass"
```

### 条件patch装饰器

```python
from verl_npu.core import conditional, is_torch_npu_available

# 类级别条件patch
@conditional(is_torch_npu_available)
class NPUPatch(NPUPatchHelper[TargetClass]):
    def npu_method(self):
        return "Only patched when NPU is available"

# 方法级别条件patch
class MixedPatch(NPUPatchHelper[TargetClass]):
    def always_patched(self):
        return "Always patched"
    
    @conditional(lambda: os.environ.get("DEBUG", "0") == "1")
    def debug_method(self):
        return "Only patched in debug mode"
```

## 快速开始

### 基本用法 - 扩展类

```python
from verl_npu.core import NPUPatchHelper

# 假设我们要扩展一个现有的模型类
from some_library import ModelClass

class NPUModelPatch(NPUPatchHelper[ModelClass]):
    """为ModelClass添加NPU支持"""
    
    # 添加NPU相关属性
    npu_enabled = True
    npu_device_count = 8
    
    def enable_npu(self):
        """启用NPU加速"""
        self.npu_enabled = True
        print("NPU acceleration enabled")
    
    def get_npu_info(self):
        """获取NPU信息"""
        return {
            "enabled": self.npu_enabled,
            "device_count": self.npu_device_count
        }
    
    @classmethod
    def create_npu_model(cls, config):
        """创建NPU优化的模型实例"""
        instance = cls(config)
        instance.enable_npu()
        return instance

# 应用patch
NPUModelPatch.apply_patch()

# 现在可以使用新功能
model = ModelClass()
model.enable_npu()  # 新方法可用
print(model.get_npu_info())  # 新方法可用
npu_model = ModelClass.create_npu_model(config)  # 新类方法可用
```

### 扩展模块

```python
import some_module
from verl_npu.core import NPUPatchHelper

class NPUModulePatch(NPUPatchHelper[some_module]):
    """为模块添加NPU相关功能"""
    
    # 添加新常量
    NPU_BACKEND = "ascend"
    NPU_PRECISION = "fp16"
    
    @staticmethod
    def get_npu_devices():
        """获取可用的NPU设备"""
        return list(range(8))  # 假设有8个NPU设备
    
    @staticmethod
    def init_npu_context():
        """初始化NPU上下文"""
        print("Initializing NPU context...")
        return True

# 应用patch
NPUModulePatch.apply_patch()

# 现在可以使用新功能
print(some_module.NPU_BACKEND)  # 新常量可用
devices = some_module.get_npu_devices()  # 新函数可用
some_module.init_npu_context()  # 新函数可用
```

## 高级特性

### 1. 方法替换和扩展

```python
class AdvancedPatch(NPUPatchHelper[TargetClass]):
    """高级patch示例：替换和扩展现有方法"""
    
    def __init__(self, *args, **kwargs):
        """扩展构造函数"""
        super().__init__(*args, **kwargs)
        self.npu_initialized = False
    
    def forward(self, x):
        """替换forward方法以支持NPU"""
        if hasattr(self, 'npu_enabled') and self.npu_enabled:
            # NPU加速的forward实现
            return self._npu_forward(x)
        else:
            # 原始实现
            return self._original_forward(x)
    
    def _npu_forward(self, x):
        """NPU优化的forward实现"""
        print("Using NPU accelerated forward")
        return x  # 简化实现
```

### 2. 条件patch - 默认NPU检查

```python
# NPUPatchHelper默认包含NPU可用性检查
class AutoNPUPatch(NPUPatchHelper[TargetClass]):
    """自动使用NPU可用性检查"""
    
    def enable_npu_training(self):
        """只有NPU可用时才会被patch"""
        return "NPU training enabled"
    
    def optimize_npu_memory(self):
        """NPU内存优化"""
        return "NPU memory optimized"
```

### 3. 条件patch - 显式条件

```python
from verl_npu.core import conditional, is_torch_npu_available

@conditional(is_torch_npu_available)
class ExplicitNPUPatch(NPUPatchHelper[TargetClass]):
    """显式指定NPU条件"""
    
    def advanced_npu_feature(self):
        return "Advanced NPU functionality"

# 复杂条件组合
@conditional.all(
    is_torch_npu_available,
    lambda: os.environ.get("LARGE_MEMORY", "0") == "1"
)
class AdvancedNPUPatch(NPUPatchHelper[TargetClass]):
    """需要NPU可用且大内存"""
    
    def train_large_model(self):
        return "Training large model on NPU"

# 方法级别条件
class MixedConditionalPatch(NPUPatchHelper[TargetClass]):
    """混合条件patch"""
    
    def basic_npu_method(self):
        """使用默认NPU条件"""
        return "Basic NPU method"
    
    @conditional(lambda: os.environ.get("EXPERIMENTAL", "0") == "1")
    def experimental_method(self):
        """实验性功能"""
        return "Experimental feature"
    
    @conditional(lambda: True)  # 总是patch
    def always_available(self):
        """无条件patch"""
        return "Always available"
```

## 条件patch特性

### 概述

条件patch允许您根据运行时条件决定是否应用patch，提供更灵活的NPU集成策略。

### 默认NPU条件

所有 `NPUPatchHelper` 子类默认包含NPU可用性检查：

```python
class AutoNPUPatch(NPUPatchHelper[TargetClass]):
    """默认只在NPU可用时应用"""
    
    def npu_method(self):
        return "NPU功能"
```

### 条件装饰器语法

#### 1. 额外条件（默认行为）

```python
from verl_npu.core import conditional, is_torch_npu_available

# 显式NPU条件
@conditional(is_torch_npu_available)
class ExplicitNPUPatch(NPUPatchHelper[TargetClass]):
    """显式指定NPU条件 - NPU可用 AND 显式条件"""
    def npu_feature(self):
        return "NPU feature"

# 自定义条件
@conditional(lambda: os.environ.get("ENABLE_EXPERIMENTAL", "0") == "1")
class ExperimentalPatch(NPUPatchHelper[TargetClass]):
    """实验性功能 - NPU可用 AND 实验性功能启用"""
    def experimental_feature(self):
        return "Experimental feature"

# 条件组合
@conditional.all(
    is_torch_npu_available,
    lambda: os.environ.get("LARGE_MEMORY", "0") == "1"
)
class AdvancedPatch(NPUPatchHelper[TargetClass]):
    """高级功能 - NPU可用 AND 大内存"""
    def advanced_feature(self):
        return "Advanced feature"
```

#### 2. 替换默认条件

```python
# 替换默认NPU条件，只用自定义条件
@conditional.only(lambda: os.environ.get("TEST_MODE", "0") == "1")
class TestModePatch(NPUPatchHelper[TargetClass]):
    """测试模式patch - 替换默认NPU条件"""
    
    def test_method(self):
        return "Only added when TEST_MODE=1, ignoring NPU check"
    
    def mock_npu_method(self):
        return "Mock NPU method for testing"

# 强制应用，忽略所有条件
@conditional.only(lambda: True)
class ForceApplyPatch(NPUPatchHelper[TargetClass]):
    """强制应用，忽略所有条件"""
    
    def force_method(self):
        return "Always available, no conditions"
```

#### 3. 条件类型对比

| 装饰器 | 检查逻辑 | 使用场景 |
|--------|----------|----------|
| 无装饰器 | 默认NPU条件 | 基本NPU依赖patch |
| `@conditional` | 默认NPU条件 AND 额外条件 | 添加额外要求 |
| `@conditional.only` | 只用自定义条件 | 替换NPU要求 |

#### 方法级别条件

```python
class MixedPatch(NPUPatchHelper[TargetClass]):
    """混合条件patch示例"""
    
    def default_npu_method(self):
        """使用默认NPU条件"""
        return "Default NPU method"
    
    @conditional(lambda: os.environ.get("DEBUG", "0") == "1")
    def debug_method(self):
        """调试模式专用"""
        return "Debug method"
    
    @conditional.any(
        lambda: os.environ.get("DEV", "0") == "1",
        lambda: os.environ.get("TEST", "0") == "1"
    )
    def dev_test_method(self):
        """开发或测试环境"""
        return "Dev/Test method"
    
    @conditional.not_(lambda: os.environ.get("PRODUCTION", "0") == "1")
    def non_production_method(self):
        """非生产环境"""
        return "Non-production method"
    
    @conditional(lambda: True)  # 总是应用
    def always_method(self):
        """无条件应用"""
        return "Always available"
```

### 条件组合语法

| 语法 | 说明 | 示例 |
|------|------|------|
| `@conditional(func)` | 单个额外条件 | `@conditional(is_torch_npu_available)` |
| `@conditional.only(func)` | 替换默认条件 | `@conditional.only(test_mode)` |
| `@conditional.all(f1, f2)` | 所有条件都满足 | `@conditional.all(npu_available, large_memory)` |
| `@conditional.any(f1, f2)` | 任一条件满足 | `@conditional.any(debug_mode, test_mode)` |
| `@conditional.not_(func)` | 条件不满足 | `@conditional.not_(production_mode)` |
| `@conditional.only.all(f1, f2)` | 替换默认条件，所有条件满足 | `@conditional.only.all(test_mode, skip_npu)` |

### 条件优先级

1. **方法级别条件** > **类级别条件** > **默认条件**
2. 方法级别条件会覆盖类级别和默认条件
3. **`@conditional.only`** 会完全替换默认条件，而不是添加额外条件

### 条件patch日志

条件patch会在摘要中显示详细信息：

```python
from verl_npu.core import print_patch_summary

print_patch_summary()

# 输出示例：
# ================ NPU Patch Summary ================
# 1. Target: some_library.ModelClass
#    Patch : MyConditionalPatch
#    Class Condition: ConditionalPatch.all(is_torch_npu_available, <lambda>)
#    Changes:
#      - added    callable    npu_method
#      - added    callable    advanced_feature
#    Skipped Methods: ['debug_method', 'experimental_method']
# ===================================================
```

## Patch 摘要和日志

### 查看Patch摘要

```python
from verl_npu.core import print_patch_summary, get_patch_summary

# 应用所有patch后，查看摘要
print_patch_summary()

# 输出示例：
# ================ NPU Patch Summary ================
# 1. Target: some_library.ModelClass
#    Patch : __main__.NPUModelPatch
#    Changes:
#      - added    attribute   npu_enabled
#      - added    callable    enable_npu
#      - added    callable    get_npu_info
#      - added    classmethod create_npu_model
# ===================================================
```


## 与插件系统集成

### 1. 创建Module Patch文件

```python
# npu_model_patches.py
from verl_npu.core import NPUPatchHelper
from target_library import ModelClass, TrainerClass

class ModelNPUPatch(NPUPatchHelper[ModelClass]):
    def enable_npu_acceleration(self):
        self.use_npu = True

class TrainerNPUPatch(NPUPatchHelper[TrainerClass]):
    def setup_npu_training(self):
        print("Setting up NPU training environment")
```

### 2. 在插件中注册

```python
# verl_npu/plugin.py
def apply_npu_plugin():
    # 应用module patch
    from .npu_model_patches import ModelNPUPatch, TrainerNPUPatch
    
    ModelNPUPatch.apply_patch()
    TrainerNPUPatch.apply_patch()
```

## 最佳实践

### 1. 命名约定

- Patch类使用描述性名称：`ModelNPUPatch`、`OptimizerNPUPatch`
- 方法名使用清晰的前缀：`npu_*`、`enable_*`、`setup_*`
- 避免与原有方法名冲突

### 2. 组织结构

```python
# 按功能组织patch
class BaseNPUPatch(NPUPatchHelper):
    """基础NPU功能"""
    
    def _init_npu_base(self):
        self.npu_initialized = True

class ModelNPUPatch(BaseNPUPatch[ModelClass]):
    """模型特定的NPU功能"""
    
    def enable_model_npu(self):
        self._init_npu_base()
        # 模型特定的NPU初始化
```


### 4. 测试和验证

```python
# 验证patch是否正确应用
def verify_patches():
    from verl_npu.core import get_patch_summary
    
    summary = get_patch_summary()
    expected_patches = ["ModelNPUPatch", "TrainerNPUPatch"]
    
    applied_patches = [entry["patch_class"] for entry in summary]
    
    for expected in expected_patches:
        if not any(expected in patch for patch in applied_patches):
            print(f"Warning: {expected} not found in applied patches")
```

## 故障排除

### 常见问题

1. **重复Patch错误**：
```python
# 错误：ValueError: TargetClass.method_name is already patched
# 解决：检查是否重复应用patch或方法名冲突
```

2. **目标类型错误**：
```python
# 错误：TypeError: NPUPatchHelper can only target a class or module
# 解决：确保目标是类或模块，不是实例
```

3. **导入顺序问题**：
```python
# 确保在使用前导入目标类
from target_library import TargetClass  # 必须在patch定义前
class MyPatch(NPUPatchHelper[TargetClass]):
    pass
```

## 总结

Module Patch 特性提供了一个安全的机制来扩展现有的类和模块。通过类型安全的语法、条件patch支持、自动冲突检测和详细的运行时summary报告，简化了各种修改场景和验证。

- ✅ **类型安全**：校验语法和编译时检查
- ✅ **条件patch**：支持类级别和方法级别的条件装饰器
- ✅ **默认NPU检查**：自动检查NPU可用性，避免无效patch
- ✅ **灵活条件组合**：支持 all、any、not 逻辑组合
- ✅ **冲突防护**：自动检测重复patch
- ✅ **完整追踪**：详细的patch摘要和日志，包含条件信息
- ✅ **灵活扩展**：支持各种类型的方法和属性
- ✅ **无侵入式**：不修改原始源代码，运行时自动注入
- ✅ **易于调试**：丰富的错误信息和调试工具
