# Variable Patch 特性

Variable Patch 是 NPU 插件框架的一个核心特性，提供了一种优雅且类型安全的方式来在运行时修改外部模块中的变量（字典和列表）。在用户需要的场景下，可以修改第三方库中的配置字典或模型列表。

## 特性概述

Variable Patch 特性解决了在集成 NPU 支持时需要动态修改外部库配置的问题。通过这个特性，您可以：

- **无侵入式集成**：无需修改外部库源码即可添加 NPU 支持
- **条件patch支持**：支持类级别条件装饰器，默认检查NPU可用性
- **类型安全**：为字典和列表操作提供独立的 API 和 Enum 支持
- **变量推断**：自动推断变量名称，生成有意义的运行时修改日志信息
- **双重语法支持**：既支持简洁的运算符语法，也支持明确的方法调用

## 核心组件

### 操作模式 Enum

```python
from verl_npu.core import DictMode, ListMode

# 字典操作模式
class DictMode(Enum):
    MERGE = "merge"      # 合并所有key（覆盖已存在的）
    ADD = "add"          # 只添加新key（跳过已存在的）
    UPDATE = "update"    # 只更新已存在的key（跳过新的）
    DELETE = "delete"    # 删除指定的key
    REPLACE = "replace"  # 替换整个字典内容

# 列表操作模式  
class ListMode(Enum):
    EXTEND = "extend"    # 添加项目到末尾（最常用）
    APPEND = "append"    # 追加项目到末尾（与extend相同）
    PREPEND = "prepend"  # 添加项目到开头
    INSERT = "insert"    # 在指定位置插入
    REMOVE = "remove"    # 移除指定项目
    REPLACE = "replace"  # 替换整个列表内容
```

### Patch 构建器

```python
from verl_npu.core import NPUVariablePatcher, D, L

# D() - 字典patch构建器
# L() - 列表patch构建器  
```

## 快速开始

### 基本用法 - 默认NPU条件

```python
from verl_npu.core import NPUVariablePatcher, D, L

# 从外部模块导入目标变量
from some_module import CONFIG_DICT, MODEL_LIST

# 定义NPU数据
npu_config = {"npu_enabled": True, "device_count": 8}
npu_models = ["qwen-npu", "deepseek-npu"]

class MyNPUPatch(NPUVariablePatcher):
    """默认只在NPU可用时应用patch"""
    # 运算符语法 - 最简洁
    D(npu_config) >> CONFIG_DICT    # 合并字典
    L(npu_models) >> MODEL_LIST     # 扩展列表

# 应用patch（只有NPU可用时才会实际修改变量）
MyNPUPatch.apply_patch()
```

### 条件patch用法

```python
from verl_npu.core import conditional, is_torch_npu_available

# 显式NPU条件
@conditional(is_torch_npu_available)
class ExplicitNPUPatch(NPUVariablePatcher):
    npu_config = {"advanced_npu": True}
    D(npu_config) >> CONFIG_DICT

# 自定义条件
@conditional(lambda: os.environ.get("EXPERIMENTAL", "0") == "1")
class ExperimentalPatch(NPUVariablePatcher):
    experimental_config = {"experimental_features": True}
    D(experimental_config) >> CONFIG_DICT

# 复杂条件组合
@conditional.all(
    is_torch_npu_available,
    lambda: os.environ.get("LARGE_MEMORY", "0") == "1"
)
class AdvancedPatch(NPUVariablePatcher):
    advanced_config = {"large_model_support": True}
    D(advanced_config) >> CONFIG_DICT
```

### 显式方法调用

```python
class MyNPUPatch(NPUVariablePatcher):
    # 显式方法 - 更具描述性
    D(npu_config, "NPU配置").merge_into(CONFIG_DICT)
    L(npu_models, "NPU模型").extend_to(MODEL_LIST)
```

## API 参考

### 字典操作

| 方法 | 运算符 | Enum | 描述 |
|------|--------|------|------|
| `.merge_into(target)` | `>>` | `DictMode.MERGE` | 合并所有key（覆盖已存在的） |
| `.add_to(target)` | `+` | `DictMode.ADD` | 只添加新key（跳过已存在的） |
| `.update_in(target)` | `\|` | `DictMode.UPDATE` | 只更新已存在的key（跳过新的） |
| `.replace_in(target)` | `<<` | `DictMode.REPLACE` | 替换整个字典内容 |

### 列表操作

| 方法 | 运算符 | Enum | 描述 |
|------|--------|------|------|
| `.extend_to(target)` | `>>` | `ListMode.EXTEND` | 添加项目到末尾（最常用） |
| `.append_to(target)` | `+` | `ListMode.APPEND` | 追加项目到末尾 |
| `.prepend_to(target)` | `*` | `ListMode.PREPEND` | 添加项目到开头 |
| `.insert_at(target, index)` | - | `ListMode.INSERT` | 在指定位置插入 |
| `.remove_from(target)` | `-` | `ListMode.REMOVE` | 从列表中移除项目 |
| `.replace_in(target)` | `<<` | `ListMode.REPLACE` | 替换整个列表内容 |


## 高级特性

### 1. 链式操作

将相同数据应用到多个目标：

```python
class ChainedPatch(NPUVariablePatcher):
    base_config = {"use_npu": True}
    
    # 链式多个操作
    D(base_config) \
        .merge_into(CONFIG_DICT) \
        .add_to(FEATURE_FLAGS) \
        .update_in(OPTIMIZER_CONFIG)
```

### 2. 类型安全的 Enum 使用

```python
from verl_npu.core import DictMode, ListMode, PatchOperation

# 直接使用Enum获得最佳类型安全
op = PatchOperation(npu_config, target_dict, DictMode.MERGE)
op.apply()
```

### 3. 智能变量名推断

Variable Patch 特性会自动推断变量名称并生成有意义的日志：

```python
npu_configuration = {"npu_enabled": True}
model_list = ["qwen-npu"]

class SmartPatch(NPUVariablePatcher):
    D(npu_configuration) >> DEFAULT_CONFIG
    L(model_list) >> SUPPORTED_MODELS

# 自动生成的日志：
# ✓ Merge npu_configuration -> DEFAULT_CONFIG
# ✓ Extend model_list -> SUPPORTED_MODELS
```

## 条件patch特性

### 概述

Variable Patch 支持类级别的条件patch，允许您根据运行时条件决定是否应用整个变量patch类。这为NPU集成提供了更灵活的策略。

### 默认NPU条件

所有 `NPUVariablePatcher` 子类默认包含NPU可用性检查：

```python
class AutoNPUPatch(NPUVariablePatcher):
    """默认只在NPU可用时应用"""
    
    npu_config = {"npu_enabled": True, "device_count": 8}
    npu_models = ["qwen2-npu", "deepseek-npu"]
    
    D(npu_config) >> CONFIG_DICT
    L(npu_models) >> MODEL_LIST

# 只有NPU可用时才会应用这些变量修改
AutoNPUPatch.apply_patch()
```

### 条件装饰器语法

#### 1. 额外条件（默认行为）

```python
from verl_npu.core import conditional, is_torch_npu_available

@conditional(is_torch_npu_available)
class ExplicitNPUPatch(NPUVariablePatcher):
    """显式指定NPU条件 - NPU可用 AND 显式条件"""
    
    advanced_npu_config = {
        "npu_optimization_level": 2,
        "npu_memory_pool": "large"
    }
    
    D(advanced_npu_config) >> CONFIG_DICT
```

#### 2. 替换默认条件

```python
@conditional.only(lambda: os.environ.get("TEST_MODE", "0") == "1")
class TestModePatch(NPUVariablePatcher):
    """替换默认NPU条件 - 只用测试模式条件"""
    
    test_config = {
        "test_mode": True,
        "mock_npu": True,
        "skip_npu_check": True
    }
    
    D(test_config) >> CONFIG_DICT
```

#### 3. 条件类型对比

| 装饰器 | 检查逻辑 | 使用场景 |
|--------|----------|----------|
| 无装饰器 | 默认NPU条件 | 基本NPU依赖patch |
| `@conditional` | 默认NPU条件 AND 额外条件 | 添加额外要求 |
| `@conditional.only` | 只用自定义条件 | 替换NPU要求 |

#### 自定义条件

```python
@conditional(lambda: os.environ.get("ENABLE_EXPERIMENTAL", "0") == "1")
class ExperimentalPatch(NPUVariablePatcher):
    """实验性功能patch"""
    
    experimental_features = {
        "flash_attention": True,
        "gradient_checkpointing": True
    }
    
    experimental_models = ["experimental-model-v1", "beta-model"]
    
    D(experimental_features) >> CONFIG_DICT
    L(experimental_models) >> MODEL_LIST
```

#### 条件组合

```python
@conditional.all(
    is_torch_npu_available,
    lambda: os.environ.get("LARGE_MEMORY", "0") == "1"
)
class LargeModelPatch(NPUVariablePatcher):
    """大模型支持patch - 需要NPU且大内存"""
    
    large_model_config = {
        "model_parallel": True,
        "pipeline_parallel": True,
        "memory_efficient_attention": True
    }
    
    large_models = ["qwen2-72b-npu", "deepseek-67b-npu"]
    
    D(large_model_config) >> CONFIG_DICT
    L(large_models) >> MODEL_LIST

@conditional.any(
    lambda: os.environ.get("DEBUG", "0") == "1",
    lambda: os.environ.get("DEV_MODE", "0") == "1"
)
class DevPatch(NPUVariablePatcher):
    """开发模式patch - NPU可用 AND (DEBUG=1 OR DEV_MODE=1)"""
    
    dev_config = {
        "debug_mode": True,
        "verbose_logging": True
    }
    
    D(dev_config) >> CONFIG_DICT

# 替换默认条件的开发模式patch
@conditional.only(lambda: os.environ.get("DEV_MODE", "0") == "1")
class DevOnlyPatch(NPUVariablePatcher):
    """开发模式patch - 只用DEV_MODE条件，忽略NPU检查"""
    
    dev_only_config = {
        "dev_mode": True,
        "assume_npu_available": True,
        "skip_npu_check": True
    }
    
    D(dev_only_config) >> CONFIG_DICT

@conditional.not_(lambda: os.environ.get("PRODUCTION", "0") == "1")
class NonProductionPatch(NPUVariablePatcher):
    """非生产环境patch"""
    
    test_config = {
        "test_mode": True,
        "mock_data": True
    }
    
    D(test_config) >> CONFIG_DICT
```

### 条件组合语法

| 语法 | 说明 | 示例 |
|------|------|------|
| `@conditional(func)` | 单个条件 | `@conditional(is_torch_npu_available)` |
| `@conditional.all(f1, f2)` | 所有条件都满足 | `@conditional.all(npu_available, large_memory)` |
| `@conditional.any(f1, f2)` | 任一条件满足 | `@conditional.any(debug_mode, dev_mode)` |
| `@conditional.not_(func)` | 条件不满足 | `@conditional.not_(production_mode)` |

### 替换默认条件

#### 使用 @conditional.only

`@conditional.only` 允许您完全替换默认的NPU条件检查，而不是添加额外条件：

```python
# 替换默认NPU条件，只用自定义条件
@conditional.only(lambda: os.environ.get("TEST_MODE", "0") == "1")
class TestModePatch(NPUVariablePatcher):
    """测试模式patch - 替换默认NPU条件"""
    
    test_config = {
        "test_mode": True,
        "mock_npu": True,
        "skip_npu_check": True
    }
    
    D(test_config) >> CONFIG_DICT
```

#### 条件类型对比

| 装饰器 | 检查逻辑 | 使用场景 |
|--------|----------|----------|
| 无装饰器 | 默认NPU条件 | 基本NPU依赖patch |
| `@conditional` | 默认NPU条件 AND 额外条件 | 添加额外要求 |
| `@conditional.only` | 只用自定义条件 | 替换NPU要求 |

#### 实际应用示例

```python
# 1. 测试环境：替换NPU检查
@conditional.only(lambda: os.environ.get("TEST_MODE") == "1")
class TestingNPUPatch(NPUVariablePatcher):
    """测试时替换NPU检查"""
    
    test_npu_config = {
        "npu_enabled": True,      # 测试时模拟NPU可用
        "npu_device_count": 1,    # 测试环境使用较少设备
        "test_mode": True
    }
    
    D(test_npu_config) >> CONFIG_DICT

# 2. 开发环境：替换NPU检查
@conditional.only(lambda: os.environ.get("DEV_MODE") == "1")
class DevelopmentNPUPatch(NPUVariablePatcher):
    """开发时替换NPU检查"""
    
    dev_npu_config = {
        "npu_enabled": True,      # 开发时假设NPU可用
        "dev_mode": True,
        "assume_npu_available": True
    }
    
    D(dev_npu_config) >> CONFIG_DICT

# 3. 强制应用：总是应用
@conditional.only(lambda: True)
class ForceApplyPatch(NPUVariablePatcher):
    """强制应用，忽略所有条件"""
    
    force_config = {
        "force_applied": True,
        "ignore_all_conditions": True
    }
    
    D(force_config) >> CONFIG_DICT
```

#### 组合条件替换

```python
# 替换默认条件，使用复杂条件组合
@conditional.only.all(
    lambda: os.environ.get("TEST_MODE") == "1",
    lambda: os.environ.get("SKIP_NPU_CHECK") == "1"
)
class ComplexTestPatch(NPUVariablePatcher):
    """复杂测试条件，替换默认NPU检查"""
    
    complex_config = {
        "test_mode": True,
        "skip_npu_check": True,
        "complex_condition": True
    }
    
    D(complex_config) >> CONFIG_DICT
```

### 实际应用场景

#### 环境相关patch

```python
@conditional(lambda: os.environ.get("ENV", "") == "development")
class DevelopmentPatch(NPUVariablePatcher):
    dev_config = {"debug_logging": True}
    D(dev_config) >> CONFIG_DICT

@conditional(lambda: os.environ.get("ENV", "") == "production")
class ProductionPatch(NPUVariablePatcher):
    prod_config = {"performance_monitoring": True}
    D(prod_config) >> CONFIG_DICT
```

#### 功能开关patch

```python
class FeatureFlagPatch(NPUVariablePatcher):
    """基于功能开关的条件patch"""
    
    # 基础NPU配置（使用默认NPU条件）
    base_npu_config = {"npu_enabled": True}
    D(base_npu_config) >> CONFIG_DICT

@conditional(lambda: os.environ.get("ENABLE_MIXED_PRECISION", "0") == "1")
class MixedPrecisionPatch(NPUVariablePatcher):
    mixed_precision_config = {"mixed_precision": True}
    D(mixed_precision_config) >> CONFIG_DICT

@conditional(lambda: os.environ.get("ENABLE_FLASH_ATTENTION", "0") == "1")
class FlashAttentionPatch(NPUVariablePatcher):
    flash_attention_config = {"flash_attention": True}
    D(flash_attention_config) >> CONFIG_DICT
```

### 条件patch日志

条件patch会提供清晰的日志输出：

```python
# NPU不可用时
MyNPUPatch.apply_patch()
# 输出: ⚠️  Skipped MyNPUPatch (conditions not met)

# NPU可用时
MyNPUPatch.apply_patch()
# 输出: 
# Applying MyNPUPatch...
# ✓ MyNPUPatch completed (2 operations)
```

## 与插件系统集成

### 1. 创建 patch 文件

```python
# my_npu_patch.py
from verl_npu.core import NPUVariablePatcher, D, L
from target_module import TARGET_CONFIG

class MyNPUPatch(NPUVariablePatcher):
    npu_config = {"npu_enabled": True}
    D(npu_config) >> TARGET_CONFIG
```

### 2. 在插件中注册

```python
# verl_npu/plugin.py
def apply_npu_plugin():
    # 现有的模块级patch...
    
    # 应用variable patch特性
    from .my_npu_patch import MyNPUPatch
    MyNPUPatch.apply_patch()
```

## 运算符语义表

### 字典操作语义

```python
# 合并操作（最常用）
D({"new_key": "value"}) >> target_dict
# 等同于: target_dict.update({"new_key": "value"})

# 添加新key（跳过已存在）
D({"new_key": "value"}) + target_dict
# 等同于: 只添加target_dict中不存在的key

# 更新已存在key（跳过新的）
D({"existing_key": "new_value"}) | target_dict
# 等同于: 只更新target_dict中已存在的key

# 完全替换
D({"replace": "all"}) << target_dict
# 等同于: target_dict.clear(); target_dict.update(...)
```

### 列表操作语义

```python
# 扩展到末尾（最常用）
L(["item1", "item2"]) >> target_list
# 等同于: target_list.extend(["item1", "item2"])

# 追加到末尾
L(["item1", "item2"]) + target_list
# 等同于: target_list.extend(["item1", "item2"])

# 添加到开头
L(["item1", "item2"]) * target_list
# 等同于: target_list[:0] = ["item1", "item2"]

# 从列表移除
L(["item1", "item2"]) - target_list
# 等同于: 从target_list中移除所有匹配的项目
```

## 类型安全和 IDE 支持

### Enum 的优势

1. **编译时类型检查**：IDE 可以在开发时检查类型错误
2. **自动完成**：输入 `DictMode.` 后 IDE 显示所有可用选项
3. **重构安全**：重命名 enum 值时 IDE 可以自动更新引用
4. **自文档化**：每个 enum 值都有清晰的注释说明
5. **更好的错误信息**：显示所有有效的模式选项

### 向后兼容

```python
# 字符串模式仍然有效（向后兼容）
D(data) >> target  # 内部使用 DictMode.MERGE

# 新的 Enum 模式（更好的类型安全）
from verl_npu.core import DictMode
PatchOperation(data, target, DictMode.MERGE)
```

## 常见问题

1. **导入错误**: 确保在 patch 之前导入目标模块
2. **类型错误**: 始终为字典使用 `D()`，为列表使用 `L()`
3. **无效果**: 检查目标变量是否是实际使用的对象