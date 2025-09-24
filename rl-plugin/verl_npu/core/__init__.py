# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""
Core module for NPU plugin framework.

This module provides:
1. Variable-level patching framework for dicts and lists
2. Module-level patching framework for classes and modules  
3. Module injection utilities for aliases and symbols
4. Conditional patching support for all patch types
"""

from .variable_patcher import (
    # Core classes
    NPUVariablePatcher,
    PatchOperation,
    DictPatch,
    ListPatch,
    # Enums
    DictMode,
    ListMode,
    # Main interface
    D,
    L,
)

from .module_patcher import (
    # Module patch main classes
    NPUPatchHelper,
)

from .module_injection import (
    # Module injection utilities
    inject_module_alias,
    bootstrap_default_aliases,
)

from .conditional_patch import (
    # Conditional patch main classes
    conditional,
    ConditionalPatch,
    # Utility functions
    should_patch_method,
    get_class_condition,
)

from .patch_utils import (
    # Patch utility functions
    get_patch_summary,
    print_patch_summary,
    record_patch_entry,
    # NPU condition functions
    is_torch_npu_available,
)