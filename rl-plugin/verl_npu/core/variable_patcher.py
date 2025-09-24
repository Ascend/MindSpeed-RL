# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""
Elegant variable-level patching framework for runtime modification of dicts and lists.

This module provides a type-safe, IDE-friendly way to patch variables in external 
modules. It supports both operator overloading for concise syntax and explicit 
method calls for clarity.

Example usage:
    from verl_npu.core import NPUVariablePatcher, D, L
    
    npu_config = {"npu_enabled": True, "device_count": 8}
    npu_models = ["qwen-npu", "deepseek-npu"]
    
    class MyPatch(NPUVariablePatcher):
        # Operator syntax (concise)
        D(npu_config) >> DEFAULT_MODEL_CONFIG
        L(npu_models) >> SUPPORTED_MODELS
        
        # Method syntax (explicit)
        D(npu_config).merge_into(DEFAULT_MODEL_CONFIG)
        L(npu_models).extend_to(SUPPORTED_MODELS)
"""

from typing import Any, List, Union, Optional
import logging
from enum import Enum

from .conditional_patch import get_class_condition
from .patch_utils import is_torch_npu_available

logger = logging.getLogger(__name__)


class DictMode(Enum):
    """Dictionary patch operation modes."""
    MERGE = "merge"      # Merge all keys (overwrites existing)
    ADD = "add"          # Add new keys only (skip existing)
    UPDATE = "update"    # Update existing keys only (skip new)
    DELETE = "delete"    # Delete specified keys
    REPLACE = "replace"  # Replace entire dict content


class ListMode(Enum):
    """List patch operation modes."""
    EXTEND = "extend"    # Add items to end (most common)
    APPEND = "append"    # Append items to end (same as extend)
    PREPEND = "prepend"  # Add items to beginning
    INSERT = "insert"    # Insert at specific position
    REMOVE = "remove"    # Remove specified items
    REPLACE = "replace"  # Replace entire list content


class PatchOperation:
    """Represents a single patch operation to be applied."""
    
    def __init__(self, source: Any, target_obj: Any, mode: Union[str, DictMode, ListMode], description: Optional[str] = None):
        self.source = source
        self.target = target_obj
        self.mode = self._normalize_mode(mode)
        self.description = description or self._generate_description()
    
    def _normalize_mode(self, mode: Union[str, DictMode, ListMode]) -> str:
        """Convert enum mode to string for internal use."""
        if isinstance(mode, (DictMode, ListMode)):
            return mode.value
        return mode
    
    def _generate_description(self) -> str:
        """Generate simple description based on mode and target type."""
        target_type = type(self.target).__name__
        return f"{self.mode.capitalize()} operation on {target_type}"
    
    def apply(self):
        """Apply the patch operation."""
        try:
            if isinstance(self.target, dict):
                self._apply_dict_patch()
            elif isinstance(self.target, list):
                self._apply_list_patch()
            else:
                raise TypeError(f"Unsupported target type: {type(self.target)}")
            
            logger.info(f"✓ {self.description}")
        except Exception as e:
            logger.error(f"✗ {self.description} failed: {e}")
            raise
    
    def _ensure_list(self, data: Any) -> List[Any]:
        """Convert data to list format for consistent processing."""
        return data if isinstance(data, (list, tuple)) else [data]
    
    # Removed _validate_mode - validation now handled in handlers
    
    def _apply_dict_patch(self):
        """Apply patch to dictionary target."""
        handlers = {
            DictMode.MERGE.value: lambda: self.target.update(self.source),
            DictMode.ADD.value: lambda: self._add_new_keys(),
            DictMode.UPDATE.value: lambda: self._update_existing_keys(),
            DictMode.DELETE.value: lambda: self._delete_keys(),
            DictMode.REPLACE.value: lambda: self._replace_dict()
        }
        
        handler = handlers.get(self.mode)
        if handler is None:
            raise ValueError(f"Unsupported dict operation mode: {self.mode}. Valid modes: {list(handlers.keys())}")
        handler()
    
    def _add_new_keys(self):
        """Add only new keys to target dict."""
        for k, v in self.source.items():
            if k not in self.target:
                self.target[k] = v
    
    def _update_existing_keys(self):
        """Update only existing keys in target dict."""
        for k, v in self.source.items():
            if k in self.target:
                self.target[k] = v
    
    def _delete_keys(self):
        """Delete specified keys from target dict."""
        keys_to_delete = self._ensure_list(self.source)
        for k in keys_to_delete:
            self.target.pop(k, None)
    
    def _replace_dict(self):
        """Replace entire dict content."""
        self.target.clear()
        self.target.update(self.source)
    
    def _apply_list_patch(self):
        """Apply patch to list target."""
        handlers = {
            ListMode.EXTEND.value: lambda: self._extend_list(),
            ListMode.APPEND.value: lambda: self._extend_list(),  # Same as extend for lists
            ListMode.PREPEND.value: lambda: self._prepend_list(),
            ListMode.INSERT.value: lambda: self._insert_list(),
            ListMode.REMOVE.value: lambda: self._remove_items(),
            ListMode.REPLACE.value: lambda: self._replace_list()
        }
        
        handler = handlers.get(self.mode)
        if handler is None:
            raise ValueError(f"Unsupported list operation mode: {self.mode}. Valid modes: {list(handlers.keys())}")
        handler()
    
    def _extend_list(self):
        """Extend target list with items."""
        items = self._ensure_list(self.source)
        self.target.extend(items)
    
    def _prepend_list(self):
        """Prepend items to target list."""
        items = self._ensure_list(self.source)
        self.target[:0] = items
    
    def _insert_list(self):
        """Insert items at specific position."""
        index, items = self.source
        items_list = self._ensure_list(items)
        for i, item in enumerate(items_list):
            self.target.insert(index + i, item)
    
    def _remove_items(self):
        """Remove items from target list."""
        items = self._ensure_list(self.source)
        for item in items:
            while item in self.target:
                self.target.remove(item)
    
    def _replace_list(self):
        """Replace entire list content."""
        self.target.clear()
        items = self._ensure_list(self.source)
        self.target.extend(items)


class DictPatch:
    """Type-safe patch builder for dictionary operations."""
    
    def __init__(self, data: dict, description: Optional[str] = None):
        if not isinstance(data, dict):
            raise TypeError(f"DictPatch requires dict data, got {type(data)}")
        self.data = data
        self.description = description
        self._operations = []
    
    def _add_operation(self, target_obj: dict, mode: DictMode, source_data: Any = None) -> 'DictPatch':
        """Helper method to create and add operations."""
        data = source_data if source_data is not None else self.data
        op = PatchOperation(data, target_obj, mode, self.description)
        self._operations.append(op)
        return self
    
    # === Dictionary-specific methods ===
    def merge_into(self, target_obj: dict) -> 'DictPatch':
        """Merge into target dict (overwrites existing keys)."""
        return self._add_operation(target_obj, DictMode.MERGE)
    
    def add_to(self, target_obj: dict) -> 'DictPatch':
        """Add to target dict (skip existing keys)."""
        return self._add_operation(target_obj, DictMode.ADD)
    
    def update_in(self, target_obj: dict) -> 'DictPatch':
        """Update target dict (only update existing keys)."""
        return self._add_operation(target_obj, DictMode.UPDATE)
    
    def replace_in(self, target_obj: dict) -> 'DictPatch':
        """Completely replace target dict content."""
        return self._add_operation(target_obj, DictMode.REPLACE)
    
    def delete_keys_from(self, target_obj: dict, keys: List[str]) -> 'DictPatch':
        """Delete specified keys from target dict."""
        return self._add_operation(target_obj, DictMode.DELETE, keys)
    
    # === Operator overloading ===
    def __rshift__(self, target_obj: dict) -> 'DictPatch':
        """>> merge operation (most common)."""
        return self.merge_into(target_obj)
    
    def __add__(self, target_obj: dict) -> 'DictPatch':
        """+  add new keys only."""
        return self.add_to(target_obj)
    
    def __or__(self, target_obj: dict) -> 'DictPatch':
        """|  update existing keys only."""
        return self.update_in(target_obj)
    
    def __lshift__(self, target_obj: dict) -> 'DictPatch':
        """<< replace entire dict."""
        return self.replace_in(target_obj)
    
    def get_operations(self) -> List[PatchOperation]:
        """Get all registered operations."""
        return self._operations.copy()


class ListPatch:
    """Type-safe patch builder for list operations."""
    
    def __init__(self, data: Union[list, Any], description: Optional[str] = None):
        self.data = data
        self.description = description
        self._operations = []
    
    def _add_operation(self, target_obj: list, mode: ListMode, source_data: Any = None) -> 'ListPatch':
        """Helper method to create and add operations."""
        data = source_data if source_data is not None else self.data
        op = PatchOperation(data, target_obj, mode, self.description)
        self._operations.append(op)
        return self
    
    # === List-specific methods ===
    def extend_to(self, target_obj: list) -> 'ListPatch':
        """Extend target list with items."""
        return self._add_operation(target_obj, ListMode.EXTEND)
    
    def append_to(self, target_obj: list) -> 'ListPatch':
        """Append items to target list."""
        return self._add_operation(target_obj, ListMode.APPEND)
    
    def prepend_to(self, target_obj: list) -> 'ListPatch':
        """Prepend items to target list."""
        return self._add_operation(target_obj, ListMode.PREPEND)
    
    def insert_at(self, target_obj: list, index: int) -> 'ListPatch':
        """Insert items at specific position."""
        return self._add_operation(target_obj, ListMode.INSERT, (index, self.data))
    
    def remove_from(self, target_obj: list) -> 'ListPatch':
        """Remove items from target list."""
        return self._add_operation(target_obj, ListMode.REMOVE)
    
    def replace_in(self, target_obj: list) -> 'ListPatch':
        """Completely replace target list content."""
        return self._add_operation(target_obj, ListMode.REPLACE)
    
    # === Operator overloading ===
    def __rshift__(self, target_obj: list) -> 'ListPatch':
        """>> extend operation (most common)."""
        return self.extend_to(target_obj)
    
    def __add__(self, target_obj: list) -> 'ListPatch':
        """+  append operation."""
        return self.append_to(target_obj)
    
    def __mul__(self, target_obj: list) -> 'ListPatch':
        """*  prepend operation."""
        return self.prepend_to(target_obj)
    
    def __lshift__(self, target_obj: list) -> 'ListPatch':
        """<< replace entire list."""
        return self.replace_in(target_obj)
    
    def __sub__(self, target_obj: list) -> 'ListPatch':
        """-  remove operation."""
        return self.remove_from(target_obj)
    
    def get_operations(self) -> List[PatchOperation]:
        """Get all registered operations."""
        return self._operations.copy()


# === Convenience aliases (main interface) ===
def D(data: dict, desc: str = None) -> DictPatch:
    """Create a dictionary patch builder."""
    return DictPatch(data, desc)


def L(data: Union[list, Any], desc: str = None) -> ListPatch:
    """Create a list patch builder."""
    return ListPatch(data, desc)


# === Metaclass for automatic operation collection ===
class PatchMeta(type):
    """Metaclass that automatically collects patch operations from class body."""
    
    def __new__(cls, name, bases, namespace, **kwargs):
        operations = []
        
        # Collect operations from patch builders
        for key, value in list(namespace.items()):
            if hasattr(value, 'get_operations'):
                operations.extend(value.get_operations())
                del namespace[key]  # Clean up class namespace
            elif isinstance(value, PatchOperation):
                operations.append(value)
                del namespace[key]
        
        namespace['_patch_operations'] = operations
        return super().__new__(cls, name, bases, namespace)


class NPUVariablePatcher(metaclass=PatchMeta):
    """
    Base class for variable-level patching using elegant syntax.
    
    Supports conditional patching through class-level decorators.
    By default, all variable patches require NPU to be available.
    """
    
    # Default condition: NPU must be available
    __default_patch_condition__ = is_torch_npu_available
    
    @classmethod
    def apply_patch(cls):
        """Apply all registered patch operations if conditions are met."""
        if not hasattr(cls, '_patch_operations'):
            logger.warning(f"{cls.__name__} has no patch operations to apply")
            return 0
        
        # Get class-level condition (if any)
        class_condition = get_class_condition(cls)
        replace_default = getattr(cls, '__replace_default__', False)
        
        if replace_default:
            # User explicitly replaced default condition, only check user condition
            if class_condition is not None and not class_condition.evaluate():
                logger.info(f"Skipping variable patch class {cls.__name__} due to unmet replace_default condition")
                print(f"Skipped {cls.__name__} (replace_default condition not met)")
                return 0
        else:
            # Check default condition first, then user condition
            if not cls.__default_patch_condition__():
                logger.info(f"Skipping variable patch class {cls.__name__} due to unmet default NPU condition")
                print(f"Skipped {cls.__name__} (NPU not available)")
                return 0
            
            # Check user additional condition (if any)
            if class_condition is not None and not class_condition.evaluate():
                logger.info(f"Skipping variable patch class {cls.__name__} due to unmet additional condition")
                print(f"Skipped {cls.__name__} (additional condition not met)")
                return 0
        
        print(f"Applying {cls.__name__}...")
        
        success_count = 0
        for operation in cls._patch_operations:
            try:
                operation.apply()
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to apply operation in {cls.__name__}: {e}")
                raise
        
        print(f"✓ {cls.__name__} completed ({success_count} operations)\n")
        return success_count
