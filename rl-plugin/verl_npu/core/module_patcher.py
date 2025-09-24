# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""
Module-level patching framework for NPU plugin.

This module provides NPUPatchHelper class for dynamically extending
existing classes and modules with conditional support.
"""

import logging
from types import MethodType, ModuleType
from typing import Type, Union, List, Dict, List

from .conditional_patch import get_class_condition
from .patch_utils import record_patch_entry, _is_primary_rank, is_torch_npu_available

logger = logging.getLogger(__name__)

Patchable = Union[Type, ModuleType]


# Copy from ArcticInference to allow patch existing classes or modules.
class NPUPatchHelper:
    """
    NPUPatchHelper provides a mechanism for cleanly patching (extending or
    modifying) existing classes or modules with conditional support.

    This class uses a subscription syntax to specify the target class or
    module to be patched. Subclasses of NPUPatchHelper should define new or
    replacement attributes and methods that will be applied in-place to the
    target when `apply_patch()` is called.

    Conditional patching is supported through decorators:
    - Class-level: @conditional decorates the entire patch class
    - Method-level: @conditional decorates individual methods

    Example 1: Basic patching

    ```python
    class ExamplePatch(NPUPatchHelper[SomeClass]):
        def new_method(self):
            return "This method will be added to SomeClass"

    ExamplePatch.apply_patch()
    ```

    Example 2: Conditional patching

    ```python
    from verl_npu.core import conditional, is_torch_npu_available

    @conditional(is_torch_npu_available)
    class NPUPatch(NPUPatchHelper[SomeClass]):
        def npu_method(self):
            return "Only added when NPU is available"

    class MixedPatch(NPUPatchHelper[SomeClass]):
        def always_added(self):
            return "Always added"
        
        @conditional(lambda: os.getenv("DEBUG") == "1")
        def debug_method(self):
            return "Only added in debug mode"

    NPUPatch.apply_patch()
    MixedPatch.apply_patch()
    ```
    
    The class includes a default NPU condition that can be overridden.
    """
    
    # Default condition: NPU must be available
    __default_patch_condition__ = is_torch_npu_available

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure that subclasses are created using the subscript syntax.
        if not hasattr(cls, '_npu_patch_target'):
            raise TypeError("Subclasses of NPUPatchHelper must be defined as "
                            "NPUPatchHelper[Target] to specify a patch target")

    @classmethod
    def __class_getitem__(cls, target: Patchable) -> Type:
        # The dynamic type created here will carry the target class as
        # _npu_patch_target.
        if not isinstance(target, Patchable):
            raise TypeError(f"NPUPatchHelper can only target a class or module, "
                            f"not {type(target)}")
        return type(f"{cls.__name__}[{target.__name__}]", (cls,),
                    {'_npu_patch_target': target})

    @classmethod
    def apply_patch(cls):
        """
        Patches the target class or module by replacing its attributes with
        those defined on the NPUPatchHelper subclass. Attributes are directly
        assigned to the target, and classmethods are re-bound to the target
        class before assignment.

        Supports conditional patching through class-level and method-level
        decorators. Methods that don't meet their conditions will be skipped.

        Raises:
            TypeError: If the NPUPatchHelper subclass is not defined with a target
                class or module.
            ValueError: If an attribute is already patched on the target.
        """
        if cls is NPUPatchHelper or not issubclass(cls, NPUPatchHelper):
            raise TypeError("apply_patch() must be called on a subclass of "
                            "NPUPatchHelper")

        target = cls._npu_patch_target

        if "_npu_patches" not in target.__dict__:
            target._npu_patches = {}

        # Get class-level condition (if any)
        class_condition = get_class_condition(cls)
        replace_default = getattr(cls, '__replace_default__', False)
        
        if replace_default:
            # User explicitly replaced default condition, only check user condition
            if class_condition is not None and not class_condition.evaluate():
                if _is_primary_rank():
                    logger.info(f"Skipping patch class {cls.__name__} due to unmet replace_default condition")
                return
        else:
            # Check default condition first, then user condition
            if not cls.__default_patch_condition__():
                if _is_primary_rank():
                    logger.info(f"Skipping patch class {cls.__name__} due to unmet default NPU condition")
                return
            
            # Check user additional condition (if any)
            if class_condition is not None and not class_condition.evaluate():
                if _is_primary_rank():
                    logger.info(f"Skipping patch class {cls.__name__} due to unmet additional condition")
                return

        changes: List[Dict[str, str]] = []
        skipped_methods: List[str] = []
        
        for name, attr in cls.__dict__.items():

            # Skip special names and the '_npu_patch_target' itself
            if name in ("_npu_patch_target", "__dict__", "__weakref__",
                        "__module__", "__doc__", "__parameters__", 
                        "__patch_condition__", "__default_patch_condition__"):
                continue

            # Check method-level conditions for callable attributes
            if callable(attr) and hasattr(attr, '__method_condition__'):
                method_condition = attr.__method_condition__
                if not method_condition.evaluate():
                    skipped_methods.append(name)
                    if _is_primary_rank():
                        logger.info(f"Skipping method {name} due to unmet method-level conditions")
                    continue
            
            # Check if the attribute has already been patched
            if name in target._npu_patches:
                patch = target._npu_patches[name]
                raise ValueError(f"{target.__name__}.{name} is already "
                                 f"patched by {patch.__name__}")
            target._npu_patches[name] = cls

            # If classmethod, re-bind it to the target
            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)

            # Patch the target with the new attribute
            replace = hasattr(target, name)
            setattr(target, name, attr)
            action = "replaced" if replace else "added"
            if _is_primary_rank():
                logger.info(f"{cls.__name__} {action} {target.__name__}.{name}")
            
            # Classify the kind of change for summary
            if isinstance(attr, classmethod):
                kind = "classmethod"
            elif isinstance(attr, staticmethod):
                kind = "staticmethod"
            elif callable(attr):
                kind = "callable"
            else:
                kind = "attribute"
            changes.append({"name": name, "action": action, "kind": kind})

        # Record a summary entry for this patch class
        summary_entry = {
            "target": f"{target.__module__}.{target.__name__}" if hasattr(target, '__module__') else str(target),
            "patch_class": f"{cls.__module__}.{cls.__name__}",
            "changes": changes,
        }
        
        # Add information about skipped methods
        if skipped_methods:
            summary_entry["skipped_methods"] = skipped_methods
            
        # Add condition information
        if class_condition is not None:
            summary_entry["class_condition"] = str(class_condition)
            if replace_default:
                summary_entry["condition_type"] = "replace_default"
            else:
                summary_entry["condition_type"] = "additional"
        else:
            summary_entry["condition_type"] = "default_only"
            
        record_patch_entry(target, cls, changes)
