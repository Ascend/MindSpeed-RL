# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""
Conditional patch utilities for NPU plugin framework.

This module provides conditional patching capabilities that allow users to apply
patches only when certain conditions are met, supporting both method-level and
class-level conditional decorators.
"""

import logging
from typing import Callable, Any, List, Union, Type
from functools import wraps

logger = logging.getLogger(__name__)


class ConditionalPatch:
    """
    A decorator class that provides conditional patching functionality.
    
    Supports multiple syntax forms:
    - @conditional(condition_func) - Additional condition (AND with default)
    - @conditional.only(condition_func) - Replace default condition
    - @conditional.all(cond1, cond2, ...) - All conditions must be true
    - @conditional.any(cond1, cond2, ...) - Any condition must be true
    - @conditional.not_(condition_func) - Condition must be false
    """
    
    def __init__(self, *conditions: Callable[[], bool], replace_default=False):
        """
        Initialize conditional patch with conditions.
        
        Args:
            *conditions: Callable functions that return boolean values
            replace_default: If True, replace default conditions instead of adding to them
        """
        self.conditions = conditions
        self.mode = 'all'  # Default mode is AND (all conditions must be true)
        self.replace_default = replace_default
    
    @classmethod
    def all(cls, *conditions: Callable[[], bool]) -> 'ConditionalPatch':
        """
        Create a conditional patch that requires ALL conditions to be true.
        
        Args:
            *conditions: Callable functions that return boolean values
            
        Returns:
            ConditionalPatch instance with 'all' mode
        """
        instance = cls(*conditions)
        instance.mode = 'all'
        return instance
    
    @classmethod
    def any(cls, *conditions: Callable[[], bool]) -> 'ConditionalPatch':
        """
        Create a conditional patch that requires ANY condition to be true.
        
        Args:
            *conditions: Callable functions that return boolean values
            
        Returns:
            ConditionalPatch instance with 'any' mode
        """
        instance = cls(*conditions)
        instance.mode = 'any'
        return instance
    
    @classmethod
    def only(cls, *conditions: Callable[[], bool]) -> 'ConditionalPatch':
        """
        Create a conditional patch that replaces default conditions.
        
        Args:
            *conditions: Callable functions that return boolean values
            
        Returns:
            ConditionalPatch instance that replaces default conditions
        """
        instance = cls(*conditions, replace_default=True)
        instance.mode = 'all'  # Default to AND mode for only
        return instance
    
    @classmethod
    def not_(cls, condition: Callable[[], bool]) -> 'ConditionalPatch':
        """
        Create a conditional patch that requires the condition to be false.
        
        Args:
            condition: Callable function that returns boolean value
            
        Returns:
            ConditionalPatch instance with 'not' mode
        """
        instance = cls(condition)
        instance.mode = 'not'
        return instance
    
    def evaluate(self) -> bool:
        """
        Evaluate the conditions based on the mode.
        
        Returns:
            True if conditions are satisfied, False otherwise
        """
        try:
            if self.mode == 'all':
                return all(condition() for condition in self.conditions)
            elif self.mode == 'any':
                return any(condition() for condition in self.conditions)
            elif self.mode == 'not':
                if len(self.conditions) != 1:
                    raise ValueError("not_ mode requires exactly one condition")
                return not self.conditions[0]()
            else:
                raise ValueError(f"Unknown condition mode: {self.mode}")
        except Exception as e:
            logger.warning("Error evaluating patch condition: %s", e)
            return False
    
    def __call__(self, target: Union[Type, Callable]) -> Union[Type, Callable]:
        """
        Apply the conditional decorator to a class or method.
        
        Args:
            target: The class or method to be conditionally patched
            
        Returns:
            The decorated class or method
        """
        if isinstance(target, type):
            # Class decorator
            target.__patch_condition__ = self
            target.__replace_default__ = self.replace_default
            if self.replace_default:
                logger.debug("Applied replace_default condition to %s", target.__name__)
            else:
                logger.debug("Applied additional condition to %s", target.__name__)
            return target
        else:
            # Method decorator
            target.__method_condition__ = self
            logger.debug("Applied method-level condition to %s", target.__name__)
            return target
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        condition_names = []
        for condition in self.conditions:
            if hasattr(condition, '__name__'):
                condition_names.append(condition.__name__)
            else:
                condition_names.append('<lambda>')
        
        prefix = "ConditionalPatch.only" if self.replace_default else "ConditionalPatch"
        
        if self.mode == 'all':
            if self.replace_default:
                return f"{prefix}({', '.join(condition_names)})"
            else:
                return f"{prefix}.all({', '.join(condition_names)})"
        elif self.mode == 'any':
            return f"{prefix}.any({', '.join(condition_names)})"
        elif self.mode == 'not':
            return f"{prefix}.not_({condition_names[0]})"
        else:
            return f"{prefix}({', '.join(condition_names)})"


# Create the main conditional decorator instance
conditional = ConditionalPatch


def should_patch_method(method: Callable, class_condition: ConditionalPatch = None) -> bool:
    """
    Determine if a method should be patched based on its conditions.
    
    Args:
        method: The method to check
        class_condition: Class-level condition (if any)
        
    Returns:
        True if the method should be patched, False otherwise
    """
    # Check for method-level condition first (takes precedence)
    if hasattr(method, '__method_condition__'):
        method_condition = method.__method_condition__
        result = method_condition.evaluate()
        logger.debug("Method %s condition %s evaluated to %s", method.__name__, method_condition, result)
        return result
    
    # Fall back to class-level condition
    if class_condition is not None:
        result = class_condition.evaluate()
        logger.debug("Method %s using class condition %s evaluated to %s", method.__name__, class_condition, result)
        return result
    
    # No conditions, always patch
    logger.debug("Method %s has no conditions, will be patched", method.__name__)
    return True


def get_class_condition(patch_class: Type) -> ConditionalPatch:
    """
    Extract the class-level condition from a patch class.
    
    Args:
        patch_class: The patch class to examine
        
    Returns:
        ConditionalPatch instance if found, None otherwise
    """
    return getattr(patch_class, '__patch_condition__', None)


def filter_methods_by_condition(patch_class: Type, methods: List[str]) -> List[str]:
    """
    Filter methods based on their conditions.
    
    Args:
        patch_class: The patch class
        methods: List of method names to filter
        
    Returns:
        List of method names that should be patched
    """
    class_condition = get_class_condition(patch_class)
    filtered_methods = []
    
    for method_name in methods:
        method = getattr(patch_class, method_name)
        if should_patch_method(method, class_condition):
            filtered_methods.append(method_name)
        else:
            logger.info("Skipping method %s due to unmet conditions", method_name)
    
    return filtered_methods


