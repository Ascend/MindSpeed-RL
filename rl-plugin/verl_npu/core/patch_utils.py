# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""
Patch utility functions for NPU plugin framework.

This module contains common utility functions used by various patch components
to avoid circular imports and provide centralized functionality.
"""

import os
import importlib
from importlib import metadata
import logging
from typing import List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

# Global in-memory summary for applied patches
_PATCH_SUMMARY: List[Dict[str, Any]] = []


def _qualname(obj: Any) -> str:
    """Return fully-qualified name for a class or module."""
    module_name = getattr(obj, "__module__", None)
    obj_name = getattr(obj, "__name__", repr(obj))
    return f"{module_name}.{obj_name}" if module_name else obj_name


def get_patch_summary() -> List[Dict[str, Any]]:
    """Get a copy of the applied patch summary."""
    return list(_PATCH_SUMMARY)


def print_patch_summary() -> None:
    """Print a well-formatted summary of all applied patches (rank0 only)."""
    if not _is_primary_rank():
        return
    if not _PATCH_SUMMARY:
        msg = "[NPU Patch] No patches applied."
        logger.info(msg)
        return

    lines: List[str] = []
    lines.append("\n================ NPU Patch Summary ================")
    for index, record in enumerate(_PATCH_SUMMARY, start=1):
        target = record.get("target", "<unknown>")
        patch_cls = record.get("patch_class", "<unknown>")
        lines.append(f"{index}. Target: {target}")
        lines.append(f"   Patch : {patch_cls}")
        changes: List[Dict[str, str]] = record.get("changes", [])
        if changes:
            lines.append("   Changes:")
            for change in changes:
                action = change.get("action", "?")
                kind = change.get("kind", "attr")
                name = change.get("name", "?")
                lines.append(f"     - {action:<8} {kind:<11} {name}")
        else:
            lines.append("   Changes: <none>")
        
        # Add condition information if available
        if "condition_type" in record:
            condition_type = record.get("condition_type", "unknown")
            lines.append(f"   Condition Type: {condition_type}")
            
        if "class_condition" in record:
            class_condition = record.get("class_condition", "unknown")
            lines.append(f"   Class Condition: {class_condition}")
            
        if "skipped_methods" in record:
            skipped_methods = record.get("skipped_methods", [])
            if skipped_methods:
                lines.append(f"   Skipped Methods: {skipped_methods}")
    
    lines.append("===================================================\n")

    msg = "\n".join(lines)
    #log for visibility in various environments
    logger.info(msg)


def record_patch_entry(target_obj: Any, patch_obj: Any, changes: List[Dict[str, str]]) -> None:
    """Record a custom patch entry (for non-NPUPatchHelper operations).

    This is useful when we inject modules or symbols that are not handled by
    NPUPatchHelper.apply_patch but still want to show in the patch summary.
    """
    _PATCH_SUMMARY.append({
        "target": _qualname(target_obj) if not isinstance(target_obj, str) else target_obj,
        "patch_class": _qualname(patch_obj) if not isinstance(patch_obj, str) else target_obj,
        "changes": changes,
    })


def _is_primary_rank() -> bool:
    """Return True if this process is the primary (rank 0) process.

    Tries torch.distributed first; falls back to environment variables commonly
    used in distributed launchers. Defaults to True for single-process runs.
    """
    # Try PyTorch distributed
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        logger.warning("torch.distributed not available, falling back to environment variables")
    except Exception as e:
        logger.warning(f"Unexpected error checking torch.distributed rank: {e}, falling back to environment variables")

    # Fallback to common env vars
    for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if var in os.environ:
            try:
                return int(os.environ.get(var, "0")) == 0
            except ValueError as e:
                logger.warning(f"Invalid value for environment variable {var}: {os.environ[var]}, error: {e}")
                return True

    # Single-process default
    return True


def _is_package_available(pkg_name: str) -> bool:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            _ = metadata.metadata(pkg_name)
            return True
        except metadata.PackageNotFoundError:
            return False
    return False


_torch_available = _is_package_available("torch")


@lru_cache
def is_torch_npu_available() -> bool:
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if not _torch_available or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    return hasattr(torch, "npu") and torch.npu.is_available()