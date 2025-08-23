# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities to inject (alias) local modules into external package namespaces at runtime.

This allows us to provide missing modules expected by upstream packages without
modifying their source files on disk.
"""

from __future__ import annotations

import importlib
import sys
import logging
from types import ModuleType
from typing import Iterable, Optional

from .patch_util import record_patch_entry

logger = logging.getLogger(__name__)


def inject_module_alias(
    source_module_name: str,
    target_module_name: str,
    export_symbols: Optional[Iterable[str]] = None,
) -> bool:
    """
    Inject an alias module at `target_module_name` that re-exports symbols from
    `source_module_name`.

    - If the target already exists, this function returns True without changes.
    - Ensures the target is also attached to its parent package attribute so that
      `import parent.child` works reliably.

    Args:
        source_module_name: Existing module to alias from (e.g.,
            'verl_npu.workers.sharding_manager.hybrid_tp_config').
        target_module_name: Target alias module to create (e.g.,
            'verl.workers.sharding_manager.hybrid_tp_config').
        export_symbols: Optional iterable of attribute names to re-export. If
            None, exports all public attributes (without leading underscore).

    Returns:
        True if injection succeeded or already present, False on failure.
    """
    try:
        # If target module already exists, treat as success
        if target_module_name in sys.modules:
            return True

        source_mod = importlib.import_module(source_module_name)

        # Create the target module and copy selected symbols
        target_mod = ModuleType(target_module_name)
        target_mod.__package__ = target_module_name.rsplit('.', 1)[0]

        if export_symbols is None:
            export_symbols = [name for name in dir(source_mod) if not name.startswith('_')]

        for name in export_symbols:
            setattr(target_mod, name, getattr(source_mod, name))

        # Register in sys.modules
        sys.modules[target_module_name] = target_mod

        # Attach to parent package as attribute for `import parent.child`
        if '.' in target_module_name:
            parent_name, child_name = target_module_name.rsplit('.', 1)
            parent_mod = importlib.import_module(parent_name)
            setattr(parent_mod, child_name, target_mod)

        # Record patch entry for summary
        changes = [{"name": name, "action": "added", "kind": "module_attr"} for name in export_symbols]
        record_patch_entry(
            target_obj=target_module_name,
            patch_obj=f"alias:{source_module_name}",
            changes=changes,
        )

        return True
    except Exception as e:
        logger.error(f"Failed to inject module alias from '{source_module_name}' to '{target_module_name}': {e}")
        return False


def inject_module_aliases_batch(module_pairs: list[tuple[str, str]]) -> None:
    """
    Inject multiple module aliases in batch. If any injection fails, the process will exit.
    
    Args:
        module_pairs: List of (source_module_name, target_module_name) tuples to inject.
    """
    for source_mod, target_mod in module_pairs:
        if not inject_module_alias(source_module_name=source_mod, target_module_name=target_mod):
            raise RuntimeError(f"Failed to bootstrap module alias from '{source_mod}' to '{target_mod}'. Process will exit.")


def bootstrap_default_aliases() -> None:
    """Bootstrap default module aliases required by upstream packages.

    This should be called as early as possible (e.g., package __init__) so that
    any subsequent imports that rely on these modules succeed without having to
    restructure import orders elsewhere.
    """
    package_root = __name__.split('.')[0]  # e.g., 'verl_npu'
    
    # Define all module alias pairs here
    module_pairs = [
        # Map local hybrid_tp_config to upstream verl path
        (f"{package_root}.workers.sharding_manager.hybrid_tp_config", 
         "verl.workers.sharding_manager.hybrid_tp_config"),
    ]
    
    inject_module_aliases_batch(module_pairs)


