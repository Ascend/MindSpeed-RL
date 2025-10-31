# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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
from pathlib import Path
from typing import Dict, List, Any
import os
import logging
import yaml
import torch


cur_file_dir = Path(__file__).absolute().parent

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

# Global in-memory summary for applied patches
_PATCH_SUMMARY: Dict[str, List[Dict[str, Any]]] = {}


def get_patch_summary():
    return _PATCH_SUMMARY


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


def print_patch_summary() -> None:
    """Print a well-formatted summary of all applied patches (rank0 only).
    _PATCH_SUMMARY:
        {
            'transformers': {
                'current_rev': '8365f70e9259c21058bf18f876006f945d2a99de',
                'current_dir': '8365f70e9',
                'patch_files': {
                    'modeling_qwen2': {
                        'file_path': 'src.transformers.models.qwen2.modeling_qwen2.py',
                        'class_changes': [{
                            'class_name': 'Qwen2RMSNorm',
                            'class_action': 'updated',
                            'changes': [{
                                'action': 'replaced',
                                'kind': 'method',
                                'name': 'forward'
                            }]
                        }, {
                            'class_name': 'Qwen2MLP',
                            'class_action': 'updated',
                            'changes': [{
                                'action': 'replaced',
                                'kind': 'method',
                                'name': 'forward'
                            }]
                        }],
                        'module_changes': [{
                            'action': 'added',
                            'kind': 'method',
                            'name': 'fused_apply_rotary_pos_emb'
                        }]
                    }
                    }
                }
            }
        }
    
    """
    if not _is_primary_rank():
        return
    if not _PATCH_SUMMARY:
        msg = "[NPU Patch] No patches applied."
        logger.info(msg)
        return

    lines: List[str] = []
    lines.append("\n================================ NPU Patch Summary ==================================")

    # repo level
    for repo, patches in _PATCH_SUMMARY.items():
        lines.append(f"\n================ {repo} Patch Summary ================")
        # file level
        file_index = 1
        patches = patches["patch_files"]
        for _, file_patches in patches.items():
            file_path = file_patches["file_path"]
            class_changes = file_patches.get("class_changes", [])
            module_changes = file_patches.get("module_changes", [])
            lines.append(f"\nPatch File{file_index}: {file_path}")
            file_index += 1
            # module method level or attr
            if module_changes:
                lines.append("   Module Changes:")
                for module_change in module_changes:
                    action = module_change.get("action", "?")
                    kind = module_change.get("kind", "attr")
                    name = module_change.get("name", "?")
                    lines.append(f"          - {action}         {kind}         {name}")

            # class level
            for index, class_change in enumerate(class_changes, start=1):
                class_name = class_change["class_name"]
                class_action = class_change["class_action"]
                lines.append(f"  ({index}) Patch class: {file_path[:-3]}.{class_name}") # erase .py
                changes: List[Dict[str, str]] = class_change.get("changes", [])
                if changes:
                    lines.append("       Class Changes:")
                    for change in changes:
                        action = change.get("action", "?")
                        kind = change.get("kind", "attr")
                        name = change.get("name", "?")
                        lines.append(f"          - {action}         {kind}         {name}")

        lines.append(f"\n============ {repo} Patch Summary End ==============")
    
    lines.append("\n============================= NPU Patch Summary End==================================\n")

    msg = "\n".join(lines)
    #log for visibility in various environments

    logger.info(msg)


# analyse_changes and analyse_class_changes reserved for extension
def analyse_changes(changes, patch_summary_in_class, module_changes=False):
    new_changes = []
    for _, change in enumerate(changes):
        new_change = {}
        new_change["action"] = change.get("action", "?")
        new_change["kind"] = change.get("kind", "attr")
        new_change["name"] = change.get("name", "?")
        new_changes.append(new_change)
    if module_changes:
        patch_summary_in_class["module_changes"] = new_changes
    else:
        patch_summary_in_class["changes"] = new_changes


def analyse_class_changes(class_changes, patch_summary_file):
    all_class_changes = []
    for _, class_change in enumerate(class_changes):
        new_class_change = {}
        new_class_change["class_name"] = class_change.get("name", "<unknown>")
        new_class_change["class_action"] = class_change.get("action", "<unknown>")
        changes = class_change.get("changes", [])
        new_class_change["changes"] = {}
        analyse_changes(changes, new_class_change)
        all_class_changes.append(new_class_change)

    patch_summary_file["class_changes"] = all_class_changes


def find_path_in_patch(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("diff --git"):
                # [-1]: b/verl/workers/megatron_workers.py [2:]:erase "b/"
                file_path = line.split(" ")[-1][2:]
                break
    return file_path.replace("/", ".").replace("\n", "")


def patch_summary():
    with open(os.path.join(cur_file_dir, "patch_summary.yaml")) as file:
        runtime_env = yaml.safe_load(file)
    patches = runtime_env["patches"]
    for _, patch in enumerate(patches):
        repo = patch["repo"]
        _PATCH_SUMMARY[repo] = {}
        _PATCH_SUMMARY[repo]["current_rev"] = patch["current_rev"]
        cur_patch = [x for x in patch["versions"] if x["rev"] == patch["current_rev"]]
        if len(cur_patch) != 1:
            raise ValueError(f"There may be multiple versions equal to the current version in the YAML file.")

        cur_patch = cur_patch[0]
        current_dir = cur_patch["dir"]
        _PATCH_SUMMARY[repo]["current_dir"] = current_dir
        rev = cur_patch["rev"]
        patch_files = cur_patch["files"]
        _PATCH_SUMMARY[repo]["patch_files"] = {}
        repo_patch_files = _PATCH_SUMMARY[repo]["patch_files"]
        for _, patch_file in enumerate(patch_files):
            file_name = patch_file["name"]
            patch_file_path = os.path.join(cur_file_dir.parent, f"patch/{repo}/{current_dir}/{file_name}.patch")
            real_path = find_path_in_patch(patch_file_path)

            diff = patch_file["diff"]
            class_changes = diff.get("class_changes", None)
            module_changes = diff.get("module_changes", None)

            repo_patch_files[file_name] = {}
            repo_patch_files[file_name]["file_path"] = real_path

            if class_changes is not None:
                analyse_class_changes(class_changes, repo_patch_files[file_name])
            
            if module_changes is not None:
                analyse_changes(module_changes, repo_patch_files[file_name], True)
