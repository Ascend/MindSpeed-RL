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
# limitations under the License.import os

import pathlib
import shutil
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Optional
import logging as logger

from verl_npu.utils.pkg_version import check_commit_id
from verl_npu.patch_summary.patch_summary import get_patch_summary, print_patch_summary, patch_summary


def get_target_path(repo):
    target_path = None
    repo_meta = subprocess.check_output(
        [sys.executable, "-m", "pip", "show", repo]
    ).decode("utf-8")
    # Prioritize editable install location, since pip show lists both locations
    # if installed in editable mode.
    for line in repo_meta.split("\n"):
        line = line.strip()
        if line.startswith("Editable project location: "):
            target_path = str(Path(line.split(": ")[1]))
            break
    else:
        for line in repo_meta.split("\n"):
            line = line.strip()
            if line.startswith("Location: "):
                target_path = str(Path(line.split(": ")[1]) / repo)

    return target_path


def apply_repo_patches(repo, current_rev, repo_dir):
    target_path = get_target_path(repo)
    # check repo commit_id, ensure its ok
    check_commit_id(target_path, current_rev)

    p = Path(os.path.dirname(__file__))
    patch_dir = str(
        p
        / "patch"
        / repo
        / repo_dir
    )
    all_patch_files = os.listdir(path=patch_dir)

    for patch_file in all_patch_files:
        # if not end with .patch, continue!
        if not patch_file.endswith(".patch"):
            continue
        patch_path = patch_dir + "/" + patch_file
        patch_binary = shutil.which("patch")
        result = subprocess.run(
            [patch_binary, "-p1", "-N", "-i", patch_path],
            cwd=target_path,
            capture_output=True,
            text=True,
        )

        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            logger.info(f"Applied verl patch {patch_path} to {target_path}")
        elif (
            "Reversed (or previously applied) patch detected" in output
            or "Skipping patch." in output
        ):
            logger.warning(
                f"verl patch {patch_path} appears to be already applied for {target_path}."
            )
        else:
            logger.error(
                "Failed to apply verl patch %s to %s. Output:\n%s",
                patch_path,
                target_path,
                output.strip(),
            )
            raise RuntimeError(
                f"verl patch {patch_path} failed with exit code {result.returncode}."
            )


def apply_patch():
    patch_summary_dict = get_patch_summary()
    for repo, patches in patch_summary_dict.items():
        current_rev = patches["current_rev"]
        repo_dir = patches["current_dir"]
        apply_repo_patches(repo, current_rev, repo_dir)


def after_patch_check():
    """
    Verify that all lines added in the git diff exist in the actual package files.
    Check that all git diff additions are present in the package files.

    """
    patch_summary_dict = get_patch_summary()
    p = Path(os.path.dirname(__file__))

    for repo, patches in patch_summary_dict.items():
        target_path = get_target_path(repo)
        current_rev = patches["current_rev"]
        repo_dir = patches["current_dir"]
        patch_path = str(
            p
            / "patch"
            / repo
            / repo_dir
        )
        all_patch_files = os.listdir(path=patch_path)

        for patch_file in all_patch_files:
            added_lines = []
            patch_file_path = patch_path + "/" + patch_file
            with open(patch_file_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    # 新增文件
                    if line.startswith("+++"):
                        relative_path = line.split(" ")[-1][2:].replace("\n", "")
                        continue
                    if line.startswith("+"):
                        added_lines.append(line[1:].strip().replace("\n", ""))
            # 去真实路径下查找
            abs_path = target_path + "/" + relative_path
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not all(s in content for s in added_lines):
                        logger.info(f"file {patch_file} may modified or patch failed")
            except Exception as e:
                logger.error(f"An exception occurred while reading the package source files \
                during the post-patch validation process.\n {e}")


def apply_npu_plugin():
    """
    patch_summary parses the YAML configuration file for the patch and must be called first.
    Please make sure to apply the patches in the correct order so that they can 
    work properly.
    """
    apply_patch()
    print_patch_summary()
    after_patch_check()

    # In verl, the driver process aggregates the computation results of workers via Ray. 
    # Therefore, after a worker completes its computation job, it will package the output 
    # using tensordict and transfer it to the CPU. Since the `to` operation of tensordict 
    # is non-blocking, when transferring data from a device to the CPU, it is necessary to 
    # ensure that a batch of data has been completely transferred before being used on the 
    # host; otherwise, unexpected precision issues may arise. Tensordict has already noticed 
    # this problem and fixed it.
    # However, the relevant modifications only cover CUDA and MPS devices and do not take effect 
    # for third-party devices such as NPUs. This patch fixes this issue, and the relevant 
    # modifications can be removed once the fix is merged into tensordict.
    from tensordict.base import TensorDictBase

    def _sync_all_patch(self):
        import torch
        from torch._utils import _get_available_device_type, _get_device_module
        try:
            from torch.compiler import is_compiling
        except ImportError:  # torch 2.0
            from torch._dynamo import is_compiling

        device_type = _get_available_device_type()
        if device_type is None:
            return

        if device_type == "cuda":
            if not is_compiling() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        else:
            device_module = _get_device_module(device_type)
            device_module.synchronize()

    TensorDictBase._sync_all = _sync_all_patch