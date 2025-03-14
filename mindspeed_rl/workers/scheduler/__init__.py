# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from .launcher import RayActorGroup, get_npu_deployment

__all__ = [
    'get_npu_deployment',
    'RayActorGroup',
]
