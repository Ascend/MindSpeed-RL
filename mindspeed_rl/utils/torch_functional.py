# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contain small torch utilities
"""
import torch
import torch.distributed


def clip_by_value(x, tensor_min, tensor_max):
    """
    Clip the input tensor x to be within the range [tensor_min, tensor_max].

    Args:
        x: Input tensor
        tensor_min: tensor_min: Minimum value for clipping
        tensor_max: Maximum value for clipping

    Returns:
        Clipped tensor
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def masked_mean(values, mask, axis=None, epsilon=1e-8):
    """
    Compute mean of tensor with a masked values.

    Args:
        values: Input tensor
        mask: A mask used to specify which elements to consider when computing the mean
        axis: Specifies the axis along which to operate; if None, operates over all axes

    Returns:
        The mean of the data after applying the mask
    """

    return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + epsilon)


def masked_var(values, mask, unbiased=True):
    """
    Compute variance of tensor with masked values.

    Args:
        values: Input tensor
        mask: Mask used to specify which values to ignore during computation.
        unbiased: Whether to perform unbiased estimation, default is True.

    Returns:
         The variance of the data after mask processing.
    """
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        elif mask_sum == 1:
            bessel_correction = mask_sum
        else:
            bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """
    Whiten values with masked values.

    Args:
        values: Input tensor
        mask: Mask used to specify which values to ignore during computation.
        shift_mean: Whether to shift the mean, default is True.

    Returns:
        The whitened values.
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

