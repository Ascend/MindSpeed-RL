# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch


class Metric(ABC):
    def __init__(self):
        self.metric = {}

    def update(self, key="", value=None, cumulate=False):
        """
        只做参数更新
        key: str
        value: dict|list|tensor. when key is None, maybe value is a dict
        """
        if cumulate:
            if isinstance(value, Dict):
                if cumulate:
                    for key in value:
                        if key in self.metric:
                            if isinstance(self.metric[key], list):
                                self.metric[key].extend(value[key])
                            else:
                                self.metric[key] = value[key]
                        else:
                            self.metric[key] = [*value[key]]
                else:
                    self.metric.update(value)
            else:
                if key in self.metric:
                    if isinstance(self.metric[key], list):
                        self.metric[key].extend(value)
                    else:
                        self.metric[key] = value
                else:
                    self.metric[key] = [*value]
        else:
            if isinstance(value, Dict):
                self.metric.update(value)
            else:
                self.metric[key] = value

    def compute_mean(self, key, value, axis=0):
        """
        计算并返回当前的指标的均值。
        """
        value_mean = None
        if isinstance(value, torch.Tensor):
            value_mean = torch.mean(value).detach().item()
        elif isinstance(value, np.ndarray):
            value_mean = np.mean(value, axis=axis)
        elif isinstance(value, list):
            # 过滤非数值元素
            filtered_data = [x for x in value if isinstance(x, (int, float))]
            value_mean = sum(filtered_data) / len(filtered_data)
        elif isinstance(value, tuple):
            value_mean = sum(value) / len(value)
        elif isinstance(value, dict):
            value_mean = sum(value.values()) / len(value)

        return value_mean

    def compute_sum(self, value, axis=0):
        """
        计算并返回当前的指标的和。
        """
        value_sum = None
        if isinstance(value, torch.Tensor):
            value_sum = torch.sum(value).detach().item()
        elif isinstance(value, np.ndarray):
            value_sum = np.sum(value, axis=axis)
        elif isinstance(value, list):
            # 过滤非数值元素
            filtered_data = [x for x in value if isinstance(x, (int, float))]
            value_sum = sum(filtered_data)
        elif isinstance(value, tuple):
            value_sum = sum(value)
        elif isinstance(value, dict):
            value_sum = sum(value.values())

        return value_sum

    def compute_max(self, key, value, axis=0):
        """
        计算并返回当前的指标的最大值。
        """
        value_max = None
        if isinstance(value, torch.Tensor):
            value_max = torch.max(value).detach().item()
        elif isinstance(value, np.ndarray):
            value_max = np.max(value, axis=axis)
        elif isinstance(value, list):
            # 过滤非数值元素
            filtered_data = [x for x in value if isinstance(x, (int, float))]
            value_max = max(filtered_data)
        elif isinstance(value, tuple):
            value_max = max(value)
        elif isinstance(value, dict):
            value_max = max(value.values())

        return value_max

    def compute_min(self, key, value, axis=0):
        """
        计算并返回当前的指标的最小值。
        """
        value_min = None
        if isinstance(value, torch.Tensor):
            value_min = torch.min(value).detach().item()
        elif isinstance(value, np.ndarray):
            value_min = np.min(value, axis=axis)
        elif isinstance(value, list):
            # 过滤非数值元素
            filtered_data = [x for x in value if isinstance(x, (int, float))]
            value_min = min(filtered_data)
        elif isinstance(value, tuple):
            value_min = min(value)
        elif isinstance(value, dict):
            value_min = min(value.values())

        return value_min

    def remove_key(self, key):
        """
        Remove the given key from the metric dictionary if it exists.
        key: str
        """
        if key in self.metric:
            del self.metric[key]
        else:
            print(f"Key '{key}' not found in metrics.")

    def remove_grpo_keys(self):
        """
        Remove grpo keys from the metric dictionary if it exists.
        """
        self.metric = {k: v for k, v in self.metric.items() if not k.startswith('grpo')}

    def reset(self):
        """
        重置指标状态。
        """
        self.metric = {}
