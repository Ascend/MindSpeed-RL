# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional, Any
from torch.utils.data import Dataset
import datasets


class BaseDataset(Dataset, ABC):
    """
    A base dataset, providing basic parameters and methods for datasets.
    """

    def __init__(self, dataset: Any, dataset_type: str = "basetype", extra_param: Optional[Any] = None):
        if not isinstance(dataset, (Dataset, datasets.arrow_dataset.Dataset)):
            raise TypeError(f"Expected dataset to be of type Dataset, got {type(dataset)}.")

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.extra_param = extra_param

    def __len__(self) -> int:
        return len(self.dataset)

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        This method should be implemented by subclasses to retrieve a data item at the given index.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_type(self) -> str:
        return self.dataset_type
