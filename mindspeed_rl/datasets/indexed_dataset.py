# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os

import struct
import glob
import re

from enum import Enum
from functools import lru_cache
from abc import ABC, abstractmethod
from itertools import accumulate
from typing import Optional, Tuple, Type, Union

import torch
import numpy

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def get_packed_indexed_dataset(data_prefix: str):
    index_dataset_name = f"{data_prefix}_packed_*_document*"
    names = glob.glob(index_dataset_name)
    template = f"{data_prefix}_packed_(.*)_document(.*)"
    all_field = set()
    for name in names:
        fields = re.match(template, name)
        all_field.add(fields.group(1))
    packed_dataset = dict()
    for field in all_field:
        packed_dataset[field] = IndexedDataset(f"{data_prefix}_packed_{field}_document")

    combine_dataset = CombinedDataset(packed_dataset)
    return combine_dataset


class IndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool): Whether the dataset is multimodal. Defaults to False.

        mmap (bool): Whether to mmap the .bin files. Defaults to True.
    """

    def __init__(
        self,
        path_prefix: str,
        multimodal: bool = False,
        mmap: bool = True,
    ) -> None:
        super().__init__()
        self.path_prefix = None
        self.multimodal = None
        self.mmap = None

        self.index = None
        self.bin_reader = None

        self.initialize(path_prefix, multimodal, mmap)

    def initialize(
        self, path_prefix: str, multimodal: bool, mmap: bool
    ) -> None:
        """Initialize the dataset

        This method is called by IndexedDataset.__init__ during object creation and by
        IndexedDataset.__setstate__ during un-pickling

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix

            multimodal (bool): Whether the dataset is multimodal

            mmap (bool): Whether to mmap the .bin file
        """
        idx_path = get_idx_path(path_prefix)
        bin_path = get_bin_path(path_prefix)

        if not os.path.exists(idx_path) or not os.path.exists(bin_path):
            raise ValueError("One or both of the .idx and .bin files "
                             "cannot be found at the path prefix {}".format(path_prefix))

        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.mmap = mmap

        if mmap:
            self.bin_reader = _MMapBinReader(bin_path)
        else:
            self.bin_reader = _FileBinReader(bin_path)

        self.index = _IndexReader(idx_path, self.multimodal)

    def __getstate__(self) -> Tuple[str, bool, bool]:
        """Get the state during pickling

        Returns:
            Tuple[str, bool, bool]: The state tuple
        """
        return self.path_prefix, self.multimodal, self.mmap

    def __setstate__(self, state: Tuple[str, bool, bool]) -> None:
        """Set the state during un-pickling

        Args:
            state (Tuple[str, bool, bool]): The state tuple
        """
        path_prefix, multimodal, mmap = state
        self.initialize(path_prefix, multimodal, mmap)

    def __del__(self) -> None:
        """Clean up the object"""
        del self.bin_reader
        del self.index

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """Return from the dataset

        Args:
            idx (Union[int, numpy.integer, slice]): The index or index slice into the dataset

        Raises:
            ValueError: When the index slice is non-contiguous

            TypeError: When the index is of an unexpected type

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and modes at the index or index slice
        """
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype,
                count=sequence_length,
                offset=sequence_pointer,
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_modes = self.index.sequence_modes[idx] if self.multimodal else None
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = numpy.split(
                self.bin_reader.read(
                    dtype=self.index.dtype,
                    count=sum(sequence_lengths),
                    offset=self.index.sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (sequences, sequence_modes) if sequence_modes is not None else sequences
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.

        Args:
            idx (Union[int, numpy.integer]): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (int): The number of tokens to grab from the sequence

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: The sequence tokens and modes at the index
        """
        sequence_pointer, sequence_length, sequence_mode = self.index[idx]
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        sequence = self.bin_reader.read(
            dtype=self.index.dtype, count=length, offset=sequence_pointer
        )
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        """Get the sequence lengths

        Returns:
            numpy.ndarray: The sequence lengths
        """
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> numpy.ndarray:
        """Get the document indices

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def get_document_indices(self) -> numpy.ndarray:
        """Get the document indices

        This method is slated for deprecation.

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def set_document_indices(self, document_indices: numpy.ndarray) -> None:
        """Set the document indices

        This method is slated for deprecation.

        Args:
            document_indices (numpy.ndarray): The document indices
        """
        self.index.document_indices = document_indices

    @property
    def sequence_modes(self) -> numpy.ndarray:
        """Get the sequence modes

        Returns:
            numpy.ndarray: The sequence modes
        """
        return self.index.sequence_modes

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Return whether the IndexedDataset exists on disk at the prefix

        Args:
            path_prefix (str): The prefix to the index (.idx) and data (.bin) files

        Returns:
            bool: Whether the IndexedDataset exists on disk at the prefix
        """
        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(
            get_bin_path(path_prefix)
        )


class CombinedDataset(torch.utils.data.Dataset):
    """
    A dataset that combines multiple datasets and returns merged data on __getitem__.
    """

    def __init__(self, datasets: dict):
        # check input dataset info
        self.length = len(list(datasets.values())[0])
        for dataset in datasets.values():
            if len(dataset) != self.length:
                raise Exception("Dimension is not correct !")
        self.datasets = datasets

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        packed_data = dict()
        for key, dataset in self.datasets.items():
            packed_data[key] = dataset.get(idx)
        return packed_data


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the index file
    """
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the data file
    """
    return path_prefix + ".bin"


class _BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        raise NotImplementedError("read is not implemented.")


class _MMapBinReader(_BinReader):
    """A _BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_buffer_mmap = numpy.memmap(bin_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        return numpy.frombuffer(
            self._bin_buffer,
            dtype=dtype,
            count=count,
            offset=offset,
        )

    def __del__(self) -> None:
        """Clean up the object."""
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap


class _FileBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using a file pointer

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_path = bin_path

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        sequence = numpy.empty(count, dtype=dtype)
        with open(self._bin_path, mode='rb', buffering=0) as bin_buffer_file:
            bin_buffer_file.seek(offset)
            bin_buffer_file.readinto(sequence)
        return sequence


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        """Get the code from the dtype

        Args:
            value (Type[numpy.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            Type[numpy.number]: The dtype
        """
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, Type[numpy.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[numpy.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            Type[numpy.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


class _IndexReader(object):
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        multimodal (bool): Whether the dataset is multimodal
    """

    def __init__(self, idx_path: str, multimodal: bool) -> None:

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            if header != _INDEX_HEADER:
                raise ValueError("bad header, cannot read: {}".format(idx_path))

            version = struct.unpack("<Q", stream.read(8))[0]
            if version != 1:
                raise ValueError("bad version, cannot read: {}".format(idx_path))

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )

        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )

        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

        self.sequence_modes = None
        if multimodal:
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )

        if self.sequence_lengths.shape[0] != len(self) or self.sequence_lengths.shape[0] != self.sequence_count \
                or self.sequence_lengths.shape[0] != self.document_indices[-1]:
            raise ValueError("sequence_lengths is error")

    def __del__(self) -> None:
        """Clean up the object"""
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]: The pointer, length and mode at the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )
