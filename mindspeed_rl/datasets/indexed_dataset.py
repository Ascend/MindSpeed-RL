# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import shutil
import struct
import glob
import re
from enum import Enum
from functools import lru_cache
from abc import ABC, abstractmethod
from itertools import accumulate
from types import TracebackType
from typing import Optional, Tuple, Type, Union, List

import torch
import numpy

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def get_packed_indexed_dataset(data_prefix: str, filter_length: Optional[int] = None, is_pairwise_dataset: bool = False):
    index_dataset_name = f"{data_prefix}_packed_*_document*"
    names = glob.glob(index_dataset_name)
    template = f"{data_prefix}_packed_(.*)_document(.*)"
    all_field = set()
    for name in names:
        fields = re.match(template, name)
        all_field.add(fields.group(1))
    packed_dataset = dict()

    for field in all_field:
        # We only do filter for input_ids when filter_length is specified
        max_len = filter_length if filter_length and field == 'input_ids' else None
        packed_dataset[field] = IndexedDataset(f"{data_prefix}_packed_{field}_document", max_len=max_len)

    if filter_length and not is_pairwise_dataset:
        filter_mask = packed_dataset['input_ids'].get_filter_mask()
        for field in packed_dataset:
            packed_dataset[field].do_filter(filter_mask)

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
            max_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.path_prefix = None
        self.multimodal = None
        self.mmap = None

        self.index = None
        self.bin_reader = None

        self.max_len = max_len
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

        self.index = _IndexReader(idx_path, self.multimodal, self.max_len)

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

    def get_filter_mask(self):
        return self.index.filter_mask

    def do_filter(self, mask):
        self.index.do_filter(mask)

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

    def __init__(self, idx_path: str, multimodal: bool, max_len: Optional[int]) -> None:

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

        if max_len:
            length_mask = self.sequence_lengths < max_len
            self.sequence_lengths = self.sequence_lengths[length_mask]
            self.sequence_pointers = self.sequence_pointers[length_mask]
            self.sequence_count = len(self.sequence_lengths)

            self.sequence_modes = self.sequence_modes[length_mask] if self.sequence_modes else None
            # document_indices is not used in training, it is ok to bypass the following check
            self.document_indices = [self.sequence_lengths.shape[0]]

            self.filter_mask = length_mask

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

    def do_filter(self, mask):
        if hasattr(self, "filter_mask"):
            return

        self.sequence_lengths = self.sequence_lengths[mask]
        self.sequence_pointers = self.sequence_pointers[mask]
        self.sequence_count = len(self.sequence_lengths)
        self.sequence_modes = self.sequence_modes[mask] if self.sequence_modes else None

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


class BufferWriter:
    """
    Write the sequences in chunks rather than one by one
    """

    def __init__(self, data_file, dtype, buffer_chunk_size=10 ** 5):
        self.data_file = data_file
        self.dtype = dtype
        self.buffer_threshold = buffer_chunk_size
        self.buffer = []

    def reset_buffer(self):
        self.buffer = []

    def write(self):
        if self.buffer:
            buffer_array = numpy.array(self.buffer, dtype=self.dtype)
            self.data_file.write(buffer_array.tobytes(order="C"))
            self.reset_buffer()

    def add(self, lst: List):
        self.buffer.extend(lst)

        if len(self.buffer) >= self.buffer_threshold:
            self.write()


class IndexedDatasetBuilder(object):
    """Builder class for the IndexedDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (Type[numpy.number], optional): The dtype of the index file. Defaults to numpy.int32.

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(
            self, bin_path: str, dtype: Type[numpy.number] = numpy.int32, multimodal: bool = False
    ) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype
        self.multimodal = multimodal

        self.sequence_lengths = []
        self.document_indices = [0]
        self.sequence_modes = [] if self.multimodal else None
        self.buffer_writer = BufferWriter(data_file=self.data_file, dtype=self.dtype)

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        """Add a single item to the dataset

        Args:
            tensor (torch.Tensor): The item to add to the data file

            mode (int, optional): The mode for the item. Defaults to 0.
        """
        if isinstance(tensor, (list, List)):
            self.buffer_writer.add(tensor)
            self.sequence_lengths.append(len(tensor))
            if self.multimodal:
                self.sequence_modes.append(mode)
            return

        np_array = numpy.array(tensor.numpy(), dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        if self.multimodal:
            self.sequence_modes.append(mode)

    def add_document(
            self, tensor: torch.Tensor, lengths: List[int], modes: Optional[List[int]] = None
    ) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor): The document to add

            lengths (List[int]): The lengths of each item in the document

            modes (Optional[List[int]], optional): The modes for each item in the document. Defaults to None.
        """
        np_array = numpy.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))
        if self.multimodal:
            self.sequence_modes.extend(modes if modes is not None else [0] * lengths)

    def end_document(self) -> None:
        """Finalize the document, for use with IndexedDatasetBuilder.add_item"""
        self.document_indices.append(len(self.sequence_lengths))

    def add_index(self, path_prefix: str) -> None:
        """Add an entire IndexedDataset to the dataset

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix
        """
        # Concatenate index
        index = _IndexReader(get_idx_path(path_prefix), multimodal=self.multimodal)
        if index.dtype != self.dtype:
            raise ValueError("index.dtype shoule be equal with dtype.")

        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(index.sequence_lengths)
        self.document_indices.extend((offset + index.document_indices)[1:])

        if self.multimodal:
            self.sequence_modes.extend(index.sequence_modes)

        # Concatenate data
        with open(get_bin_path(path_prefix), "rb") as f:
            shutil.copyfileobj(f, self.data_file)

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.buffer_writer.write()
        self.data_file.close()
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.sequence_modes, self.document_indices)


class _IndexWriter(object):
    """Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (Type[numpy.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: Type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
    ):
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()

    def write(
            self,
            sequence_lengths: List[int],
            sequence_modes: Optional[List[int]],
            document_indices: List[int],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (List[int]): The length of each sequence

            sequence_modes (Optional[List[int]]): The mode of each sequences

            document_indices (List[int]): The seqyebce indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        sequence_lengths = numpy.array(sequence_lengths, dtype=numpy.int32)
        self.idx_writer.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = numpy.array(sequence_pointers, dtype=numpy.int64)
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = numpy.array(document_indices, dtype=numpy.int64)
        self.idx_writer.write(document_indices.tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            sequence_modes = numpy.array(sequence_modes, dtype=numpy.int8)
            self.idx_writer.write(sequence_modes.tobytes(order='C'))
            del sequence_modes

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            List[int]: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr
