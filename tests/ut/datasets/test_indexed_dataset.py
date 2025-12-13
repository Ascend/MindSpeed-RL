# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import os
import tempfile
import pytest
import numpy as np
import torch

from mindspeed_rl.datasets.indexed_dataset import (
    get_packed_indexed_dataset,
    IndexedDataset,
    CombinedDataset,
    get_idx_path,
    get_bin_path,
    DType,
    IndexedDatasetBuilder,
    _IndexReader,
    _MMapBinReader,
    _FileBinReader,
    BufferWriter
)

from tests.test_tools.dist_test import DistributedTest


class TestIndexedDataset(DistributedTest):
    world_size = 1
    is_dist_test = False

    def test_get_idx_bin_path(self):
        prefix = "/tmp/test_data"
        idx_path = get_idx_path(prefix)
        bin_path = get_bin_path(prefix)
        
        assert idx_path == "/tmp/test_data.idx"
        assert bin_path == "/tmp/test_data.bin"

    def test_dtype_class(self):
        assert DType.code_from_dtype(np.uint8) == 1
        assert DType.code_from_dtype(np.int32) == 4
        
        assert DType.dtype_from_code(1) == np.uint8
        assert DType.dtype_from_code(4) == np.int32
        
        assert DType.size(np.int32) == 4
        assert DType.size(4) == 4
        
        assert DType.optimal_dtype(1000) == np.uint16
        assert DType.optimal_dtype(100000) == np.int32

    def test_buffer_writer(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, 'wb') as f:
                writer = BufferWriter(f, np.int32, buffer_chunk_size=10)
                writer.add([1, 2, 3, 4, 5])
                writer.write()
                f.flush()

            with open(tmp_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.int32)
                np.testing.assert_array_equal(data, [1, 2, 3, 4, 5])

            with open(tmp_path, 'wb') as f:
                writer = BufferWriter(f, np.int32, buffer_chunk_size=3)
                writer.add([1, 2])
                writer.add([3, 4, 5])
                writer.write()
                f.flush()
            
            with open(tmp_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.int32)
                np.testing.assert_array_equal(data, [1, 2, 3, 4, 5])
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_bin_readers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = os.path.join(tmpdir, "test.bin")
            idx_path = os.path.join(tmpdir, "test.idx")
            
            builder = IndexedDatasetBuilder(bin_path, dtype=np.int32)
            builder.add_item(torch.tensor([1, 2, 3]))
            builder.add_item(torch.tensor([4, 5]))
            builder.finalize(idx_path)
            
            mmap_reader = _MMapBinReader(bin_path)
            data = mmap_reader.read(np.int32, 3, 0)
            np.testing.assert_array_equal(data, [1, 2, 3])
            
            data = mmap_reader.read(np.int32, 2, 12)
            np.testing.assert_array_equal(data, [4, 5])
            
            file_reader = _FileBinReader(bin_path)
            data = file_reader.read(np.int32, 3, 0)
            np.testing.assert_array_equal(data, [1, 2, 3])
            
            data = file_reader.read(np.int32, 2, 12)
            np.testing.assert_array_equal(data, [4, 5])
