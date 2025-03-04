# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
"""Just an initialize test"""

from tests.test_tools.dist_test import DistributedTest


class TestMock(DistributedTest):
    world_size = 1

    def test_mock_op(self):
        assert 1 + 1 == 2, "Failed !"
