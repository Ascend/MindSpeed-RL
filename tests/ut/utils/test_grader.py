# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from mindspeed_rl.utils.math_eval_toolkit.grader import math_equal
from tests.test_tools.dist_test import DistributedTest


class TestGrader(DistributedTest):

    def test_math_equal(self):
        """测试日志格式"""
        gt = "x+2n+1"
        pred = "x+1"
        res = math_equal(pred, gt, timeout=True)
        assert res is False, "math equal failed"
        assert math_equal("x+1", "x+1", timeout=True) is True
        assert math_equal("x+1", "x+2", timeout=True) is False
        assert math_equal("1", "1.0", timeout=True) is True
        assert math_equal("1", "1.0001", timeout=True) is True
        assert math_equal("1", "1.1", timeout=True) is False
