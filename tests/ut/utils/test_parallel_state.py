# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from unittest.mock import MagicMock
import pytest

from mindspeed_rl.trainer.utils.parallel_state import get_pipeline_model_parallel_rank

from tests.test_tools.dist_test import DistributedTest


class TestParallelState(DistributedTest):
    world_size = 1
    is_dist_test = False

    def test_get_pipeline_model_parallel_rank(self):
        mpu = MagicMock()
        mpu.get_pipeline_model_parallel_rank.return_value = 1

        result = get_pipeline_model_parallel_rank(mpu, use_vllm=False)
        assert result == 1, "Parallel state method failed!"
