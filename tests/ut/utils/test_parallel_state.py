# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import unittest
from unittest.mock import MagicMock

import torch

from mindspeed_rl.trainer.utils.parallel_state import get_pipeline_model_parallel_rank


class TestParallelState(unittest.TestCase):
    def test_get_pipeline_model_parallel_rank(self):
        mpu = MagicMock()
        mpu.get_pipeline_model_parallel_rank.return_value = 1

        result = get_pipeline_model_parallel_rank(mpu, use_vllm=False)
        self.assertEqual(result, 1)
