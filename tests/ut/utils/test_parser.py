# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from mindspeed_rl.utils.math_eval_toolkit.parser import extract_answer, choice_answer_clean
from tests.test_tools.dist_test import DistributedTest


class TestLogger(DistributedTest):

    def test_log_format(self):
        """测试日志格式"""
        text = """
        This is still not equal to $0$, so we must have made another mistake.

        When we subtracted $7$ from $\frac{386}{64}$, we should have subtracted $7 \cdot 64$ from $386$, not the other way around. Let's correct that:

        \[\frac{386}{64} - 7 = \frac{386}{64} - \frac{7 \cdot 64}{1 \cdot 64} = \frac{386 - 448}{64} = \frac{-62}{64}.\]

        This is still not equal to $0$, so we must have made another mistake.

        When we subtracted $7$ from $\frac{386}{64}$, we should have subtracted $7 \cdot 64$ from $386$, not the other way around. Let's correct that:

        \[\frac{386}{64}
        """
        res = extract_answer(text, "math-oai", use_last_number=False)
        assert res == "", "extract answer failed"
        assert choice_answer_clean("\mathrm{(D) }1,008,016") == "D", "choice answer failed"
