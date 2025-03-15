# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from mindspeed_rl.models.rule_verifier import math_equal_subprocess, format_reward, strict_format_reward, reasoning_steps_reward
from tests.test_tools.dist_test import DistributedTest


class TestVerifier(DistributedTest):

    def test_verifier(self):
        """测试日志格式"""
        text = ["""<think>\nFirst, we use the property of a parallelogram that opposite sides are equal in length. 
        Therefore, we have:\n\\[AB = CD \\quad \\text{and} \\quad BC = AD\\]\n\nFrom the given measurements:\n\\
        [AB = 38 \\text{ cm}\\]\n\\[BC = 3y^3 \\text{ cm}\\]\n\\[CD = 2x + 4 \\text{ cm}\\]\n\\[AD = 24 \\text{ cm}\\]\n\n
        Setting \\(AB = CD\\), we get:\n\\[38 = 2x + 4\\]\n\nSetting \\(BC = AD\\), we get:\n\\[3y^3 = 24\\]\n\nNow, 
        we solve each equation for \\(x\\) and \\(y\\).\n</think>\n<answer>\nFirst, solve for \\(x\\):\n\\[38 = 2x + 4\\]
        \n\\[38 - 4 = 2x\\]\n\\[34 = 2x\\]\n\\[x = \\frac{34}{2} = 17\\]\n\nNext, solve for \\(y\\):\n\\[3y^3 = 24\\]\n\\
        [y^3 = \\frac{24}{3} = 8\\]\n\\[y = \\sqrt[3]{8} = 2\\]\n\nNow, find the product of \\(x\\) and \\(y\\):\n\\[xy = 
        17 \\times 2 = 34\\]\n\nTherefore, the product of \\(x\\) and \\(y\\) is \\(\\boxed{34}\\).</answer>"""]

        label = ['34']
        model_output = ['34']

        assert math_equal_subprocess(model_output[0], label[0]), "math_equal_subprocess failed"
        assert format_reward(None, text) == [1.0], "format_reward failed"
        assert strict_format_reward(None, text) == [1.0], "strict_format_verifier failed"
        assert reasoning_steps_reward(None, text) == [1.0], "strict_format_verifier failed"

