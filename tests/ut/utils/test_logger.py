# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import unittest
import logging


from mindspeed_rl.utils.loggers import Loggers


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Loggers(
            name="TestLogger",
            logger_level=logging.DEBUG
        )

    def test_log_levels(self):
        """测试不同日志等级的输出"""
        with self.assertLogs(logger="TestLogger", level="INFO") as cm:
            self.logger.debug("Debug message")
            self.logger.info("Info message")
            self.logger.warning("Warning message")
            self.logger.error("Error message")

        # 验证输出的日志数量，DEBUG等级低于INFO，所以不会输出
        self.assertEqual(len(cm.output), 3)
        # 验证日志等级
        self.assertFalse(any("DEBUG" in record for record in cm.output))
        self.assertTrue(any("INFO" in record for record in cm.output))
        self.assertTrue(any("WARNING" in record for record in cm.output))
        self.assertTrue(any("ERROR" in record for record in cm.output))

    def test_log_format(self):
        """测试日志格式"""
        with self.assertLogs(logger="TestLogger", level=logging.INFO) as cm:
            self.logger.info("Format test")

        # 验证格式
        log_record = cm.output[0]
        self.assertRegex(log_record, r".*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*")  # 时间戳
        self.assertIn("INFO", log_record)
        self.assertIn("Format test", log_record)


#python -m unittest ut_logger.py
