# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import logging
import re
import pytest

from mindspeed_rl.utils.loggers import Loggers

from tests.test_tools.dist_test import DistributedTest


class TestLogger(DistributedTest):
    world_size = 1

    @pytest.fixture(autouse=True)
    def setup_logger(self):
        """Fixture to set up the logger for each test."""
        self.logger = Loggers(
            name="test_logger",
            logger_level=logging.DEBUG
        )

    def test_log_levels(self, caplog):
        """测试不同日志等级的输出"""
        self.logger.logger.propagate = True
        caplog.set_level(logging.INFO)
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")

        # 验证输出的日志数量，DEBUG等级低于INFO，所以不会输出
        assert len(caplog.records) == 3, "Expected 3 log records, but got {}".format(len(caplog.records))

        # 验证日志等级
        assert not any(record.levelname == "DEBUG" for record in caplog.records), "DEBUG log should not be present."
        assert any(record.levelname == "INFO" for record in caplog.records), "INFO log should be present."
        assert any(record.levelname == "WARNING" for record in caplog.records), "WARNING log should be present."
        assert any(record.levelname == "ERROR" for record in caplog.records), "ERROR log should be present."

    def test_log_format(self, caplog):
        """测试日志格式"""
        self.logger.logger.propagate = True
        caplog.set_level(logging.INFO)
        self.logger.info("Format test")

        # 验证日志格式
        log_record = caplog.records[0]
        assert log_record.levelname == "INFO", "Log level should be INFO"
        assert "Format test" in log_record.message, "Log message should contain 'Format test'"
        # 手动验证时间戳
        timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        assert re.search(timestamp_pattern, log_record.message), \
        f"Log message should contain a timestamp matching pattern '{timestamp_pattern}'"
