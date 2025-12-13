# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import json
import pytest
import numpy as np

from mindspeed_rl.datasets.formatter import (
    EmptyFormatter,
    StringFormatter,
    FunctionFormatter,
    ToolFormatter,
    default_tool_formatter,
    default_tool_extractor
)

from tests.test_tools.dist_test import DistributedTest


class TestFormatter(DistributedTest):
    world_size = 1
    is_dist_test = False

    def test_empty_formatter(self):
        formatter = EmptyFormatter(slots=["Hello", "World"])
        result = formatter.apply()
        assert result == ["Hello", "World"]
        
        formatter = EmptyFormatter(slots=["Hello", {"key": "value"}, {"set"}])
        result = formatter.apply()
        assert result == ["Hello", {"key": "value"}, {"set"}]
        
        with pytest.raises(ValueError):
            EmptyFormatter(slots=["Hello {{name}}"])

    def test_string_formatter(self):
        formatter = StringFormatter(slots=["Hello {{name}}", "You are {{age}} years old"])
        result = formatter.apply(name="Alice", age="25")
        assert result == ["Hello Alice", "You are 25 years old"]
        
        formatter = StringFormatter(slots=["Hello {{name}}", {"key": "value"}])
        result = formatter.apply(name="Bob")
        assert result == ["Hello Bob", {"key": "value"}]
        
        with pytest.raises(ValueError):
            StringFormatter(slots=["Hello", "World"])

    def test_function_formatter(self):
        formatter = FunctionFormatter(slots=["Function: {{name}}", "Args: {{arguments}}"])
        
        function_content = '{"name": "add", "arguments": {"a": 1, "b": 2}}'
        result = formatter.apply(content=function_content)
        assert result == ["Function: add", "Args: {\"a\": 1, \"b\": 2}"]
        
        result = formatter.apply(content="invalid json")
        assert result == ["Function: ", "Args: "]

    def test_tool_formatter(self):
        formatter = ToolFormatter(tool_format="default")
        
        tools_content = '[]'
        result = formatter.apply(content=tools_content)
        assert result == [""]
        
        tools_content = '[{"name": "calculator", "description": "Performs calculations", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression"}}, "required": ["expression"]}}]'
        result = formatter.apply(content=tools_content)
        assert "Tool Name: calculator" in result[0]
        assert "Tool Description: Performs calculations" in result[0]
        
        result = formatter.apply(content="invalid json")
        assert result == [""]

    def test_default_tool_formatter(self):
        tools = [
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        result = default_tool_formatter(tools)
        assert "Tool Name: search" in result
        assert "Tool Description: Search for information" in result
        assert "query (string, required)" in result

    def test_default_tool_extractor(self):
        content = "Some text Action: search Action Input: {\"query\": \"test\"}"
        result = default_tool_extractor(content)
        assert result == ("search", "{\"query\": \"test\"}")
        
        content = "Just some regular text without actions"
        result = default_tool_extractor(content)
        assert result == content
        
        content = "Action: search Action Input: {invalid json}"
        result = default_tool_extractor(content)
        assert result == content

    def test_tool_formatter_extract(self):
        formatter = ToolFormatter(tool_format="default")
        
        content = "Action: search Action Input: {\"query\": \"test\"}"
        result = formatter.extract(content)
        assert result == ("search", "{\"query\": \"test\"}")
        
        content = "Just some text"
        result = formatter.extract(content)
        assert result == content