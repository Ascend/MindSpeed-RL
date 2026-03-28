# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, HUAWEI CORPORATION.
#
# This file is derived from code originally developed by Bytedance Ltd.
# Modifications have been made by HUAWEI CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from abc import ABC, abstractmethod

import regex as re

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.tools.utils.schemas import OpenAIFunctionParsedSchema


logger = Loggers("ToolParser")


class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def extract_tool_calls(self, responses_ids: list[int]) -> list[OpenAIFunctionParsedSchema]:
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass

        return decorator


@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def extract_tool_calls(self, responses_ids: list[int]) -> list[OpenAIFunctionParsedSchema]:
        text = self.tokenizer.decode(responses_ids)
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(OpenAIFunctionParsedSchema(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")

        return function_calls
