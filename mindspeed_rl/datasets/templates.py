# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import re
import json
from dataclasses import dataclass
from copy import deepcopy
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from typing_extensions import override

from mindspeed_rl.datasets.formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .formatter import Formatter


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def infer_max_len(source_len: int, target_len: int, max_len: int, reserved_label_len: int) -> Tuple[int, int]:
    if source_len + target_len == 0:
        max_target_len = 0
    else:
        max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - min(max_target_len, target_len)
    return max_source_len, max_target_len


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    format_prefix: "Formatter"
    thought_words: tuple[str, str]
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    force_system: bool
    enable_thinking: bool
    thought_words: tuple[str, str]

    def encode_oneturn(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: List[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            cutoff_len: int = 1_000_000,
            reserved_label_len: int = 1,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_pairs = self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids += query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        answer_ids = encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: List[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            cutoff_len: int = 1_000_000,
            reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        return self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)

    def _encode(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: List[Dict[str, str]],
            system: str,
            tools: str,
            cutoff_len: int,
            reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))
            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)

    def _convert_elements_to_ids(
            self, tokenizer: "PreTrainedTokenizer", elements: List[Union[str, Dict[str, str]]]
    ) -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids

    def _make_pairs(
            self,
            encoded_messages: Sequence[List[int]],
            cutoff_len: int,
            reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            max_source_len, max_target_len = infer_max_len(
                source_len=len(encoded_messages[i]),
                target_len=len(encoded_messages[i + 1]),
                max_len=(cutoff_len - total_length),
                reserved_label_len=reserved_label_len,
            )
            source_ids = encoded_messages[i][:max_source_len]
            target_ids = encoded_messages[i + 1][:max_target_len]
            total_length += len(source_ids) + len(target_ids)
            encoded_pairs.append((source_ids, target_ids))

        return encoded_pairs

    def add_thought(self, content: str = "") -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}\n\n{self.thought_words[1]}\n\n" + content

    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)


@dataclass
class Llama2Template(Template):
    def _encode(
            self,
            tokenizer: "PreTrainedTokenizer",
            messages: List[Dict[str, str]],
            system: str,
            tools: str,
            cutoff_len: int,
            reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)


@dataclass
class ReasoningTemplate(Template):
    r"""A template that add thought to assistant message."""

    @override
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        for i in range(1, len(messages) - 2, 2):
            messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:  # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if (
            self.thought_words[0] not in messages[-1]["content"]
            and self.thought_words[1] not in messages[-1]["content"]
        ):  # add empty cot
            if not self.enable_thinking:  # do not compute loss
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:  # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)[0]
        encoded_messages = [list(mes) for mes in encoded_messages]
        for i in range(0, len(messages), 2):
            if (
                self.thought_words[0] not in messages[i + 1]["content"]
                and self.thought_words[1] not in messages[i + 1]["content"]
            ):  # add empty cot
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]

        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]


templates: Dict[str, Template] = {}


TEMPLATES_MAPPING = {
    'ReasoningTemplate': ReasoningTemplate
}


def get_templates() -> Dict[str, Template]:
    return templates


def get_model_template(name, prompt_type_path, enable_thinking=False):
    name = register_custom_template(name, prompt_type_path, enable_thinking)
    if name is None:
        template = templates["empty"]  # placeholder
    else:
        template = get_templates().get(name, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(name))
    return template


def _register_template(
        name: str,
        format_user: Optional["Formatter"] = None,
        format_assistant: Optional["Formatter"] = None,
        format_system: Optional["Formatter"] = None,
        format_function: Optional["Formatter"] = None,
        format_observation: Optional["Formatter"] = None,
        format_tools: Optional["Formatter"] = None,
        format_separator: Optional["Formatter"] = None,
        format_prefix: Optional["Formatter"] = None,
        default_system: str = "",
        stop_words: List[str] = [],
        efficient_eos: bool = False,
        replace_eos: bool = False,
        force_system: bool = False,
        thought_words: Optional[tuple[str, str]] = None,
        enable_thinking: bool = False,
        template_class: str = Template
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=["Action: {{name}}\nAction Input: {{arguments}}"] + eos_slots)
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    template_class = TEMPLATES_MAPPING.get(template_class, None)
    if template_class is None:
        template_class = Llama2Template if name.startswith("llama2") else Template
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words,
        thought_words=thought_words or ("<think>", "</think>"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
        enable_thinking=enable_thinking
    )


def register_custom_template(name, json_file_path=None, enable_thinking=False) -> str:
    if name in templates:
        return name

    if not bool(re.match(r'(?:(?:/|\.{1,2}/|[^/\0]+/)(?:[^/\0]+/)*[^/\0]*|\.{1,2})', json_file_path)):
        raise ValueError(f"Invalid Path: {json_file_path}, please provide a valid custom template path.")

    with open(json_file_path, 'r') as file:
        config = json.load(file)

    templates_dict = {template['name']: template for template in config}
    config = templates_dict.get(name, None)

    if not config:
        raise ValueError(
            f"Can't find the template. Please provide a valid prompt type template in the {json_file_path}.")

    format_user = _format_custom_template(config.get("format_user", None))
    format_assistant = _format_custom_template(config.get("format_assistant", None))
    format_system = _format_custom_template(config.get("format_system", None))
    format_function = _format_custom_template(config.get("format_function", None))
    format_observation = _format_custom_template(config.get("format_observation", None))
    format_tools = _format_custom_template(config.get("format_tools", None))
    format_separator = _format_custom_template(config.get("format_separator", None))
    format_prefix = _format_custom_template(config.get("format_prefix", None))
    default_system = _format_custom_template(config.get("default_system", ""))
    stop_words = _format_custom_template(config.get("stop_words", []))
    efficient_eos = _format_custom_template(config.get("efficient_eos", False))
    replace_eos = _format_custom_template(config.get("replace_eos", False))
    force_system = _format_custom_template(config.get("force_system", False))
    template_class = _format_custom_template(config.get("template_class", False))

    if isinstance(default_system, list):
        default_system = "".join(default_system) if all(
            isinstance(sentence, str) for sentence in default_system) else default_system
    format_user = StringFormatter(**format_user) if format_user else None
    format_assistant = StringFormatter(**format_assistant) if format_assistant else None
    format_system = StringFormatter(**format_system) if format_system else None
    format_function = FunctionFormatter(**format_function) if format_function else None
    format_observation = StringFormatter(**format_observation) if format_observation else None
    format_tools = ToolFormatter(**format_tools) if format_tools else None
    format_separator = EmptyFormatter(**format_separator) if format_separator else None
    format_prefix = EmptyFormatter(**format_prefix) if format_prefix else None

    _register_template(
        name=name,
        format_user=format_user,
        format_assistant=format_assistant,
        format_system=format_system,
        format_function=format_function,
        format_observation=format_observation,
        format_tools=format_tools,
        format_separator=format_separator,
        format_prefix=format_prefix,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
        enable_thinking=enable_thinking,
        template_class=template_class
    )

    return name


def _format_custom_template(slots: Dict) -> Dict:
    if slots and isinstance(slots, Dict):
        for key, slot in slots.items():
            if key != 'tool_format':
                slots[key] = list(map(lambda slot: set(slot) if isinstance(slot, list) else slot, slot)) if slot else None
    return slots
