import importlib
import sys
from omegaconf import OmegaConf

from mindspeed_rl.tools.utils.schemas import OpenAIFunctionToolSchema


def get_tool_class(cls_name):
    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    tool_cls = getattr(module, class_name)
    return tool_cls


def initialize_tools_from_config(tools_config_file):
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    for tool_config in tools_config.tools:
        cls_name = tool_config.class_name
        tool_cls = get_tool_class(cls_name)

        if tool_config.get("tool_schema", None) is None:
            tool_schema = None
        else:
            tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
            tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
        tool = tool_cls(
            config=OmegaConf.to_container(tool_config.config, resolve=True),
            tool_schema=tool_schema,
        )
        tool_list.append(tool)

    return tool_list
