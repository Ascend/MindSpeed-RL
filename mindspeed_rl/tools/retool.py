import re
from typing import Any
import ray

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.tools.schemas import OpenAIFunctionToolSchema
from mindspeed_rl.tools.sandbox_fusion_tools import SandboxFusionTool

logger = Loggers("ReTool")


class ReTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = ray.get(self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language))
        return result, 0, {}

