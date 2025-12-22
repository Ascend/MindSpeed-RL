import json
from typing import Any, Optional
from uuid import uuid4

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.tools.utils.schemas import OpenAIFunctionToolSchema


logger = Loggers("BaseTool")


class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        if self.tool_schema is None:
            raise ValueError("tool_schema is not set")
        self.name = self.tool_schema.function.name
        print(json.dumps(self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        return "not implement", 0.0, {}

    def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    def release(self, instance_id: str, **kwargs) -> None:
        pass
