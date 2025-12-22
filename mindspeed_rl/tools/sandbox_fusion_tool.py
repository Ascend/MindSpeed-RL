from typing import Any, Optional
from uuid import uuid4
import ray

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.tools.base_tool import BaseTool
from mindspeed_rl.tools.utils.schemas import OpenAIFunctionToolSchema
from mindspeed_rl.tools.utils.tool_utils import process_single_case
from mindspeed_rl.tools.utils.execution_pool import init_execution_pool


logger = Loggers("SandboxFusionTool")


class SandboxFusionTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
        )
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        logger.info(f"Init SandboxFusionTool with config: {config}")

    def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id

    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = ray.get(self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language))
        return result, 0, {}

    def execute_code(self, instance_id, code, timeout=30, language="python"):
        result_status, metadata = process_single_case(
            0, None, None, self.sandbox_fusion_url, code, timeout, self.memory_limit_mb, language
        )
        if metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] + metadata["stderr"]
            logger.debug(f"actual_output from sandbox fusion: {actual_output},{instance_id}")
            return actual_output
        else:
            return "no stdout here"

    def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
