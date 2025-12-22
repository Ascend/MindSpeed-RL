import json
from typing import Any, Optional
from uuid import uuid4
import ray

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.tools.base_tool import BaseTool
from mindspeed_rl.tools.utils.schemas import OpenAIFunctionToolSchema
from mindspeed_rl.tools.utils.tool_utils import process_single_search_batch
from mindspeed_rl.tools.utils.execution_pool import init_execution_pool


logger = Loggers("SearchTool")


class SearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
        )
        self.retrieval_service_url = config.get("retrieval_service_url")
        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")
        self.topk = config.get("topk", 3)
        logger.info(f"Initialized SearchTool with config: {config}")

    def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id

    def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_str = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_str} Received parameters: {parameters}")
            return json.dumps({"result": error_str}), 0.0, {}

        try:
            result_text, metadata = ray.get(self.execution_pool.execute.remote(
                self.execute_search, instance_id, query_list_from_params, self.retrieval_service_url, self.topk, timeout
            ))

            self._instance_dict[instance_id]["reward"].append(result_text.strip())
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }

            return result_text, 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        result_text, metadata = process_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
