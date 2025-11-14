import threading
from contextlib import ExitStack
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
import ray

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.tools.base_tool import BaseTool
from mindspeed_rl.tools.schemas import OpenAIFunctionToolSchema
from mindspeed_rl.tools.tool_utils import process_single_case

logger = Loggers("SandboxFusionTool")

T = TypeVar("T")


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                logger.warning(f"Error when executing code: {e}")
                return ''



def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10
):
    return (
        ray.remote(ExecutionWorker)
        .options(max_concurrency=num_workers)
        .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
    )


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

    def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
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
        del self._instance_dict[instance_id]
