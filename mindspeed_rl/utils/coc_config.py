import mindspeed
from mindspeed.core.tensor_parallel.coc_feature import min_comm_cfg
from mindspeed.core.tensor_parallel.lcal_coc import min_comm_cfg
from mindspeed.core.tensor_parallel.coc_feature.min_comm_cfg import ModuleType


def print_on_device0(msg):
    pass


def acquire_module_type(self, tp_size):
    """Determine and set the module type based on configuration."""
    sequence_parallel_types = [ModuleType.ORIGINAL_SEQ_PARALLEL,
                               ModuleType.REWRITE_SEQ_PARALLEL,
                               ModuleType.COC_FOR_SEQ_PARALLEL]
    all_reduce_types = [ModuleType.ORIGINAL_ALL_REDUCE,
                        ModuleType.REWRITE_ALL_REDUCE,
                        ModuleType.COC_FOR_ALL_REDUCE]

    if self.parallel_num not in [1, 2, 4, 8]:
        raise RuntimeError("coc_parallel_num must be either 1, 2, 4 or 8. Current value not supported")
    if self.coc_mode not in [-1, 0, 1, 2]:
        raise RuntimeError("coc_mode must be either 0, 1, or 2. Current value not supported")

    if self.coc_mode == -1:
        self.coc_mode = 0 if self.parallel_num == 1 else 2

    if tp_size == 1:
        self.coc_mode = 0
        self.parallel_num = 1

    if self.sequence_parallel_enabled:
        self.module_type = sequence_parallel_types[self.coc_mode]
    else:
        self.module_type = all_reduce_types[self.coc_mode]

    if "COC" in self.module_type.name:
        self.prefix = f"module_{self.module_type.name}_parallel_num_{self.parallel_num}"
    else:
        self.prefix = f"module_{self.module_type.name}"

mindspeed.core.tensor_parallel.coc_feature.min_comm_cfg.print_on_device0 = print_on_device0
mindspeed.core.tensor_parallel.coc_feature.min_comm_cfg.MinCommConfig.acquire_module_type = acquire_module_type
mindspeed.core.tensor_parallel.lcal_coc.min_comm_cfg.MinCommConfig.acquire_module_type = acquire_module_type