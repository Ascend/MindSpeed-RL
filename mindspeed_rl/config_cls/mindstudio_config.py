# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from mindspeed_rl.config_cls.base_config import BaseConfig


class ProfilerConfig(BaseConfig):
    """Profiler configuration class for performance profiling.

    This class manages profiling parameters for performance analysis including
    profiling level, export options, and hardware-specific profiling settings.

    Attributes:
        role (str): Identifier for the profiler role.
        profile (bool): Enable/disable the profiler. Set to True to enable performance analysis.
        mstx (bool): Enable/disable lightweight collection mode. True for lightweight mode.
        stage (str): Profiling stage, options include "all", "actor_generate", 
            "actor_compute_log_prob", "actor_update", "reference_compute_log_prob".
        profile_save_path (str): Path where profiling data will be saved.
        profile_export_type (str): Export file format, options include "db" and "text".
        profile_level (str): Profiling level, options include "level0", "level1", "level2", "level_none".
        profile_with_memory (bool): Whether to analyze memory usage.
        profile_record_shapes (bool): Whether to record tensor shape information.
        profile_with_cpu (bool): Whether to analyze CPU profiling information.
        profile_with_npu (bool): Whether to analyze NPU profiling information.
        profile_with_module (bool): Whether to analyze with stack.
        profile_step_start (int): Step to start profiling.
        profile_step_end (int): Step to end profiling.
        profile_analysis (bool): Whether to analyze profile data online.
        profile_ranks (str): The ranks to be profiled, can be set to "all" for all ranks.
    """

    def __init__(self, config_dict, role=""):
        """Initialize ProfilerConfig with configuration dictionary and role.

        Args:
            config_dict (dict): Dictionary containing the profiling configuration parameters.
            role (str, optional): String identifier for the profiler role. Defaults to empty string.
        """
        self.role = role
        self.profile = False
        self.mstx = False
        self.stage = "all"
        self.profile_save_path = ""
        self.profile_export_type = "text"
        self.profile_level = "level0"
        self.profile_with_memory = False
        self.profile_record_shapes = False
        self.profile_with_cpu = True
        self.profile_with_npu = True
        self.profile_with_module = False
        self.profile_step_start = 1
        self.profile_step_end = 2
        self.profile_analysis = False
        self.profile_ranks = "all"

        self.update(config_dict)


class MsprobeConfig(BaseConfig):
    """Msprobe configuration class for debugging and data dumping.

    This class manages msprobe parameters for dumping key data, configurations,
    and training/inference information for debugging purposes.

    Attributes:
        role (str): Identifier for the msprobe role.
        msprobe (bool): Enable/disable the msprobe. Set to True to enable msprobe.
        dump_path (str): Path where msprobe dump data will be saved.
        key_data_dump (bool): Whether to dump key data.
        configurations_dump (bool): Whether to dump configurations.
        actor_train_dump (bool): Whether to dump actor training data.
        actor_infer_dump (bool): Whether to dump actor inference data.
        token_range_start (int): Start of token range for dumping.
        token_range_end (int): End of token range for dumping.
        reference_dump (bool): Whether to dump reference model data.
        critic_train_dump (bool): Whether to dump critic training data.
        step_start (int): Step to start dumping.
        step_end (int): Step to end dumping.
    """

    def __init__(self, config_dict, role=""):
        """Initialize MsprobeConfig with configuration dictionary and role.

        Args:
            config_dict (dict): Dictionary containing the msprobe configuration parameters.
            role (str, optional): String identifier for the Msprobe role. Defaults to empty string.
        """
        self.role = role
        self.msprobe = False
        self.dump_path = "./msprobe_dump"
        self.key_data_dump = False
        self.configurations_dump = False
        self.actor_train_dump = False
        self.actor_infer_dump = False
        self.token_range_start = 0
        self.token_range_end = 0
        self.reference_dump = False
        self.critic_train_dump = False
        self.step_start = 0
        self.step_end = 0

        self.update(config_dict)
