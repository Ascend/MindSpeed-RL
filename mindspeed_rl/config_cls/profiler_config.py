from mindspeed_rl.config_cls.base_config import BaseConfig


class ProfilerConfig(BaseConfig):
    '''
    Profiler configuration class.

    Parameters:
        config_dict: Dictionary containing the profiling configuration parameters
        role: String identifier for the profiler role

    Attributes:
        role (str): Identifier for the profiler role.
        profile (bool): Enable/disable the profiler. Set to True to enable performance analysis.
        mstx (bool): Enable/disable lightweight collection mode. True for lightweight mode.
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
        profile_analysis(bool): Whether to analyze profile data online.
        profile_ranks (str): The ranks to be profiled, can be set to "all" for all ranks.
        stage (str): Profiling stage, options include "all", "actor_generate", "actor_compute_log_prob", 
                     "actor_update", "reference_compute_log_prob.
    '''

    def __init__(self, config_dict, role=""):
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