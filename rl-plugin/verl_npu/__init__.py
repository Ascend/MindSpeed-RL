from verl_npu.module_injection import bootstrap_default_aliases


def _initialize_npu_plugin():
    """Initialize NPU plugin after bootstrapping aliases."""
    bootstrap_default_aliases()
    
    from verl_npu.plugin import apply_npu_plugin
    apply_npu_plugin()

# Initialize on module import
_initialize_npu_plugin()
