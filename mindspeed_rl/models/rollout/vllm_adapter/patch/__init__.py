# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from mindspeed_rl.utils.utils import is_multimodal
from .qwen2_5_vl_visionmlp_patch import replace_qwen2_5_visionmlp
from .rotary_embedding_patch import replace_qwenvl_mrope

if is_multimodal():
    replace_qwen2_5_visionmlp()
    replace_qwenvl_mrope()