# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.

import torch
from torch import nn
from vllm.config import (LoadConfig, LoadFormat, ModelConfig, ParallelConfig,
                         VllmConfig, set_current_vllm_config)
from vllm.model_executor.model_loader.loader import BaseModelLoader, _initialize_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype


class DummyMegatronModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        print("DummyMegatronModelLoader load model")
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(vllm_config=vllm_config)
        return model.eval()
