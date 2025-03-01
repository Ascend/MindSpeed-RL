# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

from .base_config import BaseConfig


class MegatronConfig(BaseConfig):
    '''
    Model configuration class.

    models_parameters:
    use_mcore_models: Whether to use MCore models (default: True)
    '''

    def __init__(self, config_dict: Dict, model_config_dict: Dict):
        '''
        Initialize model configuration from the provided config dictionary.
        All instance attributes are initialized using the dictionary keys.

        config_dict: Dictionary containing the configuration parameters
        model_config_dict: a dict of model configuration
        '''
        # Default values can still be defined if no config is provided
        self.use_mcore_models = True

        self.update(config_dict, model_config_dict)
