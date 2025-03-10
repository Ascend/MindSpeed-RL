# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

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
