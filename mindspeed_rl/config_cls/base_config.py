# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

class BaseConfig:
    '''
    Base configuration class.
    '''
    def update(self, config_dict, model_config_dict=None):
        '''
        Method to update parameters from a config dictionary
        '''
        if 'model' in config_dict:
            self.update(model_config_dict[config_dict['model']])

        for key, value in config_dict.items():
            if key == 'model':
                continue

            key = key.replace('-', '_')
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self):
        '''Represent the model config as a string for easy reading'''
        return f"<{self.__class__.__name__} {vars(self)}>"


    def items(self):
        return self.__dict__.items()
