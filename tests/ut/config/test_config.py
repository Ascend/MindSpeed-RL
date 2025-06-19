# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

from mindspeed_rl import MegatronConfig

from tests.test_tools.dist_test import DistributedTest


class TestConfig(DistributedTest):
    world_size = 1

    def test_megatron_config(self):
        model_config = {'model': {'llama_7b': {'use_mcore_models': True}}}
        config = {'model': 'llama_7b', 'use_mcore_models': False}

        m_config = MegatronConfig(config, model_config.get('model'))

        assert not m_config.use_mcore_models, "use_mcore_models Failed !"

        assert not hasattr(m_config, 'useless_case'), "useless_case Failed !"
        assert not hasattr(m_config, 'bad_case'), "bad_case Failed !"
