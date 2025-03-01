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
"""Just an initialize test"""

import pytest  # Just try can import or not

from mindspeed_rl import MegatronConfig

from tests.test_tools.dist_test import DistributedTest


class TestConfig(DistributedTest):
    world_size = 1

    def test_megatron_config(self):
        model_config = {'model': {'llama_7b': {'use_mcore_models': True, 'useless_case': 1}}}
        config = {'model': 'llama_7b', 'use_mcore_models': False, 'bad_case': 1}

        m_config = MegatronConfig(config, model_config.get('model'))

        assert not m_config.use_mcore_models, "use_mcore_models Failed !"

        assert not hasattr(m_config, 'useless_case'), "useless_case Failed !"
        assert not hasattr(m_config, 'bad_case'), "bad_case Failed !"
