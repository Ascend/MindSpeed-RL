# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from .loss_register import LossRegister


class LossFuncFactory(LossRegister):
    class_map = dict()
