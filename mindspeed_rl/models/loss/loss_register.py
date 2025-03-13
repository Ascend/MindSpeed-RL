# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.


class LossRegister:
    class_map = dict()

    @classmethod
    def register_loss(cls, stage: str, role: str):
        def _register(class_):
            cls.class_map[stage + "_" + role] = class_
            return class_

        return _register

    @classmethod
    def get_instance(cls, stage: str, role: str, *args, **kwargs):
        if stage + "_" + role not in cls.class_map:
            return None
        return cls.class_map.get(stage + "_" + role)(*args, **kwargs)

    @classmethod
    def get_class(cls, stage: str, role: str):
        return cls.class_map.get(stage + "_" + role)
