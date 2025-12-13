# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import pytest

from mindspeed_rl.models.loss.loss_register import LossRegister
from tests.test_tools.dist_test import DistributedTest


class TestDummyLoss:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TestLossRegister(DistributedTest):
    world_size = 1
    is_dist_test = False

    def setUp(self):
        LossRegister.class_map.clear()

    def test_register_loss(self):
        @LossRegister.register_loss("train", "actor")
        class TestActorLoss:
            pass

        assert "train_actor" in LossRegister.class_map
        assert LossRegister.class_map["train_actor"] == TestActorLoss

    def test_get_class(self):
        @LossRegister.register_loss("train", "critic")
        class TestCriticLoss:
            pass

        result = LossRegister.get_class("train", "critic")
        assert result == TestCriticLoss

        result = LossRegister.get_class("eval", "actor")
        assert result is None

    def test_get_instance(self):
        @LossRegister.register_loss("train", "actor")
        class TestActorLoss:
            def __init__(self, param1, param2=None):
                self.param1 = param1
                self.param2 = param2

        instance = LossRegister.get_instance("train", "actor", "value1", param2="value2")
        assert instance is not None
        assert isinstance(instance, TestActorLoss)
        assert instance.param1 == "value1"
        assert instance.param2 == "value2"

        instance = LossRegister.get_instance("eval", "critic")
        assert instance is None

    def test_multiple_registrations(self):
        @LossRegister.register_loss("train", "actor")
        class TrainActorLoss:
            pass

        @LossRegister.register_loss("train", "critic")
        class TrainCriticLoss:
            pass

        @LossRegister.register_loss("eval", "actor")
        class EvalActorLoss:
            pass

        assert LossRegister.get_class("train", "actor") == TrainActorLoss
        assert LossRegister.get_class("train", "critic") == TrainCriticLoss
        assert LossRegister.get_class("eval", "actor") == EvalActorLoss

    def test_register_decorator_returns_class(self):
        @LossRegister.register_loss("train", "actor")
        class TestActorLoss:
            pass

        assert TestActorLoss.__name__ == "TestActorLoss"