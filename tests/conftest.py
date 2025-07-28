# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
# reworked/refactored some parts to make it run.
from multiprocessing.pool import RUN
import pytest


def pytest_configure(config):
    config.option.durations = 0
    config.option.durations_min = 1
    config.option.verbose = True


def always_true():
    return True


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, "is_dist_test", False):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = always_true  # Dummy function so test is not run twice


# We allow DistributedTest to reuse distributed environments. When the last
# test for a class is run, we want to make sure those distributed environments
# are destroyed.
def pytest_runtest_teardown(item, nextitem):
    if getattr(item.cls, "reuse_dist_env", True) and (not nextitem or item.cls != nextitem.cls):
        dist_test_class = item.cls()
        if hasattr(dist_test_class, '_pool_cache'):
            for num_procs, pool in dist_test_class._pool_cache.items():
                if pool._state == RUN:  # pool not run when skip UT
                    dist_test_class._close_pool(pool, num_procs, force=True)
            dist_test_class._pool_cache.clear()


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if getattr(fixturedef.func, "is_dist_fixture", False):
        dist_fixture_class = fixturedef.func()
        dist_fixture_class(request)
