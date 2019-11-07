# tests for quat_distance

from __future__ import absolute_import

import pytest

from common import function_implementations


all_impls = pytest.mark.parametrize('quat_distance_impl, module_name', 
                                    function_implementations('quat_distance'))


@all_impls
def test_sample1(quat_distance_impl, module_name):
    pass

@all_impls
def test_sample2(quat_distance_impl, module_name):
    pass

