# tests for map_angle

from __future__ import absolute_import

import pytest

from common import function_implementations

all_impls = pytest.mark.parametrize('map_angle_impl, module_name', 
                                    function_implementations('map_angle'))


@all_impls
def test_sample1(map_angle_impl, module_name):
    pass

@all_impls
def test_sample2(map_angle_impl, module_name):
    pass
