# tests for validate_angle_ranges

from __future__ import absolute_import

import pytest

from common import function_implementations


all_impls = pytest.mark.parametrize('validate_angle_ranges_impl, module_name', 
                                    function_implementations('validate_angle_ranges'))


@all_impls
def test_sample1(validate_angle_ranges_impl, module_name):
    pass

@all_impls
def test_sample2(validate_angle_ranges_impl, module_name):
    pass
