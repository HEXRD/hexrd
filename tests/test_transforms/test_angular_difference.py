# tests for angular_difference

from __future__ import absolute_import

import pytest

from common import function_implementations

all_impls = pytest.mark.parametrize('angular_difference_impl, module_name',
                                    function_implementations('angular_difference'))


@all_impls
def test_sample1(angular_difference_impl, module_name):
    pass

@all_impls
def test_sample2(angular_difference_impl, module_name):
    pass
