# tests for rotate_vecs_about_axis

from __future__ import absolute_import

import pytest

from common import function_implementations

all_impls = pytest.mark.parametrize('rotate_vecs_about_axis_impl, module_name', 
                                    function_implementations('rotate_vecs_about_axis'))


@all_impls
def test_sample1(rotate_vecs_about_axis_impl, module_name):
    pass

@all_impls
def test_sample2(rotate_vecs_about_axis_impl, module_name):
    pass
