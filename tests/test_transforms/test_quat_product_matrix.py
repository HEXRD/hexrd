# tests for quat_product_matrix

from __future__ import absolute_import

import pytest

from common import function_implementations


all_impls = pytest.mark.parametrize('quat_product_matrix_impl, module_name', 
                                    function_implementations('quat_product_matrix'))


@all_impls
def test_sample1(quat_product_matrix_impl, module_name):
    pass

@all_impls
def test_sample2(quat_product_matrix_impl, module_name):
    pass

