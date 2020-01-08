# tests for angles_to_gvec

from __future__ import absolute_import

import pytest

from common import function_implementations


all_impls = pytest.mark.parametrize('gvec_to_xy_impl, module_name', 
                                    function_implementations('gvec_to_xy'))


@all_impls
def test_sample1(gvec_to_xy_impl, module_name):
    pass

@all_impls
def test_sample2(gvec_to_xy_impl, module_name):
    pass
