# tests for xy_to_gvec

from __future__ import absolute_import

import pytest

from common import function_implementations

all_impls = pytest.mark.parametrize('xy_to_gvec_impl, module_name', 
                                    function_implementations('xy_to_gvec'))


@all_impls
def test_sample1(xy_to_gvec_impl, module_name):
    pass

@all_impls
def test_sample2(xy_to_gvec_impl, module_name):
    pass
