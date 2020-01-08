# tests for angles_to_dvec

from __future__ import absolute_import

import pytest

from common import function_implementations


all_impls = pytest.mark.parametrize('angles_to_dvec_impl, module_name', 
                                    function_implementations('angles_to_dvec'))


@all_impls
def test_sample1(angles_to_dvec_impl, module_name):
    pass

@all_impls
def test_sample2(angles_to_dvec_impl, module_name):
    pass
