# tests for solve_omega

from __future__ import absolute_import

import pytest

from common import function_implementations

all_impls = pytest.mark.parametrize('solve_omega_impl, module_name', 
                                    function_implementations('solve_omega'))


@all_impls
def test_sample1(solve_omega_impl, module_name):
    pass

@all_impls
def test_sample2(solve_omega_impl, module_name):
    pass
