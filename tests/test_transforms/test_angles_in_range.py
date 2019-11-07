# tests for angles_in_range

from __future__ import absolute_import

import pytest

from common import function_implementations

all_impls = pytest.mark.parametrize('angles_in_range_impl, module_name',
                                    function_implementations('angles_in_range')
                                )


@all_impls
def test_sample1(angles_in_range_impl, module_name):
    pass

@all_impls
def test_sample2(angles_in_range_impl, module_name):
    pass
