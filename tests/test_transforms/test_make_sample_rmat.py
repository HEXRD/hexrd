# tests for make_sample_rmat

from __future__ import absolute_import

import pytest
from numpy.testing import assert_allclose as np_assert_allclose

from common import function_implementations
from common import xf_cnst

all_impls = pytest.mark.parametrize('make_sample_rmat_impl, module_name',
                                    function_implementations('make_sample_rmat'))


@all_impls
def test_make_sample_rmat_chi0_ome0(make_sample_rmat_impl, module_name):
    # when chi = 0.0 and ome = 0.0 the resulting sample rotation matrix should
    # be the identity
    chi = 0.0
    ome = 0.0
    result = make_sample_rmat_impl(0.0, 0.0)

    np_assert_allclose(xf_cnst.identity_3x3, result)


