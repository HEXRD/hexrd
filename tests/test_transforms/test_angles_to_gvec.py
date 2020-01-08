# tests for angles_to_gvec

from __future__ import absolute_import

import pytest
import numpy as np
import numpy.testing as np_testing

from common import function_implementations


all_impls = pytest.mark.parametrize('angles_to_gvec_impl, module_name', 
                                    function_implementations('angles_to_gvec')
                                )

@all_impls
def test_simple_pair(angles_to_gvec_impl, module_name):
    bHat = np.r_[0.0, 0.0, -1.0]
    eHat = np.r_[1.0, 0.0, 0.0]
    angs = np.array([np.pi, 0.0], dtype= np.double)
    expected = np.r_[0.0, 0.0, 1.0]

    # single entry codepath
    res = angles_to_gvec_impl(angs, bHat, eHat)
    np_testing.assert_almost_equal(res, expected)

    # vector codepath (should return dimensions accordingly)
    res = angles_to_gvec_impl(np.atleast_2d(angs), bHat, eHat)
    np_testing.assert_almost_equal(res, np.atleast_2d(expected))

@all_impls
def test_simple_triplet(angles_to_gvec_impl, module_name):
    bHat = np.r_[0.0, 0.0, -1.0]
    eHat = np.r_[1.0, 0.0, 0.0]
    angs = np.array([np.pi, 0.0, 0.0], dtype= np.double)
    expected = np.r_[0.0, 0.0, 1.0]

    # single entry codepath
    res = angles_to_gvec_impl(angs, bHat, eHat)
    np_testing.assert_almost_equal(res, expected)

    # vector codepath (should return dimensions accordingly)
    res = angles_to_gvec_impl(np.atleast_2d(angs), bHat, eHat)
    np_testing.assert_almost_equal(res, np.atleast_2d(expected))


    
