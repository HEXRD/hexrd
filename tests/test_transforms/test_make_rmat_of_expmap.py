# tests for make_rmat_of_expmap

from __future__ import absolute_import

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations

from common import xf_cnst
from common import ATOL_IDENTITY
from common import convert_axis_angle_to_expmap
from common import convert_axis_angle_to_rmat

all_impls = pytest.mark.parametrize('make_rmat_of_expmap_impl, module_name',
                                    function_implementations('make_rmat_of_expmap'))


@pytest.fixture(scope="module")
def axes():
    def parametric_point_in_sphere(t):
        # t going from -1.0 to 1.0 will form an spiral over a sphere
        alpha = 0.5 * t * np.pi
        beta  = t * np.pi * 42.0

        z = np.sin(alpha)
        o = np.cos(alpha)

        x = o*np.cos(beta)
        y = o*np.sin(beta)

        return np.stack((x, y, z), axis=-1)

    return parametric_point_in_sphere(np.linspace(-1.0, 1.0, num=32))


@pytest.fixture(scope="module")
def angs():
    return np.linspace(-np.pi, np.pi, num=120)

# ------------------------------------------------------------------------------

# Test trivial case

@all_impls
def test_zero_expmap(make_rmat_of_expmap_impl, module_name):
    exp_map = np.zeros((3,))

    rmat = make_rmat_of_expmap_impl(exp_map)

    assert_allclose(rmat, xf_cnst.identity_3x3, atol=ATOL_IDENTITY)


@all_impls
def test_2pi_expmap(make_rmat_of_expmap_impl, module_name):
    """all this should result in identity - barring numerical error.
    Note this goes via a different codepath as phi in the code is not 0."""

    rmat = make_rmat_of_expmap_impl(np.array([2*np.pi, 0., 0.]))
    assert_allclose(rmat, xf_cnst.identity_3x3, atol=ATOL_IDENTITY)

    rmat = make_rmat_of_expmap_impl(np.array([0., 2*np.pi, 0.]))
    assert_allclose(rmat, xf_cnst.identity_3x3, atol=ATOL_IDENTITY)

    rmat = make_rmat_of_expmap_impl(np.array([0., 0.,2*np.pi]))
    assert_allclose(rmat, xf_cnst.identity_3x3, atol=ATOL_IDENTITY)


@all_impls
def test_random_cases(axes, angs, make_rmat_of_expmap_impl, module_name):
    for axis in axes:
        for ang in angs:
            expmap = convert_axis_angle_to_expmap(axis, ang)
            expected = convert_axis_angle_to_rmat(axis, ang)
            result = make_rmat_of_expmap_impl(expmap)
            assert_allclose(result, expected,
                            atol=ATOL_IDENTITY)


# ------------------------------------------------------------------------------

# check that for some random inputs the resulting matrix is orthonormal

@all_impls
def test_orthonormal(axes, angs, make_rmat_of_expmap_impl, module_name):
    for axis in axes:
        for ang in angs:
            expmap = convert_axis_angle_to_expmap(axis, ang)
            rmat = make_rmat_of_expmap_impl(expmap)
            # dot(A, A.T) == IDENTITY is a good orthonormality check
            assert_allclose(np.dot(rmat, rmat.T), xf_cnst.identity_3x3,
                            atol=ATOL_IDENTITY)



# ------------------------------------------------------------------------------

# Test strided input
@all_impls
def test_strided(make_rmat_of_expmap_impl, module_name):
    exp_map = np.array([42.0, 3., 32.5]) # A random expmap

    buff = np.zeros((3, 3), order='C')
    buff[:,0] = exp_map[:] # assign the expmap to a column, so it is strided

    result_contiguous = make_rmat_of_expmap_impl(exp_map)
    result_strided = make_rmat_of_expmap_impl(buff[:,0])

    # in fact, a stricter equality check should work as well,
    # but anyways...
    assert_allclose(result_contiguous, result_strided)
