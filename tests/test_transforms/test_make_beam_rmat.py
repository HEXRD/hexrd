# tests for make_beam_rmat

from __future__ import absolute_import

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations
from common import xf_cnst

ATOL_IDENTITY = 1e-10

all_impls = pytest.mark.parametrize('make_beam_rmat_impl, module_name',
                                    function_implementations('make_beam_rmat'))

# ------------------------------------------------------------------------------

# Test reference frame result

@all_impls
def test_reference_beam_rmat(make_beam_rmat_impl, module_name):
    """Building from the standard beam_vec and eta_vec should
    yield an identity matrix.

    This is somehow assumed in other parts of the code where using the default
    cnst.beam_vec and cnst.eta_vec implies an identity beam rotation matrix that
    is ellided in operations"""

    rmat = make_beam_rmat_impl(xf_cnst.beam_vec, xf_cnst.eta_vec)

    assert_allclose(rmat, xf_cnst.identity_3x3, atol=ATOL_IDENTITY)


# ------------------------------------------------------------------------------

# Test error conditions

@all_impls
def test_zero_beam_vec(make_beam_rmat_impl, module_name):
    beam_vec = np.array([0. ,0., 0.]) # this is bad...
    eta_vec = np.array([1., 0., 0.])

    with pytest.raises(RuntimeError):
        make_beam_rmat_impl(beam_vec, eta_vec)


@all_impls
def test_colinear_beam_eta_vec(make_beam_rmat_impl, module_name):
    with pytest.raises(RuntimeError):
        make_beam_rmat_impl(xf_cnst.beam_vec, xf_cnst.beam_vec)


# ------------------------------------------------------------------------------

# Test orthonormal results

@all_impls
def test_orthonormal_1(make_beam_rmat_impl, module_name):
    beam_vec = np.array([1.0, 2.0, 3.0])
    other_vec = np.array([5.0, 2.0, 1.0])

    # force both inputs to be orthogonal and normalized
    eta_vec = np.cross(beam_vec, other_vec)

    beam_vec /= np.linalg.norm(beam_vec)
    eta_vec /= np.linalg.norm(eta_vec)

    rmat = make_beam_rmat_impl(beam_vec, eta_vec)

    # dot(A, A.T) == Identity seems a good orthonormality check
    # Note: atol needed as rtol is not useful for '0.' entries.
    assert_allclose(np.dot(rmat, rmat.T), xf_cnst.identity_3x3, atol=ATOL_IDENTITY)


@all_impls
def test_orthonormal_2(make_beam_rmat_impl, module_name):
    # same as above although the inputs are not normalized
    beam_vec = np.array([1.0, 2.0, 3.0])
    other_vec = np.array([5.0, 2.0, 1.0])

    # force both inputs to be orthogonal
    eta_vec = np.cross(beam_vec, other_vec)

    rmat = make_beam_rmat_impl(beam_vec, eta_vec)

    # dot(A, A.T) == Identity seems a good orthonormality check
    # Note: atol needed as rtol is not useful for '0.' entries.
    assert_allclose(np.dot(rmat, rmat.T), xf_cnst.identity_3x3, atol=ATOL_IDENTITY)


@all_impls
def test_orthonormal_3(make_beam_rmat_impl, module_name):
    # same as above although the inputs are neither normalized nor orthogonal
    beam_vec = np.array([1.0, 2.0, 3.0])
    other_vec = np.array([5.0, 2.0, 1.0])

    rmat = make_beam_rmat_impl(beam_vec, other_vec)

    # dot(A, A.T) == Identity seems a good orthonormality check
    # Note: atol needed as rtol is not useful for '0.' entries.
    assert_allclose(np.dot(rmat, rmat.T), xf_cnst.identity_3x3, atol=ATOL_IDENTITY)


# ------------------------------------------------------------------------------

# Test strided inputs

@all_impls
def test_strided_beam(make_beam_rmat_impl, module_name):
    buff = np.zeros((3,2), order="C")
    buff[:,:] = 42.0 # fill with some trash value
    buff[:,0] = xf_cnst.beam_vec # but set a strided vector to the valid value

    rmat = make_beam_rmat_impl(buff[:,0], xf_cnst.eta_vec)

    # This should result in identity (see test_reference_beam_rmat)
    assert_allclose(rmat, xf_cnst.identity_3x3, atol=ATOL_IDENTITY)


@all_impls
def test_strided_eta(make_beam_rmat_impl, module_name):
    buff = np.zeros((3,2), order="C")
    buff[:,:] = 42.0 # fill with some trash value
    buff[:,0] = xf_cnst.eta_vec # but set a strided vector to the valid value

    rmat = make_beam_rmat_impl(xf_cnst.beam_vec, buff[:,0])

    # This should result in identity (see test_reference_beam_rmat)
    assert_allclose(rmat, xf_cnst.identity_3x3, atol=ATOL_IDENTITY)


@all_impls
def test_strided_beam_eta(make_beam_rmat_impl, module_name):
    buff = np.zeros((3,4), order="C")
    buff[:,:] = 42.0
    buff[:,0] = xf_cnst.beam_vec
    buff[:,2] = xf_cnst.eta_vec

    rmat = make_beam_rmat_impl(buff[:,0], buff[:,2])
    # This should result in identity (see test_reference_beam_rmat)
    assert_allclose(rmat, xf_cnst.identity_3x3, atol=ATOL_IDENTITY)

