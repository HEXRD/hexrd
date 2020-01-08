# tests for row_norm

from __future__ import absolute_import

import pytest
import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations


all_impls = pytest.mark.parametrize('row_norm_impl, module_name',
                                    function_implementations('row_norm'))


def _get_random_vectors_array():
    # return a (n,3) array with some vectors and a (n) array with the expected
    # result norms.
    arr = np.array([[42.0,  0.0,  0.0],
                    [12.0, 12.0, 12.0],
                    [ 0.0,  0.0,  0.0],
                    [ 0.7, -0.7,  0.0],
                    [-0.0, -0.0, -0.0]])
    return arr


@all_impls
def test_random_vectors(row_norm_impl, module_name):
    # checking against numpy.linalg.norm
    vecs = np.ascontiguousarray(_get_random_vectors_array())

    # element by element
    for i in range(len(vecs)):
        result = row_norm_impl(vecs[i])
        expected = np.linalg.norm(vecs[i])
        assert type(result) == type(expected)
        assert result.dtype == expected.dtype
        assert_allclose(result, expected)

    # all in a row
    result = row_norm_impl(vecs)
    expected = np.linalg.norm(vecs, axis=1)
    assert type(result) == type(expected)
    assert result.dtype == expected.dtype

    assert_allclose(result, expected)


@all_impls
def test_keep_dimensions(row_norm_impl, module_name):
    # check that the case of a 2d array with a single vector is handled
    # in a consistent way (reduces 1 dimension)
    test_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.double)

    result = row_norm_impl(test_vec)

    assert result.shape == (1,)

    result2 = row_norm_impl(test_vec[0])

    assert result2.shape == ()


@all_impls
def test_random_vectors_strided(row_norm_impl, module_name):
    # this is the same as test_random_vectors, but in a layout that forces
    # strided memory access for the inner dimension
    vecs = np.asfortranarray(_get_random_vectors_array())

    # element by element
    for i in range(len(vecs)):
        result = row_norm_impl(vecs[i])
        expected = np.linalg.norm(vecs[i])
        assert_allclose(result, expected)

    # all in a row
    result = row_norm_impl(vecs)
    expected = np.linalg.norm(vecs, axis=1)
    assert_allclose(result, expected)


@all_impls
def test_too_many_dimensions(row_norm_impl, module_name):
    # our norm should fail on 3 dimensional arrays using a ValueError
    test_vec = np.arange(16., dtype=np.double).reshape((4,2,2))
    with pytest.raises(ValueError):
        row_norm_impl(test_vec)
