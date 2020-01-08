# -*- mode: python; coding: utf-8 -*-

# tests for make_binary_rmat


from __future__ import absolute_import

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations


all_impls = pytest.mark.parametrize('make_binary_rmat_impl, module_name',
                                    function_implementations('make_binary_rmat'))


def reference(axis):
    # make_binary_rmat is basically the formula:
    #  2Ĝ⊗Ĝ−𝑰
    # its only argument being Ĝ.
    a0, a1, a2 = axis[0:3]
    result = np.empty((3,3), dtype=np.double)

    result[0,0] = 2.0*a0*a0 - 1.0
    result[0,1] = 2.0*a0*a1
    result[0,2] = 2.0*a0*a2
    result[1,0] = 2.0*a1*a0
    result[1,1] = 2.0*a1*a1 - 1.0
    result[1,2] = 2.0*a1*a2
    result[2,0] = 2.0*a2*a0
    result[2,1] = 2.0*a2*a1
    result[2,2] = 2.0*a2*a2 - 1.0

    return result


@pytest.fixture(scope="module")
def sample_vectors():
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


@all_impls
def test_basic(sample_vectors, make_binary_rmat_impl, module_name):
    for vector in sample_vectors:
        result = make_binary_rmat_impl(vector)
        expected = reference(vector)
        assert_allclose(result, expected)


@all_impls
def test_strided(sample_vectors, make_binary_rmat_impl, module_name):
    sample_vectors = np.asfortranarray(sample_vectors)
    for vector in sample_vectors:
        result = make_binary_rmat_impl(vector)
        expected = reference(vector)
        assert_allclose(result, expected)


@all_impls
def test_exception(make_binary_rmat_impl, module_name):
    with pytest.raises(ValueError):
        make_binary_rmat_impl(np.zeros((2,), dtype=np.double))

    with pytest.raises(ValueError):
        make_binary_rmat_impl(np.zeros((4,), dtype=np.double))

    with pytest.raises(ValueError):
        make_binary_rmat_impl(np.zeros((3,4), dtype=np.double))
