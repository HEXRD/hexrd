# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness


from __future__ import absolute_import
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import make_binary_rmat
from hexrd.rotations import quatOfAngleAxis, rotMatOfQuat
from transforms.common import random_unit_vectors


def test_make_binary_rmat():
    np.random.seed(0)
    for _ in range(100):
        axis = random_unit_vectors()
        rmat = make_binary_rmat(axis)

        # Two binary rmats should be the identity
        assert np.allclose(rmat @ rmat, np.eye(3))
        assert np.allclose(rmat.T @ rmat, np.eye(3)), "It is orthogonal"
        assert np.all((np.abs(rmat) - 1 < 1e-10) | (np.abs(rmat) < 1e-10)), "It is binary"
        rmat_expected = rotMatOfQuat(
            quatOfAngleAxis(np.pi, np.c_[axis])
        )
        assert np.allclose(rmat, rmat_expected)
