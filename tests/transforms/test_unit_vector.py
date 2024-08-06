from hexrd.transforms.xfcapi import unit_vector
import numpy as np


def test_unit_vector():
    np.random.seed(0)
    # Generate some random vectors
    n = 50
    vecs = np.random.rand(n, 3) * 4

    # Turn them into unit vectors
    vecs_unit = unit_vector(vecs)

    # Check against elementwise numpy function
    for i in range(n):
        assert np.allclose(
            vecs_unit[i, :], vecs[i, :] / np.linalg.norm(vecs[i, :])
        )
