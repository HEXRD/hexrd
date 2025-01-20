"""
Not a complete test, just checking that the stress and strain tensor vs
vector representation changes are inverses
"""

import numpy as np
from hexrd.core import matrixutil as mu


def test_stress_repr():
    """Test the representations of stress"""
    for _ in range(100):
        vec = np.random.rand(6)
        ten = mu.stressVecToTen(vec)
        vec_back = mu.stressTenToVec(ten).T[0]
        assert np.allclose(vec, vec_back)


def test_strain_repr():
    """Test the representations of strain"""
    for _ in range(100):
        vec = np.random.rand(6)
        ten = mu.strainVecToTen(vec)
        vec_back = mu.strainTenToVec(ten).T[0]
        assert np.allclose(vec, vec_back)
