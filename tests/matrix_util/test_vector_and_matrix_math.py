import numpy as np
from hexrd.core import matrixutil as mu


def test_cross():
    """Test cross product"""
    np.random.seed(0)
    for _ in range(100):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        assert np.allclose(np.cross(v1, v2), mu.cross(v1, v2))


def test_determinant():
    """Test determinant"""
    # Deteminant 0 testcase first
    m2 = mu.rankOneMatrix(np.array([[1], [0], [0]]))
    assert np.allclose(mu.determinant3(m2), 0)
    np.random.seed(0)
    for _ in range(100):
        m = np.random.rand(3, 3)
        assert np.allclose(np.linalg.det(m), mu.determinant3(m))


def test_trace():
    """Test trace"""
    np.random.seed(0)
    for _ in range(100):
        m = np.random.rand(3, 3)
        assert np.allclose(np.trace(m), mu.trace3(m))


def test_null_space():
    """Test null space and rankOneMatrix together"""
    np.random.seed(0)
    for _ in range(100):
        m = np.random.rand(6, 6)
        # Random matrices will almost never have a null space
        assert len(mu.nullSpace(m)) == 0

    # Test with rank one matrix
    for _ in range(100):
        vec = np.random.rand(5).reshape((5, 1))
        m = mu.rankOneMatrix(vec)
        null_space = mu.nullSpace(m)
        assert len(null_space) == 4
        assert np.allclose(np.dot(m, null_space.T), 0)

    # Test with 0 matrix
    m = np.zeros((6, 6))
    null_space = mu.nullSpace(m)
    assert len(null_space) == 6
