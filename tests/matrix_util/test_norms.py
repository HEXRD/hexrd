import numpy as np
from hexrd.core import matrixutil as mu


def test_column_norm(n_dim):
    """Test column normalization"""
    np.random.seed(0)
    for _ in range(100):
        if n_dim == 1:
            v = np.random.rand(10)
        else:
            v = np.random.rand(10, 10)
        v_norm_expected = np.linalg.norm(v, axis=0)
        v_norm = mu.columnNorm(v)
        assert np.allclose(v_norm, v_norm_expected)


def test_row_norm():
    """Test column normalization"""
    np.random.seed(0)
    for _ in range(100):
        v = np.random.rand(10, 10)
        v_norm_expected = np.linalg.norm(v, axis=1)
        v_norm = mu.rowNorm(v)
        assert np.allclose(v_norm, v_norm_expected)


def test_unit_vector(n_dim):
    """Test normalizing column vectors"""
    np.random.seed(0)
    for _ in range(100):
        if n_dim == 1:
            v = np.random.rand(10)
        else:
            v = np.random.rand(10, 10)
        v_unit_expected = v / np.linalg.norm(v, axis=0)
        v_unit = mu.unitVector(v)
        assert np.allclose(v_unit, v_unit_expected)


def test_normvec():
    """Test the normvec method"""
    np.random.seed(0)
    for _ in range(100):
        v = np.random.rand(10)
        norm_expected = np.linalg.norm(v)
        norm = mu.normvec(v)
        assert np.allclose(norm_expected, norm)


def test_normvec3():
    """Test the normvec3 method"""
    np.random.seed(0)
    for _ in range(100):
        v = np.random.rand(3)
        norm_expected = mu.normvec(v)
        norm = mu.normvec3(v)
        assert np.allclose(norm_expected, norm)


def test_normalized():
    """Test the normalized method"""
    np.random.seed(0)
    for _ in range(100):
        v = np.random.rand(10)
        v_norm = np.linalg.norm(v)
        v_normed_expected = v / v_norm
        v_normed = mu.normalized(v)
        assert np.allclose(v_normed, v_normed_expected)


def pytest_generate_tests(metafunc):
    """
    Make sure methods work on different dimension sizes.
    """
    if 'n_dim' in metafunc.fixturenames:
        metafunc.parametrize('n_dim', [1, 2])
