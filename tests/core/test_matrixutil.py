import pytest
import numpy as np
from scipy import sparse
from hexrd.core import matrixutil
from hexrd.core import constants

# --- Fixtures ---


@pytest.fixture
def symm_matrix():
    """A standard 3x3 symmetric matrix."""
    return np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])


@pytest.fixture
def skew_matrix():
    """A standard 3x3 skew-symmetric matrix."""
    return np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])


@pytest.fixture
def random_vectors():
    """Set of random 3D vectors."""
    np.random.seed(42)
    return np.random.rand(3, 10)


# --- Norm & Vector Utils ---


def test_columnNorm(random_vectors):
    res = matrixutil.columnNorm(random_vectors)
    assert res.shape == (10,)
    expected = np.linalg.norm(random_vectors, axis=0)
    np.testing.assert_allclose(res, expected)

    with pytest.raises(RuntimeError):
        matrixutil.columnNorm(np.zeros((3, 3, 3)))


def test_rowNorm(random_vectors):
    res = matrixutil.rowNorm(random_vectors)
    assert res.shape == (3,)
    expected = np.linalg.norm(random_vectors, axis=1)
    np.testing.assert_allclose(res, expected)

    with pytest.raises(RuntimeError):
        matrixutil.rowNorm(np.zeros((3, 3, 3)))


def test_unitVector(random_vectors):
    res = matrixutil.unitVector(random_vectors)
    norms = np.linalg.norm(res, axis=0)
    np.testing.assert_allclose(norms, 1.0)

    zeros = np.zeros((3, 2))
    res_z = matrixutil.unitVector(zeros)
    np.testing.assert_allclose(res_z, 0.0)


# --- Null Space ---


def test_nullSpace():
    A = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    ns = matrixutil.nullSpace(A)

    prod = np.dot(A, ns.T)
    np.testing.assert_allclose(prod, 0.0, atol=1e-10)


def test_nullSpace_transpose_branch():
    A = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # 3x2
    ns = matrixutil.nullSpace(A)
    np.testing.assert_allclose(np.dot(ns.T, A), 0.0, atol=1e-10)


# --- Sparse Matrix Construction ---


def test_blockSparseOfMatArray():
    mats = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    smat = matrixutil.blockSparseOfMatArray(mats)
    assert isinstance(smat, sparse.csc_matrix)
    dense = smat.toarray()
    expected = np.array(
        [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8]]
    )
    np.testing.assert_array_equal(dense, expected)

    with pytest.raises(RuntimeError):
        matrixutil.blockSparseOfMatArray(np.zeros((2, 2)))


# --- Mandel-Voigt Conversions ---


def test_symmToVecMV_vecMVToSymm(symm_matrix):
    vec = matrixutil.symmToVecMV(symm_matrix, scale=True)
    mat = matrixutil.vecMVToSymm(vec, scale=True)
    np.testing.assert_allclose(mat, symm_matrix)

    vec_ns = matrixutil.symmToVecMV(symm_matrix, scale=False)
    mat_ns = matrixutil.vecMVToSymm(vec_ns, scale=False)
    np.testing.assert_allclose(mat_ns, symm_matrix)

    expected = np.array([1.0, 4.0, 6.0, 5.0, 3.0, 2.0])
    np.testing.assert_allclose(vec_ns, expected)


def test_nrmlProjOfVecMV():
    vec = np.array([[1.0], [0.0], [0.0]])
    proj = matrixutil.nrmlProjOfVecMV(vec)
    assert proj.shape == (1, 6)
    np.testing.assert_allclose(proj.flatten(), [1, 0, 0, 0, 0, 0])


# --- Matrix Generators & Errors (Coverage) ---


def test_rankOneMatrix(random_vectors):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        dyads = matrixutil.rankOneMatrix(random_vectors)
        assert dyads.shape == (10, 3, 3)
        for i in range(10):
            v = random_vectors[:, i : i + 1]
            expected = np.dot(v, v.T)
            np.testing.assert_allclose(dyads[i], expected)

        v2 = np.random.rand(3, 10)
        dyads2 = matrixutil.rankOneMatrix(random_vectors, v2)
        assert dyads2.shape == (10, 3, 3)


def test_rankOneMatrix_errors():
    with pytest.raises(RuntimeError, match="input vec1 is the wrong shape"):
        matrixutil.rankOneMatrix(np.zeros((3, 3, 3)))

    with pytest.raises(ValueError):
        matrixutil.rankOneMatrix(np.zeros((3, 1)), np.zeros((3, 3, 3)))

    with pytest.raises(RuntimeError, match="Number of vectors differ"):
        matrixutil.rankOneMatrix(np.zeros((3, 2)), np.zeros((3, 3)))


def test_skew(random_vectors):
    mats = np.random.rand(10, 3, 3)
    skews = matrixutil.skew(mats.copy())
    for i in range(10):
        m = mats[i]
        expected = 0.5 * (m - m.T)
        np.testing.assert_allclose(skews[i], expected)

    m = np.random.rand(3, 3)
    s = matrixutil.skew(m.copy())
    np.testing.assert_allclose(s, 0.5 * (m - m.T))


def test_skew_errors():
    with pytest.raises(RuntimeError, match="only works for square arrays"):
        matrixutil.skew(np.zeros((3, 4)))

    with pytest.raises(RuntimeError, match="only works for square arrays"):
        matrixutil.skew(np.zeros((2, 3, 4)))

    with pytest.raises(RuntimeError, match="only works for square arrays"):
        matrixutil.skew(np.zeros((10,)))


def test_symm(random_vectors):
    m = np.random.rand(3, 3)
    s = matrixutil.symm(m.copy())
    np.testing.assert_allclose(s, 0.5 * (m + m.T))


def test_symm_errors():
    with pytest.raises(RuntimeError, match="only works for square arrays"):
        matrixutil.symm(np.zeros((3, 4)))

    with pytest.raises(RuntimeError, match="only works for square arrays"):
        matrixutil.symm(np.zeros((2, 3, 4)))

    with pytest.raises(RuntimeError, match="only works for square arrays"):
        matrixutil.symm(np.zeros((10,)))


# --- Vector <-> Skew Matrix Conversions & Errors ---


def test_skewMatrixOfVector_vectorOfSkewMatrix(skew_matrix):
    w_expected = np.array([1.0, 2.0, 3.0])
    w_calc = matrixutil.vectorOfSkewMatrix(skew_matrix.copy())
    np.testing.assert_allclose(w_calc.flatten(), w_expected)

    mat_calc = matrixutil.skewMatrixOfVector(w_calc)
    np.testing.assert_allclose(mat_calc, skew_matrix)

    w_batch = np.array([[1, 2, 3], [4, 5, 6]]).T
    mats_batch = matrixutil.skewMatrixOfVector(w_batch)
    w_back = matrixutil.vectorOfSkewMatrix(mats_batch)
    np.testing.assert_allclose(w_back, w_batch)


def test_skewMatrixOfVector_errors():
    with pytest.raises(RuntimeError, match="input is not a 3-d vector"):
        matrixutil.skewMatrixOfVector(np.zeros(4))

    with pytest.raises(RuntimeError, match="input is of incorrect shape"):
        matrixutil.skewMatrixOfVector(np.zeros((4, 2)))

    with pytest.raises(RuntimeError, match="input is incorrect shape"):
        matrixutil.skewMatrixOfVector(np.zeros((3, 2, 2)))


def test_vectorOfSkewMatrix_errors():
    with pytest.raises(RuntimeError, match=r"input is not \(3, 3\)"):
        matrixutil.vectorOfSkewMatrix(np.zeros((4, 4)))

    with pytest.raises(RuntimeError, match=r"input is not \(3, 3\)"):
        matrixutil.vectorOfSkewMatrix(np.zeros((2, 4, 3)))

    with pytest.raises(RuntimeError, match="input is incorrect shape"):
        matrixutil.vectorOfSkewMatrix(np.zeros((2, 3, 3, 1)))


# --- Matrix Multiplication & Errors ---


def test_multMatArray():
    ma1 = np.tile(np.eye(2), (2, 1, 1))
    ma2 = np.tile(2 * np.eye(2), (2, 1, 1))
    res = matrixutil.multMatArray(ma1, ma2)
    assert res.shape == (2, 2, 2)
    np.testing.assert_allclose(res[0], 2 * np.eye(2))


def test_multMatArray_errors():
    with pytest.raises(RuntimeError, match="input is incorrect shape"):
        matrixutil.multMatArray(np.zeros((2, 2)), np.zeros((2, 2)))

    with pytest.raises(RuntimeError, match="mismatch on number of matrices"):
        matrixutil.multMatArray(np.zeros((2, 2, 2)), np.zeros((3, 2, 2)))

    with pytest.raises(
        RuntimeError, match="mismatch on internal matrix dimensions"
    ):
        matrixutil.multMatArray(np.zeros((2, 2, 3)), np.zeros((2, 2, 2)))


# --- Vector Duplication (Including Legacy & Numba PyFunc) ---


def test_uniqueVectors():
    vecs = np.array([[1, 0, 1], [0, 1, 0]])
    unique = matrixutil.uniqueVectors(vecs)
    assert unique.shape == (2, 2)


def test_findDuplicateVectors():
    vec = np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0]])
    eqv, uid = matrixutil.findDuplicateVectors(vec)
    assert len(eqv) == 1
    group = eqv[0]
    assert 0 in group and 2 in group
    assert 1 in uid


def test_findduplicatevectors_numba_coverage():
    """Explicitly test the pure python version of the numba function for coverage."""
    func = matrixutil._findduplicatevectors

    vec = np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0]])
    eqv = func(vec, 1e-8, False)
    assert eqv[0, 0] == 2.0

    vec_pm = np.array([[1.0, -1.0], [2.0, -2.0]])
    eqv_pm = func(vec_pm, 1e-8, True)
    assert eqv_pm[0, 0] == 1.0

    vec_none = np.array([[1.0, 2.0], [2.0, 3.0]])
    eqv_none = func(vec_none, 1e-8, False)
    assert np.all(np.isnan(eqv_none))


# --- Numba Utilities ---


def test_extract_ijv():
    arr = np.array([[0, 10, 0], [5, 0, 0], [0, 0, 20]])
    thresh = 1
    out_i = np.zeros(9, dtype=int)
    out_j = np.zeros(9, dtype=int)
    out_v = np.zeros(9, dtype=arr.dtype)

    count = matrixutil.extract_ijv(arr, thresh, out_i, out_j, out_v)
    assert count == 3
    found_vals = sorted(out_v[:count])
    assert found_vals == [5, 10, 20]


def test_extract_ijv_numba_coverage():
    """Test pure python version of extract_ijv for coverage lines."""
    func = matrixutil.extract_ijv
    arr = np.array([[0, 10], [5, 0]])
    thresh = 1
    out_i = np.zeros(4, dtype=int)
    out_j = np.zeros(4, dtype=int)
    out_v = np.zeros(4, dtype=arr.dtype)

    count = func(arr, thresh, out_i, out_j, out_v)
    assert count == 2
    assert out_v[0] == 10
    assert out_v[1] == 5


# --- Tensor Vectorization ---


def test_strain_conversions():
    ten = np.random.rand(3, 3)
    ten = 0.5 * (ten + ten.T)
    vec = matrixutil.strainTenToVec(ten)
    ten_back = matrixutil.strainVecToTen(vec.flatten())
    np.testing.assert_allclose(ten, ten_back)


def test_stress_conversions():
    ten = np.random.rand(3, 3)
    ten = 0.5 * (ten + ten.T)
    vec = matrixutil.stressTenToVec(ten)
    ten_back = matrixutil.stressVecToTen(vec.flatten())
    np.testing.assert_allclose(ten, ten_back)


def test_vecds_conversions(symm_matrix):
    vec = matrixutil.symmToVecds(symm_matrix)
    mat_back = matrixutil.vecdsToSymm(vec)
    np.testing.assert_allclose(mat_back, symm_matrix)


def test_ale3dStrainOutToV():
    vec = np.random.rand(6) * 1e-4
    vec[5] = 0.0
    V, Vinv = matrixutil.ale3dStrainOutToV(vec)
    prod = np.dot(V, Vinv)
    np.testing.assert_allclose(prod, np.eye(3), atol=1e-8)


# --- Wahba's Problem ---


def test_solve_wahba():
    v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    w = v.copy()
    R = matrixutil.solve_wahba(v, w)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    theta = np.radians(90)
    R_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    v = np.random.rand(10, 3)
    w = np.dot(v, R_true.T)
    R_calc = matrixutil.solve_wahba(v, w)
    np.testing.assert_allclose(R_calc, R_true, atol=1e-10)

    weights = np.ones(10)
    R_calc_w = matrixutil.solve_wahba(v, w, weights=weights)
    np.testing.assert_allclose(R_calc_w, R_true, atol=1e-10)
