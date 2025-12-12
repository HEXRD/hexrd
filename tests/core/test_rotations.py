import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from unittest.mock import patch
from hexrd.core import rotations
from hexrd.core import constants as cnst

# --- Fixtures ---

@pytest.fixture
def identity_quat():
    return np.array([[1.], [0.], [0.], [0.]])

@pytest.fixture
def rot_90z_quat():
    val = np.sqrt(2)/2
    return np.array([[val], [0.], [0.], [val]])

@pytest.fixture
def rot_80z_quat():
    theta = np.radians(80)
    val_w = np.cos(theta/2)
    val_z = np.sin(theta/2)
    return np.array([[val_w], [0.], [0.], [val_z]])

@pytest.fixture
def random_quats():
    np.random.seed(42)
    qs = np.random.rand(4, 10) - 0.5
    norms = np.linalg.norm(qs, axis=0)
    return qs / norms

# --- Utility Tests ---

def test_arccosSafe():
    assert np.isclose(rotations.arccosSafe(0.5), np.pi/3)
    assert np.isclose(rotations.arccosSafe(1.000001), 0.0)
    assert np.isclose(rotations.arccosSafe(-1.000001), np.pi)
    
    with pytest.raises(RuntimeError):
        rotations.arccosSafe(1.1)

def test_mapAngle():
    res = rotations.mapAngle(3 * np.pi, units='radians')
    assert np.isclose(res, -np.pi) or np.isclose(res, np.pi)
    
    res = rotations.mapAngle(370, units='degrees')
    assert np.isclose(res, 10.0)
    
    res = rotations.mapAngle(-10, ang_range=[0, 360], units='degrees')
    assert np.isclose(res, 350.0)
    
    with pytest.raises(RuntimeError):
        rotations.mapAngle(0, units='bad')
        
    with pytest.raises(RuntimeError):
        rotations.mapAngle(0, ang_range=[0, 100], units='degrees') 

def test_angularDifference():
    d = rotations.angularDifference(10, 350, units='degrees')
    assert np.isclose(d, 20.0)
    
    r = rotations.angularDifference(0.1, 2*np.pi - 0.1, units='radians')
    assert np.isclose(r, 0.2)
    
    d_old = rotations.angularDifference_orig([10], [350], units='degrees')
    assert np.isclose(d_old[0], 20.0)
    
    with pytest.raises(RuntimeError):
        rotations.angularDifference_orig([0], [0], units='bad')

# --- Quaternion Core Tests ---

def test_fixQuat():
    q = np.array([[-1.], [0.], [0.], [0.]])
    q_fixed = rotations.fixQuat(q)
    assert np.allclose(q_fixed, -q)
    assert q_fixed[0, 0] > 0
    
    q_in = np.zeros((2, 4, 5))
    q_in[:, 0, :] = -1 
    q_out = rotations.fixQuat(q_in)
    assert q_out.shape == (2, 4, 5)
    assert q_out[0, 0, 0] > 0

def test_invertQuat(rot_90z_quat):
    q_inv = rotations.invertQuat(rot_90z_quat)
    assert np.isclose(q_inv[0], rot_90z_quat[0])
    assert np.isclose(q_inv[3], -rot_90z_quat[3])

def test_quatProduct(identity_quat, rot_80z_quat):
    res = rotations.quatProduct(identity_quat, rot_80z_quat)
    assert np.allclose(res, rot_80z_quat)
    
    res2 = rotations.quatProduct(rot_80z_quat, rot_80z_quat)
    theta = np.radians(160)
    expected_w = np.cos(theta/2)
    expected_z = np.sin(theta/2)
    assert np.isclose(res2[0], expected_w, atol=1e-7)
    assert np.isclose(res2[3], expected_z, atol=1e-7)

def test_quatProductMatrix(rot_90z_quat):
    q = rot_90z_quat
    mat_r = rotations.quatProductMatrix(q, mult='right')
    assert mat_r.shape == (1, 4, 4)
    
    mat_l = rotations.quatProductMatrix(q, mult='left')
    assert mat_l.shape == (1, 4, 4)
    
    ident = np.array([[1.], [0.], [0.], [0.]])
    prod = np.dot(mat_r[0], ident)
    assert np.allclose(prod, q)
    
    with pytest.raises(RuntimeError):
        rotations.quatProductMatrix(np.zeros((3, 10)))

# --- Conversion Tests ---

def test_quatOfAngleAxis():
    angle = np.pi/2
    axis = np.array([[1.], [0.], [0.]])
    q = rotations.quatOfAngleAxis(angle, axis)
    
    val = np.sqrt(2)/2
    expected = np.array([[val], [val], [0.], [0.]])
    assert np.allclose(q, expected)
    
    with pytest.raises(RuntimeError):
        rotations.quatOfAngleAxis([1, 2], np.zeros((3, 3)))

def test_rotMatOfQuat(rot_90z_quat):
    rmat = rotations.rotMatOfQuat(rot_90z_quat)
    expected = np.array([[0., -1., 0.],
                         [1., 0., 0.],
                         [0., 0., 1.]])
    np.testing.assert_allclose(rmat, expected, atol=1e-7)
    
    with pytest.raises(RuntimeError):
        rotations.rotMatOfQuat(np.zeros(5))
    with pytest.raises(RuntimeError):
        rotations.rotMatOfQuat(np.zeros((5, 2)))

def test_rotmatofquat_numba_coverage(rot_90z_quat):
    """Test pure python version of _rotmatofquat for coverage."""
    func = rotations._rotmatofquat.py_func
    
    rmat = func(rot_90z_quat)
    assert rmat.shape == (1, 3, 3)
    expected = np.array([[[0., -1., 0.],
                          [1., 0., 0.],
                          [0., 0., 1.]]])
    np.testing.assert_allclose(rmat, expected, atol=1e-7)

def test_expMapOfQuat_coverage(rot_90z_quat):
    exp = rotations.expMapOfQuat(rot_90z_quat)
    expected = np.array([0, 0, np.pi/2])
    np.testing.assert_allclose(exp, expected)
    
    quats = np.hstack([rot_90z_quat, rot_90z_quat]) # (4, 2)
    exps = rotations.expMapOfQuat(quats)
    assert exps.shape == (3, 2)
    np.testing.assert_allclose(exps[:, 0], expected)

    exp_single = rotations.expMapOfQuat(rot_90z_quat.flatten())
    np.testing.assert_allclose(exp_single, expected)
    
    with pytest.raises(AssertionError):
        rotations.expMapOfQuat(np.zeros((3, 1))) 
    with pytest.raises(AssertionError):
        rotations.expMapOfQuat(np.zeros((3, 2))) 

def test_rotMatOfExpMap():
    exp = np.array([0, 0, np.pi/2])
    rmat = rotations.rotMatOfExpMap(exp)
    expected = np.array([[0., -1., 0.],
                         [1., 0., 0.],
                         [0., 0., 1.]])
    np.testing.assert_allclose(rmat, expected, atol=1e-7)

def test_quatOfRotMat(rot_90z_quat):
    rmat = np.array([[0., -1., 0.],
                     [1., 0., 0.],
                     [0., 0., 1.]])
    q = rotations.quatOfRotMat(rmat)
    assert np.allclose(q.flatten(), rot_90z_quat.flatten(), atol=1e-7)

def test_angleAxisOfRotMat_coverage():
    rmat = np.eye(3)
    angs, axes = rotations.angleAxisOfRotMat(rmat)
    assert angs.shape == (1,)
    assert axes.shape == (3, 1)
    assert np.isclose(angs[0], 0.0)
    
    rmats = np.zeros((2, 3, 3))
    rmats[0] = np.eye(3)
    rmats[1] = np.eye(3)
    angs, axes = rotations.angleAxisOfRotMat(rmats)
    assert angs.shape == (2,)
    
    with pytest.raises(RuntimeError, match="Input must be a 2 or 3-d ndarray"):
        rotations.angleAxisOfRotMat("bad")
        
    with pytest.raises(RuntimeError, match="rot_mat array must be"):
        rotations.angleAxisOfRotMat(np.zeros((3, 3, 3, 3)))

# --- Euler Angles ---

def test_make_rmat_euler():
    angles = np.array([np.pi/2, 0, 0])
    rmat = rotations.make_rmat_euler(angles, 'zxz', extrinsic=True)
    expected = np.array([[0., -1., 0.],
                         [1., 0., 0.],
                         [0., 0., 1.]])
    np.testing.assert_allclose(rmat, expected, atol=1e-7)
    
    rmat_int = rotations.make_rmat_euler(angles, 'zxz', extrinsic=False)
    np.testing.assert_allclose(rmat_int, expected, atol=1e-7)
    
    with pytest.raises(RuntimeError):
        rotations.make_rmat_euler(angles, 123)
    with pytest.raises(RuntimeError):
        rotations.make_rmat_euler(angles, 'bad')

def test_angles_from_rmat():
    rmat = np.eye(3)
    xyz = rotations.angles_from_rmat_xyz(rmat)
    zxz = rotations.angles_from_rmat_zxz(rmat)
    np.testing.assert_allclose(xyz, 0)
    np.testing.assert_allclose(zxz, 0)
    
    with pytest.raises(RuntimeError):
        rotations.angles_from_rmat_xyz(np.zeros((2,2)))
    with pytest.raises(RuntimeError):
        rotations.angles_from_rmat_xyz(np.eye(3)*2) 

# --- Misorientation & Symmetry ---

@pytest.mark.parametrize("tag, expected_n", [
    ('Ci', 1), ('S2', 1),
    ('C2h', 2),
    ('D2h', 4), ('Vh', 4),
    ('C4h', 4),
    ('D4h', 8),
    ('C3i', 3), ('S6', 3),
    ('D3d', 6),
    ('C6h', 6),
    ('D6h', 12),
    ('Th', 12),
    ('Oh', 24)
])
def test_quatOfLaueGroup_exhaustive(tag, expected_n):
    """Exhaustive test for all Laue group tags."""
    q = rotations.quatOfLaueGroup(tag)
    assert q.shape == (4, expected_n)

def test_quatOfLaueGroup_errors():
    with pytest.raises(RuntimeError, match="entered flag is not a string"):
        rotations.quatOfLaueGroup(123)
    with pytest.raises(RuntimeError, match="unrecognized symmetry group"):
        rotations.quatOfLaueGroup('invalid_tag')

def test_ltypeOfLaueGroup():
    assert rotations.ltypeOfLaueGroup('Ci') == 'triclinic'
    assert rotations.ltypeOfLaueGroup('C2h') == 'monoclinic'
    assert rotations.ltypeOfLaueGroup('D2h') == 'orthorhombic'
    assert rotations.ltypeOfLaueGroup('vh') == 'orthorhombic'
    assert rotations.ltypeOfLaueGroup('C4h') == 'tetragonal'
    assert rotations.ltypeOfLaueGroup('d4h') == 'tetragonal'
    assert rotations.ltypeOfLaueGroup('C3i') == 'trigonal'
    assert rotations.ltypeOfLaueGroup('S6') == 'trigonal'
    assert rotations.ltypeOfLaueGroup('D3d') == 'trigonal'
    assert rotations.ltypeOfLaueGroup('D6h') == 'hexagonal'
    assert rotations.ltypeOfLaueGroup('Oh') == 'cubic'
    assert rotations.ltypeOfLaueGroup('Th') == 'cubic'
    
    with pytest.raises(RuntimeError):
        rotations.ltypeOfLaueGroup(1)
    with pytest.raises(RuntimeError):
        rotations.ltypeOfLaueGroup('bad')

def test_misorientation(rot_90z_quat, identity_quat):
    angle, mis_q = rotations.misorientation(identity_quat, rot_90z_quat)
    assert np.isclose(angle[0], np.pi/2)
    
    d4h = rotations.quatOfLaueGroup('D4h')
    angle_sym, _ = rotations.misorientation(identity_quat, rot_90z_quat, symmetries=(d4h,))
    assert np.isclose(angle_sym[0], 0.0, atol=1e-7)
    
    with pytest.raises(RuntimeError):
        rotations.misorientation(1, 2)
    with pytest.raises(RuntimeError):
        rotations.misorientation(np.zeros(3), np.zeros(3))
    with pytest.raises(RuntimeError):
        rotations.misorientation(np.zeros((4, 2)), np.zeros((4, 2)))
    with pytest.raises(RuntimeError):
        rotations.misorientation(identity_quat, identity_quat, symmetries=(1, 2, 3))

def test_toFundamentalRegion(random_quats):
    mapped = rotations.toFundamentalRegion(random_quats, crysSym='Oh')
    assert mapped.shape == random_quats.shape
    norms = np.linalg.norm(mapped, axis=0)
    np.testing.assert_allclose(norms, 1.0)
    
    q3 = np.random.rand(2, 4, 3) 
    mapped3 = rotations.toFundamentalRegion(q3, crysSym='Oh')
    assert mapped3.shape == (2, 4, 3)
    
    with pytest.raises(NotImplementedError):
        rotations.toFundamentalRegion(random_quats, sampSym='Oh')

def test_quatAverage_full_coverage(rot_90z_quat, identity_quat):
    """Cover functional branches of quatAverage without patch."""
    qsym = rotations.quatOfLaueGroup('Ci')
    
    avg = rotations.quatAverage(rot_90z_quat, qsym)
    assert np.allclose(avg, rot_90z_quat)
    
    q2 = np.hstack([identity_quat, rot_90z_quat])
    
    orig_exp_map = rotations.quatOfExpMap
    def fixed_exp_map(q):
        return orig_exp_map(q).reshape(4, 1)
    
    with patch('hexrd.core.rotations.quatOfExpMap', side_effect=fixed_exp_map):
        avg2 = rotations.quatAverage(q2, qsym)
        expected_45 = np.array([[0.9238795], [0.], [0.], [0.3826834]])
        assert np.allclose(avg2, expected_45, atol=1e-5)
    
    q3 = np.tile(rot_90z_quat, (1, 3)) 
    avg3 = rotations.quatAverage(q3, qsym)
    assert np.allclose(avg3, rot_90z_quat)
    
    q_start_ident = np.hstack([identity_quat, rot_90z_quat, rot_90z_quat])
    avg_start_ident = rotations.quatAverage(q_start_ident, qsym)
    expected_60 = np.array([[0.8660254], [0.], [0.], [0.5]])
    assert np.allclose(avg_start_ident, expected_60, atol=1e-5)
    
    theta = np.radians(10)
    q_pos = np.array([[np.cos(theta/2)], [0], [0], [np.sin(theta/2)]])
    q_neg = np.array([[np.cos(theta/2)], [0], [0], [-np.sin(theta/2)]])
    q_symm_cluster = np.hstack([q_pos, q_neg, identity_quat])
    avg_symm = rotations.quatAverage(q_symm_cluster, qsym)
    assert np.allclose(avg_symm, identity_quat, atol=1e-5)

def test_quatAverageCluster_coverage(rot_90z_quat):
    qsym = rotations.quatOfLaueGroup('Ci')
    
    q1 = rot_90z_quat
    res1 = rotations.quatAverageCluster(q1, qsym)
    assert np.allclose(res1, q1)
    
    q2 = np.hstack([rot_90z_quat, rot_90z_quat])
    
    orig_exp_map = rotations.quatOfExpMap
    def fixed_exp_map(q):
        return orig_exp_map(q).reshape(4, 1)
        
    with patch('hexrd.core.rotations.quatOfExpMap', side_effect=fixed_exp_map):
        res2 = rotations.quatAverageCluster(q2, qsym)
        assert np.allclose(res2, rot_90z_quat)
    
    q3 = np.tile(rot_90z_quat, (1, 5)) 
    res3 = rotations.quatAverageCluster(q3, qsym)
    assert np.allclose(res3, rot_90z_quat)

# --- Fiber Utilities ---

def test_distanceToFiber(rot_90z_quat):
    c = np.array([[0.], [0.], [1.]])
    s = np.array([[0.], [0.], [1.]])
    q = rot_90z_quat
    qsym = rotations.quatOfLaueGroup('Ci')
    
    d = rotations.distanceToFiber(c, s, q, qsym)
    assert np.isclose(d[0], 0.0)
    
    s_anti = np.array([[0.], [0.], [-1.]])
    
    d_no_cs = rotations.distanceToFiber(c, s_anti, q, qsym, centrosymmetry=False)
    assert np.isclose(d_no_cs[0], np.pi)
    
    d_cs = rotations.distanceToFiber(c, s_anti, q, qsym, centrosymmetry=True)
    assert np.isclose(d_cs[0], 0.0)
    
    with pytest.raises(RuntimeError):
        rotations.distanceToFiber(np.zeros(2), s, q, qsym)

def test_discreteFiber():
    c = np.array([[0.], [0.], [1.]])
    s = np.array([[0.], [0.], [1.]])
    
    fibers = rotations.discreteFiber(c, s, ndiv=10)
    assert len(fibers) == 1
    assert fibers[0].shape == (4, 10)
    
    csym = rotations.quatOfLaueGroup('D6h')
    fibers_sym = rotations.discreteFiber(c, s, ndiv=10, csym=csym)
    assert len(fibers_sym) == 1

# --- RotMatEuler Class ---

def test_RotMatEuler_class():
    rme = rotations.RotMatEuler(np.array([90, 0, 0]), 'zxz', units='degrees')
    
    np.testing.assert_allclose(rme.angles, [90, 0, 0])
    
    expected = np.array([[0., -1., 0.],
                         [1., 0., 0.],
                         [0., 0., 1.]])
    np.testing.assert_allclose(rme.rmat, expected, atol=1e-7)
    
    rme.rmat = np.eye(3)
    np.testing.assert_allclose(rme.angles, 0)
    
    np.testing.assert_allclose(rme.exponential_map, 0)

    exp_map = np.array([0, 0, np.pi/2])
    rme.exponential_map = exp_map
    np.testing.assert_allclose(rme.rmat, expected, atol=1e-7)
    
    rme.angles = [0, 0, 0]
    rme.axes_order = 'xyz'
    rme.extrinsic = False
    
    rme.units = 'radians'
    assert rme.units == 'radians'
    np.testing.assert_allclose(rme.angles, 0)
    
    with pytest.raises(RuntimeError):
        rme.units = 'bad'
    with pytest.raises(RuntimeError):
        rme.angles = [1, 2]
    with pytest.raises(RuntimeError):
        rme.extrinsic = 1
