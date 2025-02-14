from scipy.spatial.transform import Rotation as R

import numpy as np

from hexrd.core import rotations


def allclose(a, b):
    """
    numpy all close but make sure shapes are the same
    """
    return a.shape == b.shape and np.allclose(a, b)


def rand_quat(n=1):
    """
    Generate a random unit quaternion for other methods
    """
    quat = np.random.rand(n, 4) * 2 - 1  # n x 4 array
    return quat / np.linalg.norm(quat, axis=1, keepdims=True)


def quat_to_scipy_rotation(q):
    """
    scipy has quaternions written in a differnt order, so we need to convert
    """
    return R.from_quat(np.roll(q, -1, axis=q.ndim - 1))


def test_make_rotmat(num_quats):
    """
    Make rotmats with scipy and rotations.py from quternions, and compare
    """
    np.random.seed(0)

    for _ in range(100):
        q = rand_quat(num_quats)
        if num_quats == 1:
            q = q[0]
        # Compute the rotation matrix using scipy
        R_scipy = quat_to_scipy_rotation(q).as_matrix()
        # Compute the rotation matrix using rotations.py
        R_rotations = rotations.rotMatOfQuat(q if num_quats == 1 else q.T)
        # Compare the rotation matrices
        assert allclose(R_scipy, R_rotations)


def test_quat_mult(num_quats):
    """
    Generate quaternions and test if multiplication works
    """
    np.random.seed(0)
    for _ in range(100):
        # Generate two random unit quaternions
        q1 = rand_quat(num_quats)
        q2 = rand_quat(num_quats)
        # Compute the product of the quaternions
        qp = rotations.quatProduct(q1.T, q2.T).T

        # Do the same in scipy
        r1 = quat_to_scipy_rotation(q1)
        r2 = quat_to_scipy_rotation(q2)

        r_12 = r2 * r1
        r_qp = quat_to_scipy_rotation(qp)

        # Compare the results
        for a, b in zip(r_12, r_qp):
            assert a.approx_equal(b)


def test_quat_mult_one_to_many():
    """
    quatProduct supports one quaternion multiplied by many others
    """
    np.random.seed(0)
    for _ in range(100):
        # Generate two random unit quaternions
        q1 = rand_quat(10)
        q2 = rand_quat()
        # Compute the product of the quaternions
        qp1 = rotations.quatProduct(q1.T, q2.T).T
        qp2 = rotations.quatProduct(q2.T, q1.T).T

        # Do the same in scipy
        r1 = quat_to_scipy_rotation(q1)
        r2 = quat_to_scipy_rotation(q2)

        r_12 = r2 * r1
        r_21 = r1 * r2
        r_qp1 = quat_to_scipy_rotation(qp1)
        r_qp2 = quat_to_scipy_rotation(qp2)

        # Compare the results
        for a, b in zip(r_12, r_qp1):
            assert a.approx_equal(b)

        for a, b in zip(r_21, r_qp2):
            assert a.approx_equal(b)


def test_invert_quat(num_quats):
    """
    Generate quaternions and test if inversion works
    """
    np.random.seed(0)
    for _ in range(100):
        # Generate a random unit quaternion
        q = rand_quat(num_quats)
        # Compute the inverse of the quaternion
        q_inv = rotations.invertQuat(q.T).T

        # Do the same in scipy
        r = quat_to_scipy_rotation(q)
        r_inv = r.inv()
        r_inv2 = quat_to_scipy_rotation(q_inv)

        # Compare the results
        for a, b in zip(r_inv, r_inv2):
            assert a.approx_equal(b)


def test_quat_product_matrix(num_quats):
    """
    Ensure quatProductMatrix works
    """
    np.random.seed(0)
    for _ in range(100):
        # Generate the quaternions for the matrix
        q1 = rand_quat(num_quats)
        # Quat to multiply by
        q_mult = rand_quat()
        # Compute the product matrix
        left_matrix = rotations.quatProductMatrix(q1.T, mult='left')
        right_matrix = rotations.quatProductMatrix(q1.T, mult='right')
        prod1 = np.dot(left_matrix, q_mult.T)
        prod2 = np.dot(right_matrix, q_mult.T)

        # Normalize products (should only change the sign if "necessary").
        prod1 = rotations.fixQuat(prod1).squeeze()
        prod2 = rotations.fixQuat(prod2).squeeze()

        prod1_scipy = (
            quat_to_scipy_rotation(q1) * quat_to_scipy_rotation(q_mult)
        ).as_quat()
        prod2_scipy = (
            quat_to_scipy_rotation(q_mult) * quat_to_scipy_rotation(q1)
        ).as_quat()

        # Fix the scipy results
        prod1_scipy = np.roll(prod1_scipy, 1, axis=1)
        prod2_scipy = np.roll(prod2_scipy, 1, axis=1)
        prod1_scipy = rotations.fixQuat(prod1_scipy.T).T
        prod2_scipy = rotations.fixQuat(prod2_scipy.T).T

        if num_quats == 1:
            prod1_scipy = prod1_scipy[0]
            prod2_scipy = prod2_scipy[0]

        assert allclose(prod1, prod1_scipy)
        assert allclose(prod2, prod2_scipy)


def test_quat_of_angle_axis(num_quats):
    """
    Generate quaternions from angles and axes, and compare with scipy
    """
    np.random.seed(0)
    for _ in range(100):
        # Generate a random angle and axis
        angle = np.random.rand(num_quats) * 2 * np.pi

        # This is just so more of quatOfAngleAxis is tested
        if len(angle) == 1:
            angle = angle[0]

        axis = np.random.rand(3, num_quats) * 2 - 1
        axis /= np.linalg.norm(axis, axis=0, keepdims=True)
        # Compute the quaternion using rotations.py
        q_rotations = rotations.quatOfAngleAxis(angle, axis).T
        # Compute the quaternion using scipy
        q_scipy = R.from_rotvec((axis * angle).T).as_quat()
        # Fix scipy results so it's in the right format
        q_scipy = np.roll(q_scipy, 1, axis=1)
        q_scipy = rotations.fixQuat(q_scipy.T).T
        assert allclose(q_rotations, q_scipy)


def test_quat_of_angle_axis_single_axis():
    """
    quatOfAngleAxis supports a list of angles with one axis, testing that
    """
    np.random.seed(0)
    for _ in range(100):
        angles = np.random.rand(10) * 2 * np.pi
        axis = np.random.rand(3) * 2 - 1
        axis /= np.linalg.norm(axis)
        q_rotations = rotations.quatOfAngleAxis(angles, np.array([axis]).T)
        # Turn axis into a list of axes
        axes = np.tile(axis, (len(angles), 1)).T
        q_rotations2 = rotations.quatOfAngleAxis(angles, axes)

        assert allclose(q_rotations, q_rotations2)


def test_exp_map_of_quat(num_quats):
    """
    Generate quaternions from exponential maps, and compare with rotMats
    """
    np.random.seed(0)
    for _ in range(100):
        quat = rand_quat(num_quats)
        if num_quats == 1:
            quat = quat[0]
        else:
            quat = quat.T
        rot_mat = rotations.rotMatOfQuat(quat)
        exp_map = rotations.expMapOfQuat(quat)
        # Make sure these representations agree
        rot_mat2 = rotations.rotMatOfExpMap(exp_map)
        assert allclose(rot_mat, rot_mat2)


def test_quat_of_exp_map(num_quats):
    """
    Make sure quatOfExpMap and expMapOfQuat are inverses
    """
    np.random.seed(0)
    for _ in range(100):
        quat = rand_quat(num_quats)
        if num_quats == 1:
            quat = rotations.fixQuat(quat.T).T[0]
        else:
            quat = rotations.fixQuat(quat.T)
        exp_map = rotations.expMapOfQuat(quat)
        quat2 = rotations.quatOfExpMap(exp_map)
        assert allclose(quat, quat2)


def test_quat_of_rot_mat(num_quats):
    """
    Make sure quatOfRotMat and rotMatOfQuat are inverses
    """
    np.random.seed(0)
    for _ in range(100):
        quat = rand_quat(num_quats)
        if num_quats == 1:
            quat = rotations.fixQuat(quat.T).T[0]
        else:
            quat = rotations.fixQuat(quat.T)
        rot_mat = rotations.rotMatOfQuat(quat)
        quat2 = rotations.quatOfRotMat(rot_mat)
        if num_quats == 1:
            quat2 = quat2.T[0]
        assert allclose(quat, quat2)


def test_angle_axis_of_rot_mat(num_quats):
    """
    Assuming quat math is right, check if this function agrees
    """
    np.random.seed(0)
    for _ in range(100):
        quat = rand_quat(num_quats)
        if num_quats == 1:
            quat = rotations.fixQuat(quat.T).T[0]
        else:
            quat = rotations.fixQuat(quat.T)
        rot_mat = rotations.rotMatOfQuat(quat)
        angle, axis = rotations.angleAxisOfRotMat(rot_mat)
        quat2 = rotations.quatOfAngleAxis(angle, axis)
        if num_quats == 1:
            quat2 = quat2.T[0]
        assert allclose(quat, quat2)


def pytest_generate_tests(metafunc):
    """
    Make sure methods work with multiple quaternions as well as single inputs.
    """
    if 'num_quats' in metafunc.fixturenames:
        metafunc.parametrize('num_quats', [1, 10])
