from scipy.spatial.transform import Rotation as R

import numpy as np

from math import copysign

from hexrd import rotations


def random_rot_mat_euler():
    """
    Generate a random scipy rotation and equivalent rotMatEuler
    """
    quats = np.random.rand(4) * 2 - 1
    quats /= np.linalg.norm(quats)
    rotation = R.from_quat(quats)

    # Extrinsic or intrinsic, random
    extrinsic = np.random.choice([True, False])

    # Generate random euler angle sequence
    seq = np.random.choice(
        [
            'xyz',
            'zyx',
            'zxy',
            'yxz',
            'yzx',
            'xzy',
            'xyx',
            'xzx',
            'yxy',
            'yzy',
            'zxz',
            'zyz',
        ]
    )
    if not extrinsic:
        seq = seq.upper()

    angs = rotation.as_euler(seq)

    return rotations.RotMatEuler(angs, seq.lower(), extrinsic), rotation


def test_rot_mat_euler_constructor():
    """
    Generate a bunch of random rotMatEulers and check that they are set right
    """
    for _ in range(1000):
        rot, scipy_rot = random_rot_mat_euler()
        assert isinstance(rot, rotations.RotMatEuler)

        seq = rot.axes_order
        if not rot.extrinsic:
            seq = seq.upper()

        assert np.allclose(rot.angles, scipy_rot.as_euler(seq))


def test_vals_from_rot_mat_euler():
    """
    Make sure that the euler maps create the correct rotation mat and expmap
    """
    for _ in range(1000):
        rot, scipy_rot = random_rot_mat_euler()
        assert np.allclose(rot.rmat, scipy_rot.as_matrix())

        # Generate it from the expmap and check that it's the same as the
        # rotation matrices (note expmap representations are not unique,
        # but rotation matrix rotations are)
        exp_map = rot.exponential_map
        rot_from_expmap = rotations.rotMatOfExpMap(exp_map)
        assert np.allclose(rot_from_expmap, rot.rmat)


def test_units_setter():
    """
    Check if updating units breaks anything
    """
    for _ in range(1000):
        rot, scipy_rot = random_rot_mat_euler()
        old_angs = rot.angles

        # Change to degrees, make sure angs changed but rotation matrix doesn't
        rot.units = 'degrees'
        assert rot.units == 'degrees'
        assert not np.allclose(old_angs, rot.angles)
        assert np.allclose(rot.rmat, scipy_rot.as_matrix())

        # Change back
        rot.units = 'radians'
        assert rot.units == 'radians'
        assert np.allclose(old_angs, rot.angles)
        assert np.allclose(rot.rmat, scipy_rot.as_matrix())


def pytest_generate_tests(metafunc):
    """
    Make sure methods work with multiple quaternions as well as single inputs.
    """
    if 'num_quats' in metafunc.fixturenames:
        metafunc.parametrize('num_quats', [1, 10])
