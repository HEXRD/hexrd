from scipy.spatial.transform import Rotation as R

import numpy as np

from hexrd import rotations


def random_rot_mat_euler():
    """
    Generate a random scipy rotation and equivalent rotMatEuler
    """
    quats = np.random.rand(4) * 2 - 1
    quats /= np.linalg.norm(quats)
    rotation = R.from_quat(quats)

    # Extrinsic or intrinsic, random
    extrinsic = bool(np.random.choice([True, False]))

    units = np.random.choice(['degrees', 'radians'])

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

    angs = rotation.as_euler(seq, units == 'degrees')

    return rotations.RotMatEuler(angs, seq.lower(), extrinsic, units), rotation


def test_rot_mat_euler_constructor():
    """
    Generate a bunch of random rotMatEulers and check that they are set right
    """
    np.random.seed(0)
    for _ in range(1000):
        rot, scipy_rot = random_rot_mat_euler()
        assert isinstance(rot, rotations.RotMatEuler)

        seq = rot.axes_order
        if not rot.extrinsic:
            seq = seq.upper()

        assert np.allclose(
            rot.angles, scipy_rot.as_euler(seq, rot.units == 'degrees')
        )


def test_vals_from_rot_mat_euler():
    """
    Make sure that the euler maps create the correct rotation mat and expmap
    """
    np.random.seed(0)
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
    np.random.seed(0)
    for _ in range(1000):
        rot, scipy_rot = random_rot_mat_euler()
        old_angs = rot.angles

        cur_unit = rot.units
        other_unit = 'radians' if cur_unit == 'degrees' else 'degrees'

        # Change to degrees, make sure angs changed but rotation matrix doesn't
        rot.units = other_unit
        assert rot.units == other_unit
        assert not np.allclose(old_angs, rot.angles)
        assert np.allclose(rot.rmat, scipy_rot.as_matrix())

        # Change back
        rot.units = cur_unit
        assert rot.units == cur_unit
        assert np.allclose(old_angs, rot.angles)
        assert np.allclose(rot.rmat, scipy_rot.as_matrix())


def test_rmat_setter():
    """
    Make sure updating the rmat changes things properly
    """
    np.random.seed(0)
    for _ in range(1000):
        rot1, _ = random_rot_mat_euler()
        rot2, _ = random_rot_mat_euler()

        # Change the rmat
        rot1.units = rot2.units
        rot1.axes_order = rot2.axes_order
        rot1.extrinsic = rot2.extrinsic
        rot1.rmat = rot2.rmat
        assert np.allclose(rot1.angles, rot2.angles)


def test_exp_map_setter():
    """
    Make sure updating the expmap changes things properly
    """
    np.random.seed(0)
    for _ in range(1000):
        rot1, _ = random_rot_mat_euler()
        rot2, _ = random_rot_mat_euler()

        rot1.units = rot2.units
        rot1.axes_order = rot2.axes_order
        rot1.extrinsic = rot2.extrinsic
        rot1.exponential_map = rot2.exponential_map
        assert np.allclose(rot1.angles, rot2.angles)
