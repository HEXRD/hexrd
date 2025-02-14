import math

import numpy as np

import hexrd.core.constants as ct
from hexrd.core.transforms.new_capi.xf_new_capi import unit_vector


def convert_axis_angle_to_rmat(axis, angle):
    # Copied from: https://github.com/ovillellas/xrd-transforms/blob/b94f8b2d7839d883829d00a2adc5bec9c80e0116/test_xrd_transforms/common.py#L59  # noqa
    # This is based on
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/ # noqa

    angle = float(angle)
    axis = np.array(axis, dtype=float)
    assert axis.shape == (3,)

    if abs(angle) < ct.epsf:
        return ct.identity_3x3

    axis_norm = np.linalg.norm(axis)
    if axis_norm < ct.epsf:
        raise ValueError("axis is zero")

    axis /= axis_norm

    m = np.empty((3, 3), dtype=float)

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c

    m[0, 0] = c + axis[0] * axis[0] * t
    m[0, 1] = axis[0] * axis[1] * t - axis[2] * s
    m[0, 2] = axis[0] * axis[2] * t + axis[1] * s
    m[1, 0] = axis[0] * axis[1] * t + axis[2] * s
    m[1, 1] = c + axis[1] * axis[1] * t
    m[1, 2] = axis[1] * axis[2] * t - axis[0] * s
    m[2, 0] = axis[0] * axis[2] * t - axis[1] * s
    m[2, 1] = axis[1] * axis[2] * t + axis[0] * s
    m[2, 2] = c + axis[2] * axis[2] * t

    return m


def random_rotation_matrix():
    # Generate a random unit quaternion
    q = np.random.rand(4) * 2 - 1
    q /= np.linalg.norm(q)
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    # fmt: off
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    # fmt: on
    return rot_matrix


def random_unit_vectors(shape1=3, shape2=None):
    if shape2 is None:
        return unit_vector(np.random.rand(shape1) * 2 - 1)
    return unit_vector(np.random.rand(shape1, shape2) * 2 - 1)
