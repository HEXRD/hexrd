import math

import numpy as np

import hexrd.constants as ct


def convert_axis_angle_to_rmat(axis, angle):
    # Copied from: https://github.com/ovillellas/xrd-transforms/blob/b94f8b2d7839d883829d00a2adc5bec9c80e0116/test_xrd_transforms/common.py#L59  # noqa
    # This is based on
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/

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


def compare_vector_set(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> bool:
    """Compares two sets of vectors for equality. Ignores the order."""
    i = np.lexsort(vectors1.T)
    j = np.lexsort(vectors2.T)

    return np.allclose(
        vectors1[i], vectors2[j], rtol=rtol, atol=atol, equal_nan=True
    )
