#
# This will allow to easily adapt tests in case the module name changes.
#
# The tests are meant to be kept independent so that they don't need to be
# installed.
#

from __future__ import absolute_import

import math

import numpy as np

import xrd_transforms as xf
from xrd_transforms import constants as xf_cnst

def function_implementations(api_func_name):
    """returns a list of pairs (function, implementation_name) for all
    implementations of the API function.

    This is useful in parametrization of tests"""
    
    assert api_func_name in xf.API

    impls = [(getattr(xf, api_func_name), 'default')]
    for name, module in xf.implementations.items():
        impl = getattr(module, api_func_name, None)
        if impl is not None:
            impls.append((impl, name))
    
    return impls


ATOL_IDENTITY = 1e-10

def convert_axis_angle_to_expmap(axis, angle):
    # expmap is just the normalized axis multiplied by the angle
    angle = float(angle)
    axis = np.array(axis, dtype=float)

    assert axis.shape == (3,)

    if abs(angle) < xf_cnst.epsf:
        return np.zeros((3,), dtype=float)

    axis_norm = np.linalg.norm(axis)
    if axis_norm < xf_cnst.epsf:
        raise ValueError("axis is zero")

    ratio = angle/axis_norm

    return axis * ratio


def convert_axis_angle_to_rmat(axis, angle):
    # This is based on
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/

    angle = float(angle)
    axis = np.array(axis, dtype=float)
    assert axis.shape == (3,)

    if abs(angle) < xf_cnst.epsf:
        return xf_cnst.identity_3x3

    axis_norm = np.linalg.norm(axis)
    if axis_norm < xf_cnst.epsf:
        raise ValueError("axis is zero")

    axis /= axis_norm

    m = np.empty((3, 3), dtype=float)

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c

    m[0,0] = c + axis[0]*axis[0]*t
    m[0,1] = axis[0]*axis[1]*t - axis[2]*s
    m[0,2] = axis[0]*axis[2]*t + axis[1]*s
    m[1,0] = axis[0]*axis[1]*t + axis[2]*s
    m[1,1] = c + axis[1]*axis[1]*t
    m[1,2] = axis[1]*axis[2]*t - axis[0]*s
    m[2,0] = axis[0]*axis[2]*t - axis[1]*s
    m[2,1] = axis[1]*axis[2]*t + axis[0]*s
    m[2,2] = c + axis[2]*axis[2]*t

    return m
