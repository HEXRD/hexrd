"""GE41RT Detector Distortion"""

from typing import List

import numpy as np
import numba

from .distortionabc import DistortionABC
from .registry import _RegisterDistortionClass
from .utils import newton

from hexrd.core import constants as cnst
from hexrd.core.extensions import inverse_distortion

RHO_MAX = 204.8  # max radius in mm for ge detector


# NOTE: Deprecated in favor of inverse_distortion.ge_41rt_inverse_distortion
@numba.njit(nogil=True, cache=True, fastmath=True)
def _ge_41rt_inverse_distortion(
    inputs: np.ndarray[np.float64, np.float64],
    rhoMax: float,
    params: List[float],
):
    radii = np.hypot(inputs[:, 0], inputs[:, 1])
    inverted_radii = np.reciprocal(radii)
    cosines = inputs[:, 0] * inverted_radii
    cosine_double_angles = 2 * np.square(cosines) - 1
    cosine_quadruple_angles = 2 * np.square(cosine_double_angles) - 1
    sqrt_p_is = rhoMax / np.sqrt(
        -(
            params[0] * cosine_double_angles
            + params[1] * cosine_quadruple_angles
            + params[2]
        )
    )
    solutions = (
        (2 / np.sqrt(3))
        * sqrt_p_is
        * np.cos(
            np.arccos(((-3 * np.sqrt(3) / 2) * radii / sqrt_p_is)) / 3
            + 4 * np.pi / 3
        )
    )

    return solutions[:, None] * inputs * inverted_radii[:, None]


@numba.njit(nogil=True, cache=True)
def _ge_41rt_distortion(out, in_, rhoMax, params):
    p0, p1, p2, p3, p4, p5 = params[0:6]
    rxi = 1.0 / rhoMax

    for el in range(len(in_)):
        xi, yi = in_[el, 0:2]
        ri = np.sqrt(xi * xi + yi * yi)
        if ri < cnst.sqrt_epsf:
            ri_inv = 0.0
        else:
            ri_inv = 1.0 / ri
        sinni = yi * ri_inv
        cosni = xi * ri_inv
        cos2ni = cosni * cosni - sinni * sinni
        sin2ni = 2 * sinni * cosni
        cos4ni = cos2ni * cos2ni - sin2ni * sin2ni
        ratio = ri * rxi

        ri = (
            p0 * ratio**p3 * cos2ni
            + p1 * ratio**p4 * cos4ni
            + p2 * ratio**p5
            + 1
        ) * ri
        xi = ri * cosni
        yi = ri * sinni
        out[el, 0] = xi
        out[el, 1] = yi

    return out


def _rho_scl_func_inv(ri, ni, ro, rx, p):
    retval = (
        p[0] * (ri / rx) ** p[3] * np.cos(2.0 * ni)
        + p[1] * (ri / rx) ** p[4] * np.cos(4.0 * ni)
        + p[2] * (ri / rx) ** p[5]
        + 1
    ) * ri - ro
    return retval


def _rho_scl_dfunc_inv(ri, ni, ro, rx, p):
    retval = (
        p[0] * (ri / rx) ** p[3] * np.cos(2.0 * ni) * (p[3] + 1)
        + p[1] * (ri / rx) ** p[4] * np.cos(4.0 * ni) * (p[4] + 1)
        + p[2] * (ri / rx) ** p[5] * (p[5] + 1)
        + 1
    )
    return retval


def inverse_distortion_numpy(rho0, eta0, rhoMax, params):
    return newton(
        rho0,
        _rho_scl_func_inv,
        _rho_scl_dfunc_inv,
        (eta0, rho0, rhoMax, params),
    )


class GE_41RT(DistortionABC, metaclass=_RegisterDistortionClass):
    maptype = "GE_41RT"

    def __init__(self, params, **kwargs):
        self._params = np.asarray(params, dtype=float).flatten()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        assert len(x) == 6, "parameter list must have len of 6"
        self._params = np.asarray(x, dtype=float).flatten()

    @property
    def is_trivial(self):
        return (
            self.params[0] == 0 and self.params[1] == 0 and self.params[2] == 0
        )

    def apply(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_in = np.asarray(xy_in, dtype=float)
            xy_out = np.empty_like(xy_in)
            _ge_41rt_distortion(
                xy_out, xy_in, float(RHO_MAX), np.asarray(self.params)
            )
            return xy_out

    def apply_inverse(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_in = np.asarray(xy_in, dtype=float)
            xy_out = inverse_distortion.ge_41rt_inverse_distortion(
                xy_in, float(RHO_MAX), np.asarray(self.params[:3])
            )
            return xy_out
