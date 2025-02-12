import numpy as np
import numba

from .distortionabc import DistortionABC
from .registry import _RegisterDistortionClass


class Dexela_2923_quad(DistortionABC, metaclass=_RegisterDistortionClass):

    maptype = "Dexela_2923_quad"

    def __init__(self, params, **kwargs):
        assert len(params) == 6, "parameter list must have len of 6"
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
        return np.all(self.params == 0)

    def apply(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_in = np.asarray(xy_in, dtype=float)
            xy_out = np.empty_like(xy_in)
            _dexela_2923_quad_distortion(
                xy_out, xy_in, np.asarray(self.params)
            )
            return xy_out

    def apply_inverse(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_in = np.asarray(xy_in, dtype=float)
            xy_out = np.empty_like(xy_in)
            _dexela_2923_quad_inverse_distortion(
                xy_out, xy_in, np.asarray(self.params)
            )
            return xy_out


def _find_quadrant(xy_in):
    quad_label = np.zeros(len(xy_in), dtype=int)
    in_2_or_3 = xy_in[:, 0] < 0.0
    in_1_or_4 = ~in_2_or_3
    in_3_or_4 = xy_in[:, 1] < 0.0
    in_1_or_2 = ~in_3_or_4
    quad_label[np.logical_and(in_1_or_4, in_1_or_2)] = 1
    quad_label[np.logical_and(in_2_or_3, in_1_or_2)] = 2
    quad_label[np.logical_and(in_2_or_3, in_3_or_4)] = 3
    quad_label[np.logical_and(in_1_or_4, in_3_or_4)] = 4
    return quad_label


@numba.njit(nogil=True, cache=True)
def _dexela_2923_quad_distortion(out, in_, params):
    # 1 + x + y, inverse. Someone should definitely check my math here...
    p0, p1, p2, p3, p4, p5 = params[0:6]
    p1 = p1 + 1e-12
    p5 = p5 + 1e-12
    out[:, 0] = (
        in_[:, 0] / p1 - p0 / p1 - (p2 / (p1 * p5) * (in_[:, 1] - p3))
    ) / (1 - (p2 * p4) / (p1 * p5))
    out[:, 1] = (in_[:, 1] - p3 - p4 * out[:, 0]) / p5

    return out


@numba.njit(nogil=True, cache=True)
def _dexela_2923_quad_inverse_distortion(out, in_, params):
    # 1 + x + y
    p0, p1, p2, p3, p4, p5 = params[0:6]
    p1 = p1 + 1e-12
    p5 = p5 + 1e-12
    out[:, 0] = p0 + p1 * in_[:, 0] + p2 * in_[:, 1]
    out[:, 1] = p3 + p4 * in_[:, 0] + p5 * in_[:, 1]

    return out
