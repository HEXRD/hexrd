from hexrd.distortion.dexela_2923 import Dexela_2923, _find_quadrant
from hexrd.distortion.dexela_2923_quad import Dexela_2923_quad
from hexrd import constants
import numpy as np


def test_dexela_2923_distortion():
    pts = np.random.randn(16, 2)
    qi = _find_quadrant(pts)

    # test trivial
    params = np.zeros(8)
    dc = Dexela_2923(params)
    if not np.all(dc.apply(pts) - pts == 0.0):
        raise RuntimeError("distortion apply failed!")
    if not np.all(dc.apply_inverse(pts) - pts == 0.0):
        raise RuntimeError("distortion apply_inverse failed!")

    # test non-trivial
    params = np.random.randn(8)
    dc = Dexela_2923(params)
    ptile = np.vstack([params.reshape(4, 2)[j - 1, :] for j in qi])
    result = dc.apply(pts) - pts
    result_inv = dc.apply_inverse(pts) - pts
    if not np.all(abs(result - ptile) <= constants.ten_epsf):
        raise RuntimeError("distortion apply failed!")
    if not np.all(abs(result_inv + ptile) <= constants.ten_epsf):
        raise RuntimeError("distortion apply_inverse failed!")


def test_dexela_2923_quad_distortion():
    pts = np.random.randn(16, 2)
    qi = _find_quadrant(pts)

    # test trivial
    params = np.zeros(10)
    dc = Dexela_2923(params)
    if not np.all(dc.apply(pts) - pts == 0.0):
        raise RuntimeError("distortion apply failed!")
    if not np.all(dc.apply_inverse(pts) - pts == 0.0):
        raise RuntimeError("distortion apply_inverse failed!")

    # test non-trivial

    # this is the original test submited in
    # https://github.com/HEXRD/hexrd/issues/749
    # but it does not currently work. First, params needs to be of size 6, but
    # this break vstack command bellow. Not sure how to adapt it.
    # params = np.random.randn(10)
    # dc = Dexela_2923_quad(params)
    # ptile = np.vstack([params.reshape(4, 2)[j - 1, :] for j in qi])
    # result = dc.apply(pts) - pts
    # result_inv = dc.apply_inverse(pts) - pts
    # if not np.all(abs(result - ptile) <= constants.epsf):
    #    raise RuntimeError("distortion apply failed!")
    # if not np.all(abs(result_inv + ptile) <= constants.epsf):
    #    raise RuntimeError("distortion apply_inverse failed!")
    # return True

    # we simply test that apply and reverse cancel each other
    params = np.random.randn(6)
    dc = Dexela_2923_quad(params)
    result = dc.apply_inverse(dc.apply(pts))
    if not np.all(abs(result - pts) <= 100 * constants.epsf):
        raise RuntimeError("distortion apply_inverse(apply) failed!")
