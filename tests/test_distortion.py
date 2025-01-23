from hexrd.distortion.dexela_2923 import Dexela_2923, _find_quadrant
from hexrd import constants
import numpy as np


def test_distortion():
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
    if not np.all(abs(result - ptile) <= constants.epsf):
        raise RuntimeError("distortion apply failed!")
    if not np.all(abs(result_inv + ptile) <= constants.epsf):
        raise RuntimeError("distortion apply_inverse failed!")
