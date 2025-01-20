import numpy as np

from hexrd.core.material.crystallography import hklToStr


def hkl_to_str(hkl):
    # Must be a list for consistency
    hkl = hkl.tolist() if isinstance(hkl, np.ndarray) else hkl
    return hklToStr(hkl)


def str_to_hkl(hkl_str):
    return list(map(int, hkl_str.split()))
