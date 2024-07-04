import numpy as np
from numba import njit
from hexrd.wppf.peakfunctions import _unit_gaussian, _unit_lorentzian
"""
naming convention for the derivative is as follows:
_d_<peakshape>_<parameter>

available <peakshape> :
pvfcl: finger cox jephcoat asymmetric
pv_wppf: symmetric pseudo voight
pvoight_pink_beam: pink beam case

available <parameters>: (only a couple listed 
since the total number is very large)
fwhm: full width half maxima
tth: mean of peak
.....

"""

@njit(cache=True, nogil=True)
def _d_pvfcj_fwhm():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_tth():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_U():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_V():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_W():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_P():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_X():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_Y():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_Xe():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_Ye():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_Xs():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_HL():
    pass

@njit(cache=True, nogil=True)
def _d_fwhm_SL():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_scale():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_phase_fraction():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_trns():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_shft():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_zero_error():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_shkls():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_a():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_b():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_c():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_alpha():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_beta():
    pass

@njit(cache=True, nogil=True)
def _d_pvfcj_gamma():
    pass
