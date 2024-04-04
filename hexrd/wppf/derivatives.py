import numpy as np
from hexrd.utils.decorators import numba_njit_if_available
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

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_fwhm():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_tth():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_U():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_V():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_W():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_P():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_X():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_Y():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_Xe():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_Ye():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_Xs():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_HL():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_fwhm_SL():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_scale():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_phase_fraction():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_trns():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_shft():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_zero_error():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_shkls():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_a():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_b():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_c():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_alpha():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_beta():
    pass

@numba_njit_if_available(cache=True, nogil=True)
def _d_pvfcj_gamma():
    pass
