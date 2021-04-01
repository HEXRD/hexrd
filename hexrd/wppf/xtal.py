import numpy as np
from hexrd.utils.decorators import numba_njit_if_available

@numba_njit_if_available(cache=True, nogil=True)
def _calc_dspacing(rmt, hkls):
    nhkls = hkls.shape[0]
    dsp = np.zeros(hkls.shape[0])

    for ii in np.arange(nhkls):
        g = hkls[ii,:]
        dsp[ii] = 1.0/np.sqrt(np.dot(g, 
            np.dot(rmt, g)))
    return dsp

@numba_njit_if_available(cache=True, nogil=True)
def _get_tth(dsp, wavelength):
    nn = dsp.shape[0]
    tth = np.zeros([nn,])
    wavelength_allowed_hkls = np.zeros([nn,], dtype=bool)
    for ii in nn:
        d = dsp[ii]
        glen = 1./d
        sth = glen*wavelength/2.
        if(np.abs(sth) <= 1.0):
            t = 2. * np.degrees(np.arcsin(sth))
            tth[i] = t
            wavelength_allowed_hkls[ii] = True
        else:
            tth[ii] = np.nan
            wavelength_allowed_hkls[ii] = False

    return tth, wavelength_allowed_hkls
