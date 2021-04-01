import numpy as np
from hexrd.utils.decorators import numba_njit_if_available
from hexrd import constants
import numba

if constants.USE_NUMBA:
    from numba import prange
else:
    prange = range

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
    tth = np.zeros(dsp.shape[0])
    wavelength_allowed_hkls = np.zeros(dsp.shape[0])
    for ii in np.arange(nn):
        d = dsp[ii]
        glen = 1./d
        sth = glen*wavelength/2.
        if(np.abs(sth) <= 1.0):
            t = 2. * np.degrees(np.arcsin(sth))
            tth[ii] = t
            wavelength_allowed_hkls[ii] = 1
        else:
            tth[ii] = np.nan
            wavelength_allowed_hkls[ii] = 0

    return tth, wavelength_allowed_hkls

@numba_njit_if_available(cache=True, nogil=True)
def _calcanomalousformfactor(atom_type,
    wavelength,
    frel,
    f_anomalous_data,
    f_anomalous_data_sizes):

    f_anam = np.zeros(atom_type.shape,dtype=np.complex64)

    for i in range(atom_type.shape[0]):
        nd = f_anomalous_data_sizes[i]
        Z = atom_type[i]
        fr = frel[i]
        f_data = f_anomalous_data[i,:,:]
        xp = f_data[:nd,0]
        yp = f_data[:nd,1]
        ypp = f_data[:nd,2]

        f1 = np.interp(wavelength,xp,yp)
        f2 = np.interp(wavelength,xp,ypp)

        f_anam[i] = np.complex(f1+fr-Z,f2)
    return f_anam

@numba_njit_if_available(cache=True, nogil=True)
def _calcxrayformfactor(wavelength,
    s, 
    atom_type,
    scatfac, 
    fNT, 
    frel, 
    f_anomalous_data,
    f_anomalous_data_sizes):

    f_anomalous = _calcanomalousformfactor(atom_type,
                  wavelength,
                  frel,
                  f_anomalous_data,
                  f_anomalous_data_sizes)
    ff = np.zeros(atom_type.shape,dtype=np.complex64)
    for ii in range(atom_type.shape[0]):
        sfact = scatfac[ii,:]
        fe = sfact[5]
        for jj in range(5):
            fe += sfact[jj] * np.exp(-sfact[jj+6]*s)

        ff[ii] = fe+fNT[ii]+f_anomalous[ii]

    return ff


@numba_njit_if_available(cache=True, nogil=True, parallel=True)
def _calcxrsf(hkls,
              nref,
              multiplicity,
              w_int,
              wavelength,
              rmt,
              atom_type,
              atom_ntype,
              betaij,
              occ,
              asym_pos_arr,
              numat,
              scatfac,
              fNT,
              frel,
              f_anomalous_data,
              f_anomalous_data_sizes):

    struct_factors = np.zeros(multiplicity.shape,
        dtype=np.float64)

    for ii in prange(nref):
        g = hkls[ii,:]
        mm = multiplicity[ii]
        glen = np.dot(g,np.dot(rmt,g))
        s = 0.25 * glen * 1E-2
        sf = np.complex(0., 0.)
        formfact = _calcxrayformfactor(wavelength,
             s, 
             atom_type,
             scatfac, 
             fNT, 
             frel, 
             f_anomalous_data,
             f_anomalous_data_sizes)

        for jj in range(atom_ntype):
            natom = numat[jj]
            apos = asym_pos_arr[:natom,jj,:]
            if betaij.ndim > 1:
                b = betaij[:,:,jj]
                arg = b[0,0]*g[0]**2+\
                b[1,1]*g[1]**2+\
                b[2,2]*g[2]**2+\
                2.0*(b[0,1]*g[0]*g[1]+\
                    b[0,2]*g[0]*g[2]+\
                    b[1,2]*g[1]*g[2])
                arg = -arg
            else:
                arg = -8.0*np.pi**2 * betaij[jj]*s

            T = np.exp(arg)
            ff = formfact[jj]*occ[jj]*T

            for kk in range(natom):
                r = apos[kk,:]
                arg = 2.0 * np.pi * np.sum(g*r)
                sf = sf + ff*np.complex(np.cos(arg), -np.sin(arg))

        struct_factors[ii] = w_int*mm*np.abs(sf)**2

    ma = struct_factors.max()
    struct_factors = 100.0*struct_factors/ma

    return struct_factors