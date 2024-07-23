import numpy as np
from numba import njit, prange

from hexrd import constants


@njit(cache=True, nogil=True)
def _calc_dspacing(rmt, hkls):
    nhkls = hkls.shape[0]
    dsp = np.zeros(hkls.shape[0])

    for ii in np.arange(nhkls):
        g = hkls[ii, :]
        dsp[ii] = 1.0 / np.sqrt(np.dot(g, np.dot(rmt, g)))
    return dsp


@njit(cache=True, nogil=True)
def _get_tth(dsp, wavelength):
    nn = dsp.shape[0]
    tth = np.zeros(dsp.shape[0])
    wavelength_allowed_hkls = np.zeros(dsp.shape[0])
    for ii in np.arange(nn):
        d = dsp[ii]
        glen = 1.0 / d
        sth = glen * wavelength / 2.0
        if np.abs(sth) <= 1.0:
            t = 2.0 * np.degrees(np.arcsin(sth))
            tth[ii] = t
            wavelength_allowed_hkls[ii] = 1
        else:
            tth[ii] = np.nan
            wavelength_allowed_hkls[ii] = 0

    return tth, wavelength_allowed_hkls


@njit(cache=True, nogil=True)
def _calcanomalousformfactor(
    atom_type, wavelength, frel, f_anomalous_data, f_anomalous_data_sizes
):

    f_anam = np.zeros(atom_type.shape, dtype=np.complex64)

    for i in range(atom_type.shape[0]):
        nd = f_anomalous_data_sizes[i]
        Z = atom_type[i]
        fr = frel[i]
        f_data = f_anomalous_data[i, :, :]
        xp = f_data[:nd, 0]
        yp = f_data[:nd, 1]
        ypp = f_data[:nd, 2]

        f1 = np.interp(wavelength, xp, yp)
        f2 = np.interp(wavelength, xp, ypp)

        f_anam[i] = complex(f1 + fr - Z, f2)
    return f_anam


@njit(cache=True, nogil=True)
def _calcxrayformfactor(
    wavelength,
    s,
    atom_type,
    scatfac,
    fNT,
    frel,
    f_anomalous_data,
    f_anomalous_data_sizes,
):
    """we are using the following form factors for 
       x-aray scattering:
        1. coherent x-ray scattering, f0 tabulated in 
           Acta Cryst. (1995). A51,416-431
        2. Anomalous x-ray scattering (complex (f'+if")) 
           tabulated in J. Phys. Chem. Ref. Data, 24, 71 (1995)
           and J. Phys. Chem. Ref. Data, 29, 597 (2000).
        3. Thompson nuclear scattering, fNT tabulated in 
           Phys. Lett. B, 69, 281 (1977).

        the anomalous scattering is a complex number (f' + if"), 
        where the two terms are given by:
        f' = f1 + frel - Z
        f" = f2

        f1 and f2 have been tabulated as a function of energy in 
        Anomalous.h5 in hexrd folder

        overall f = (f0 + f' + if" +fNT)
    """

    f_anomalous = _calcanomalousformfactor(
        atom_type, wavelength, frel, f_anomalous_data, f_anomalous_data_sizes
    )
    ff = np.zeros(atom_type.shape, dtype=np.complex64)
    for ii in range(atom_type.shape[0]):
        sfact = scatfac[ii, :]
        fe = sfact[5]
        for jj in range(5):
            fe += sfact[jj] * np.exp(-sfact[jj + 6] * s)

        ff[ii] = fe + fNT[ii] + f_anomalous[ii]

    return ff


@njit(cache=True, nogil=True, parallel=True)
def _calcxrsf(
    hkls,
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
    f_anomalous_data_sizes,
):

    struct_factors = np.zeros(multiplicity.shape, dtype=np.float64)

    struct_factors_raw = np.zeros(multiplicity.shape, dtype=np.float64)

    for ii in prange(nref):
        g = hkls[ii, :]
        mm = multiplicity[ii]
        glen = np.dot(g, np.dot(rmt, g))
        s = 0.25 * glen * 1e-2
        sf = complex(0.0, 0.0)
        formfact = _calcxrayformfactor(
            wavelength,
            s,
            atom_type,
            scatfac,
            fNT,
            frel,
            f_anomalous_data,
            f_anomalous_data_sizes,
        )

        for jj in range(atom_ntype):
            natom = numat[jj]
            apos = asym_pos_arr[:natom, jj, :]
            if betaij.ndim > 1:
                b = betaij[:, :, jj]
                arg = (
                    b[0, 0] * g[0] ** 2
                    + b[1, 1] * g[1] ** 2
                    + b[2, 2] * g[2] ** 2
                    + 2.0
                    * (
                        b[0, 1] * g[0] * g[1]
                        + b[0, 2] * g[0] * g[2]
                        + b[1, 2] * g[1] * g[2]
                    )
                )
                arg = -arg
            else:
                arg = -8.0 * np.pi**2 * betaij[jj] * s

            T = np.exp(arg)
            ff = formfact[jj] * occ[jj] * T

            for kk in range(natom):
                r = apos[kk, :]
                arg = 2.0 * np.pi * np.sum(g * r)
                sf = sf + ff * complex(np.cos(arg), -np.sin(arg))

        struct_factors_raw[ii] = np.abs(sf) ** 2
        struct_factors[ii] = w_int * mm * struct_factors_raw[ii]

    # ma = struct_factors.max()
    # struct_factors = 100.0*struct_factors/ma
    # ma = struct_factors_raw.max()
    # struct_factors_raw = 100.0*struct_factors_raw/ma
    return struct_factors, struct_factors_raw


@njit(cache=True, nogil=True)
def _calc_x_factor(K, v_unitcell, wavelength, f_sqr, D):
    return f_sqr * (K * wavelength * D / v_unitcell) ** 2


@njit(cache=True, nogil=True)
def _calc_bragg_factor(x, tth):
    stth = np.sin(np.radians(tth * 0.5)) ** 2
    return stth / np.sqrt(1.0 + x)


@njit(cache=True, nogil=True)
def _calc_laue_factor(x, tth):
    ctth = np.cos(np.radians(tth * 0.5)) ** 2
    if x <= 1.0:
        El = (
            1.0
            - 0.5 * x
            + 0.25 * x**2
            - (5.0 / 48.0) * x**3
            + (7.0 / 192.0) * x**4
        )
    elif x > 1.0:
        El = (2.0 / np.pi / x) ** 2 * (
            1.0
            - (1 / 8.0 / x)
            - (3.0 / 128.0) * (1.0 / x) ** 2
            - (15.0 / 1024.0) * (1 / x) ** 3
        )
    return El * ctth


@njit(cache=True, nogil=True, parallel=True)
def _calc_extinction_factor(hkls, tth, v_unitcell, wavelength, f_sqr, K, D):
    nref = np.min(np.array([hkls.shape[0], tth.shape[0]]))

    extinction = np.zeros(nref)

    for ii in prange(nref):
        fs = f_sqr[ii]
        t = tth[ii]
        x = _calc_x_factor(K, v_unitcell, wavelength, fs, D)
        extinction[ii] = _calc_bragg_factor(x, t) + _calc_laue_factor(x, t)

    return extinction


@njit(cache=True, nogil=True, parallel=True)
def _calc_absorption_factor(abs_fact, tth, phi, wavelength):
    nref = tth.shape[0]
    absorption = np.zeros(nref)
    phir = np.radians(phi)

    abl = -abs_fact * wavelength
    for ii in prange(nref):
        t = np.radians(tth[ii]) * 0.5

        if np.abs(phir) > 1e-3:
            c1 = np.cos(t + phir)
            c2 = np.cos(t - phir)

            f1 = np.exp(abl / c1)
            f2 = np.exp(abl / c2)
            if np.abs(c2) > 1e-3:
                f3 = abl * (1.0 - c1 / c2)
            else:
                f3 = np.inf

            absorption[ii] = (f1 - f2) / f3
        else:
            c = np.cos(t)
            absorption[ii] = np.exp(abl / c)
    return absorption
