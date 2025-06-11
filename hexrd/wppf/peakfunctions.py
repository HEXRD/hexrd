# ============================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# ============================================================

import numpy as np
import copy
from hexrd import constants
from numba import vectorize, float64, njit, prange
from hexrd.fitting.peakfunctions import erfc, exp1exp

# from scipy.special import erfc, exp1

# addr = get_cython_function_address("scipy.special.cython_special", "exp1")
# functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
# exp1_fn = functype(addr)

gauss_width_fact = constants.sigma_to_fwhm
lorentz_width_fact = 2.0

# FIXME: we need this for the time being to be able to parse multipeak fitting
# results; need to wrap all this up in a class in the future!
mpeak_nparams_dict = {
    'gaussian': 3,
    'lorentzian': 3,
    'pvoigt': 4,
    'split_pvoigt': 6,
}

"""
Calgliotti and Lorentzian FWHM functions
"""


@njit(cache=True, nogil=True)
def _gaussian_fwhm(uvw, P, gamma_ani_sqr, eta_mixing, tth, dsp):
    """
    @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    @DATE:       05/20/2020 SS 1.0 original
    @DETAILS:    calculates the fwhm of gaussian component
                 uvw cagliotti parameters
                 P Scherrer broadening (generally not refined)
                 gamma_ani_sqr anisotropic broadening term
                 eta_mixing mixing factor
                 tth two theta in degrees
                 dsp d-spacing
    """
    U, V, W = uvw
    th = np.radians(0.5 * tth)
    tanth = np.tan(th)
    cth2 = np.cos(th) ** 2.0
    sig2_ani = gamma_ani_sqr * (1.0 - eta_mixing) ** 2 * dsp**4
    sigsqr = (U + sig2_ani) * tanth**2 + V * tanth + W + P / cth2
    if sigsqr <= 0.0:
        sigsqr = 1.0e-12

    return np.sqrt(sigsqr) * 1e-2


@njit(cache=True, nogil=True)
def _lorentzian_fwhm(xy, xy_sf, gamma_ani_sqr, eta_mixing, tth, dsp):
    """
    @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    @DATE:       07/20/2020 SS 1.0 original
    @DETAILS:    calculates the size and strain broadening for Lorentzian peak
                xy [X, Y] lorentzian broadening
                xy_sf stacking fault dependent peak broadening
                gamma_ani_sqr anisotropic broadening term
                eta_mixing mixing factor
                tth two theta in degrees
                dsp d-spacing
                is_in_sublattice if hkl in sublattice, then extra broadening
                else regular broadening
    """
    X, Y = xy
    th = np.radians(0.5 * tth)
    tanth = np.tan(th)
    cth = np.cos(th)
    sig_ani = np.sqrt(gamma_ani_sqr) * eta_mixing * dsp**2
    gamma = (X + xy_sf) / cth + (Y + sig_ani) * tanth
    return gamma * 1e-2


@njit(cache=True, nogil=True)
def _anisotropic_peak_broadening(shkl, hkl):
    """
    this function generates the broadening as
    a result of anisotropic broadening. details in
    P.Stephens, J. Appl. Cryst. (1999). 32, 281-289
    a total of 15 terms, some of them zero. in this
    function, we will just use all the terms. it is
    assumed that the user passes on the correct values
    for shkl with appropriate zero values
    The order is the same as the wppfsupport._shkl_name
    variable
    ["s400", "s040", "s004", "s220", "s202", "s022",
              "s310", "s103", "s031", "s130", "s301", "s013",
              "s211", "s121", "s112"]
    """
    # l_val is just l, but l is an ambiguous variable name, looks like I
    h, k, l_val = hkl
    gamma_sqr = (
        shkl[0] * h**4
        + shkl[1] * k**4
        + shkl[2] * l_val**4
        + 3.0
        * (
            shkl[3] * (h * k) ** 2
            + shkl[4] * (h * l_val) ** 2
            + shkl[5] * (k * l_val) ** 2
        )
        + 2.0
        * (
            shkl[6] * k * h**3
            + shkl[7] * h * l_val**3
            + shkl[8] * l_val * k**3
            + shkl[9] * h * k**3
            + shkl[10] * l_val * h**3
            + shkl[11] * k * l_val**3
        )
        + 4.0
        * (
            shkl[12] * k * l_val * h**2
            + shkl[13] * h * l_val * k**2
            + shkl[14] * h * k * l_val**2
        )
    )

    return gamma_sqr


def _anisotropic_gaussian_component(gamma_sqr, eta_mixing):
    """
    gaussian component in anisotropic broadening
    """
    return gamma_sqr * (1.0 - eta_mixing) ** 2


def _anisotropic_lorentzian_component(gamma_sqr, eta_mixing):
    """
    lorentzian component in anisotropic broadening
    """
    return np.sqrt(gamma_sqr) * eta_mixing


# =============================================================================
# 1-D Gaussian Functions
# =============================================================================
# Split the unit gaussian so this can be called for 2d and 3d functions


@njit(cache=True, nogil=True)
def _unit_gaussian(p, x):
    """
    Required Arguments:
    p -- (m) [x0,FWHM]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    x0 = p[0]
    FWHM = p[1]
    sigma = FWHM / gauss_width_fact

    f = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2.0))
    return f


# =============================================================================
# 1-D Lorentzian Functions
# =============================================================================
# Split the unit function so this can be called for 2d and 3d functions
@njit(cache=True, nogil=True)
def _unit_lorentzian(p, x):
    """
    Required Arguments:
    p -- (m) [x0,FWHM]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    x0 = p[0]
    FWHM = p[1]
    gamma = FWHM / lorentz_width_fact

    f = gamma / ((x - x0) ** 2 + gamma**2)
    return f

@njit(cache=True, nogil=True)
def _heaviside(x, x0):
    y = np.zeros_like(x)
    for ii in np.arange(x.size):
        if x[ii] < 0.:
            y[ii] = 0.
        elif x[ii] == 0.:
            y[ii] = x0
        else:
            y[ii] = 1.

    return y


# =====================================================
# 1-D split gaussian functions
# =====================================================
# split gaussian function for asymmetric peaks
@njit(cache=True, nogil=True)
def _split_unit_gaussian(p, x):
    '''
    Parameters
    ----------
    p : numpy.ndarray
        numpy of size (3,) with (cent, fwhm_l, fwhm_r)
    x : numpy.ndarray
        coordinate positions

    Returns
    -------
    y : numpy.ndarray
        intensity of split gaussian function
    '''
    x0     = p[0]
    fwhm_l = p[1]
    fwhm_r = p[2]

    sigma_l = fwhm_l / gauss_width_fact
    sigma_r = fwhm_r / gauss_width_fact

    heav_l = _heaviside(x0-x, 1.0)
    heav_r = _heaviside(x-x0, 1.0)

    p_l = np.array([x0, fwhm_l])
    p_r = np.array([x0, fwhm_r])

    gauss_l = _unit_gaussian(p_l, x)
    gauss_r = _unit_gaussian(p_r, x)

    return (gauss_l*heav_l + 
            gauss_r*heav_r)

# =========================================================
# 1-D split pseudo-voight functions
# =========================================================
# split pseudo-voight function for asymmetric peaks
@njit(cache=True, nogil=True)
def _split_unit_pv(p, x):
    '''
    Parameters
    ----------
    p : numpy.ndarray
        numpy of size (3,) with (cent, fwhm_g_l, fwhm_l_l,
        fwhm_g_r, fwhm_l_r)
    x : numpy.ndarray
        coordinate positions

    Returns
    -------
    y : numpy.ndarray
        intensity of split gaussian function
    '''

    '''get the mixing factors for the left and right
    pseudo-voight functions
    '''
    # center
    x0 = p[0]

    # left branch
    fwhm_g_l = p[1]
    fwhm_l_l = p[2]

    # right branch
    fwhm_g_r = p[3]
    fwhm_l_r = p[4]

    heav_l = _heaviside(x0-x, 1.0)
    heav_r = _heaviside(x-x0, 1.0)

    eta_l, fwhm_l = _mixing_factor_pv(fwhm_g_l, fwhm_l_l)
    eta_r, fwhm_r = _mixing_factor_pv(fwhm_g_r, fwhm_l_r)

    # Ag_l = 0.9394372787 / fwhm_l  # normalization factor for unit area
    # Ag_r = 0.9394372787 / fwhm_r  # normalization factor for unit area
    Al = 1.0 / np.pi  # normalization factor for unit area

    gamma_l = fwhm_l / lorentz_width_fact
    gamma_r = fwhm_r / lorentz_width_fact

    g_l = _unit_gaussian(np.array([x0, fwhm_l]), x)
    l_l = _unit_lorentzian(np.array([x0, fwhm_l]), x)*gamma_l

    g_r = _unit_gaussian(np.array([x0, fwhm_r]), x)
    l_r = _unit_lorentzian(np.array([x0, fwhm_r]), x)*gamma_r

    pv_l = eta_l * l_l + (1.0 - eta_l) * g_l

    pv_r = eta_r * l_r + (1.0 - eta_r) * g_r

    return pv_l*heav_l + pv_r*heav_r

@njit(cache=True, nogil=True)
def _mixing_factor_pv(fwhm_g, fwhm_l):
    """
    @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
    saransh1@llnl.gov
    @DATE: 05/20/2020 SS 1.0 original
           01/29/2021 SS 2.0 updated to depend only on fwhm of profile
           P. Thompson, D.E. Cox & J.B. Hastings, J. Appl. Cryst.,20,79-83,
           1987
    @DETAILS: calculates the mixing factor eta to best approximate voight
    peak shapes
    """
    fwhm = (
        fwhm_g**5
        + 2.69269 * fwhm_g**4 * fwhm_l
        + 2.42843 * fwhm_g**3 * fwhm_l**2
        + 4.47163 * fwhm_g**2 * fwhm_l**3
        + 0.07842 * fwhm_g * fwhm_l**4
        + fwhm_l**5
    )

    fwhm = fwhm**0.20
    eta = (
        1.36603 * (fwhm_l / fwhm)
        - 0.47719 * (fwhm_l / fwhm) ** 2
        + 0.11116 * (fwhm_l / fwhm) ** 3
    )
    if eta < 0.0:
        eta = 0.0
    elif eta > 1.0:
        eta = 1.0

    return eta, fwhm


@njit(cache=True, nogil=True)
def pvoight_wppf(uvw, p, xy, xy_sf, shkl, eta_mixing, tth, dsp, hkl, tth_list):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/22/2021 SS 1.0 original
    @details pseudo voight peak profile for WPPF
    """
    gamma_ani_sqr = _anisotropic_peak_broadening(shkl, hkl)
    fwhm_g = _gaussian_fwhm(uvw, p, gamma_ani_sqr, eta_mixing, tth, dsp)
    fwhm_l = _lorentzian_fwhm(xy, xy_sf, gamma_ani_sqr, eta_mixing, tth, dsp)

    n, fwhm = _mixing_factor_pv(fwhm_g, fwhm_l)

    Ag = 0.9394372787 / fwhm  # normalization factor for unit area
    Al = 1.0 / np.pi  # normalization factor for unit area

    g = Ag * _unit_gaussian(np.array([tth, fwhm]), tth_list)
    l_val = Al * _unit_lorentzian(np.array([tth, fwhm]), tth_list)

    return n * l_val + (1.0 - n) * g


@njit(cache=True, nogil=True)
def _func_h(tau, tth_r):
    cph = np.cos(tth_r - tau)
    ctth = np.cos(tth_r)
    return np.sqrt((cph / ctth) ** 2 - 1.0)


@njit(cache=True, nogil=True)
def _func_W(HoL, SoL, tau, tau_min, tau_infl, tth):

    if tth < np.pi / 2.0:
        if tau >= 0.0 and tau <= tau_infl:
            res = 2.0 * min(HoL, SoL)
        elif tau > tau_infl and tau <= tau_min:
            res = HoL + SoL + _func_h(tau, tth)
        else:
            res = 0.0
    else:
        if tau <= 0.0 and tau >= tau_infl:
            res = 2.0 * min(HoL, SoL)
        elif tau < tau_infl and tau >= tau_min:
            res = HoL + SoL + _func_h(tau, tth)
        else:
            res = 0.0
    return res


@njit(cache=True, nogil=True)
def pvfcj(
    uvw,
    p,
    xy,
    xy_sf,
    shkl,
    eta_mixing,
    tth,
    dsp,
    hkl,
    tth_list,
    HoL,
    SoL,
    xn,
    wn,
):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 04/02/2021 SS 1.0 original
    @details pseudo voight convolved with the slit functions
    xn, wn are the Gauss-Legendre weights and abscissae
    supplied using the scipy routine
    """

    # angle of minimum
    tth_r = np.radians(tth)
    ctth = np.cos(tth_r)

    arg = ctth * np.sqrt(((HoL + SoL) ** 2 + 1.0))
    cinv = np.arccos(arg)
    tau_min = tth_r - cinv

    # two theta of inflection point
    arg = ctth * np.sqrt(((HoL - SoL) ** 2 + 1.0))
    cinv = np.arccos(arg)
    tau_infl = tth_r - cinv

    tau = tau_min * xn

    cx = np.cos(tau)
    res = np.zeros(tth_list.shape)
    den = 0.0

    for i in np.arange(tau.shape[0]):
        x = tth_r - tau[i]
        xx = tau[i]

        W = _func_W(HoL, SoL, xx, tau_min, tau_infl, tth_r)
        h = _func_h(xx, tth_r)
        fact = wn[i] * (W / h / cx[i])
        den += fact

        pv = pvoight_wppf(
            uvw,
            p,
            xy,
            xy_sf,
            shkl,
            eta_mixing,
            np.degrees(x),
            dsp,
            hkl,
            tth_list,
        )
        res += pv * fact

    res = np.sin(tth_r) * res / den / 4.0 / HoL / SoL
    a = np.trapz(res, tth_list)
    return res / a


@njit(cache=True, nogil=True)
def _calc_alpha(alpha, tth):
    a0, a1 = alpha
    return a0 + a1 * np.tan(np.radians(0.5 * tth))


@njit(cache=True, nogil=True)
def _calc_beta(beta, tth):
    b0, b1 = beta
    return b0 + b1 * np.tan(np.radians(0.5 * tth))


@njit(cache=True, nogil=True)
def _gaussian_pink_beam(alpha, beta, fwhm_g, tth, tth_list):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/22/2021 SS 1.0 original
    @details the gaussian component of the pink beam peak profile
    obtained by convolution of gaussian with normalized back to back
    exponentials. more details can be found in
    Von Dreele et. al., J. Appl. Cryst. (2021). 54, 3–6
    """
    del_tth = tth_list - tth
    sigsqr = fwhm_g**2
    f1 = alpha * sigsqr + 2.0 * del_tth
    f2 = beta * sigsqr - 2.0 * del_tth
    f3 = np.sqrt(2.0) * fwhm_g
    u = 0.5 * alpha * f1
    v = 0.5 * beta * f2
    y = (f1 - del_tth) / f3
    z = (f2 + del_tth) / f3
    t1 = erfc(y)
    t2 = erfc(z)
    g = np.zeros(tth_list.shape)
    zmask = np.abs(del_tth) > 5.0
    g[~zmask] = (0.5 * (alpha * beta) / (alpha + beta)) * np.exp(
        u[~zmask]
    ) * t1[~zmask] + np.exp(v[~zmask]) * t2[~zmask]
    mask = np.isnan(g)
    g[mask] = 0.0

    return g


@njit(cache=True, nogil=True)
def _lorentzian_pink_beam(alpha, beta, fwhm_l, tth, tth_list):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/22/2021 SS 1.0 original
    @details the lorentzian component of the pink beam peak profile
    obtained by convolution of gaussian with normalized back to back
    exponentials. more details can be found in
    Von Dreele et. al., J. Appl. Cryst. (2021). 54, 3–6
    """
    del_tth = tth_list - tth
    p = -alpha * del_tth + 1j * 0.5 * alpha * fwhm_l
    q = -beta * del_tth + 1j * 0.5 * beta * fwhm_l

    y = np.zeros(tth_list.shape)

    f1 = exp1exp(p)
    f2 = exp1exp(q)
    # f1 = exp1(p)
    # f2 = exp1(q)

    y = -(alpha * beta) / (np.pi * (alpha + beta)) * (f1 + f2).imag

    mask = np.isnan(y)
    y[mask] = 0.0

    return y


@njit(cache=True, nogil=True)
def pvoight_pink_beam(
    alpha, beta, uvw, p, xy, xy_sf, shkl, eta_mixing, tth, dsp, hkl, tth_list
):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/22/2021 SS 1.0 original
    @details compute the pseudo voight peak shape for the pink
    beam using von dreele's function
    """
    alpha_exp = _calc_alpha(alpha, tth)
    beta_exp = _calc_beta(beta, tth)

    gamma_ani_sqr = _anisotropic_peak_broadening(shkl, hkl)

    fwhm_g = _gaussian_fwhm(uvw, p, gamma_ani_sqr, eta_mixing, tth, dsp)
    fwhm_l = _lorentzian_fwhm(xy, xy_sf, gamma_ani_sqr, eta_mixing, tth, dsp)

    n, fwhm = _mixing_factor_pv(fwhm_g, fwhm_l)

    g = _gaussian_pink_beam(alpha_exp, beta_exp, fwhm_g, tth, tth_list)
    l_val = _lorentzian_pink_beam(alpha_exp, beta_exp, fwhm_l, tth, tth_list)
    ag = np.trapz(g, tth_list)
    al = np.trapz(l_val, tth_list)
    if np.abs(ag) < 1e-6:
        ag = 1.0
    if np.abs(al) < 1e-6:
        al = 1.0

    return n * l_val / al + (1.0 - n) * g / ag


@njit(cache=True, nogil=True, parallel=True)
def computespectrum_pvfcj(
    uvw,
    p,
    xy,
    xy_sf,
    shkl,
    eta_mixing,
    HL,
    SL,
    tth,
    dsp,
    hkl,
    tth_list,
    Iobs,
    xn,
    wn,
):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/31/2021 SS 1.0 original
    @details compute the spectrum given all the input parameters.
    moved outside of the class to allow numba implementation
    this is called for multiple wavelengths and phases to generate
    the final spectrum
    """

    spec = np.zeros(tth_list.shape)
    nref = np.min(
        np.array([Iobs.shape[0], tth.shape[0], dsp.shape[0], hkl.shape[0]])
    )
    for ii in prange(nref):

        II = Iobs[ii]
        t = tth[ii]
        d = dsp[ii]
        g = hkl[ii]
        xs = xy_sf[ii]

        pv = pvfcj(
            uvw, p, xy, xs, shkl, eta_mixing, t, d, g, tth_list, HL, SL, xn, wn
        )

        spec += II * pv
    return spec


@njit(cache=True, nogil=True, parallel=True)
def computespectrum_pvtch(
    uvw, p, xy, xy_sf, shkl, eta_mixing, tth, dsp, hkl, tth_list, Iobs
):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/31/2021 SS 1.0 original
    @details compute the spectrum given all the input parameters.
    moved outside of the class to allow numba implementation
    this is called for multiple wavelengths and phases to generate
    the final spectrum
    """

    spec = np.zeros(tth_list.shape)
    nref = np.min(
        np.array([Iobs.shape[0], tth.shape[0], dsp.shape[0], hkl.shape[0]])
    )
    for ii in prange(nref):

        II = Iobs[ii]
        t = tth[ii]
        d = dsp[ii]
        g = hkl[ii]
        xs = xy_sf[ii]

        pv = pvoight_wppf(uvw, p, xy, xs, shkl, eta_mixing, t, d, g, tth_list)

        spec += II * pv
    return spec


@njit(cache=True, nogil=True, parallel=True)
def computespectrum_pvpink(
    alpha,
    beta,
    uvw,
    p,
    xy,
    xy_sf,
    shkl,
    eta_mixing,
    tth,
    dsp,
    hkl,
    tth_list,
    Iobs,
):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/31/2021 SS 1.0 original
    @details compute the spectrum given all the input parameters.
    moved outside of the class to allow numba implementation
    this is called for multiple wavelengths and phases to generate
    the final spectrum
    """

    spec = np.zeros(tth_list.shape)
    nref = np.min(
        np.array([Iobs.shape[0], tth.shape[0], dsp.shape[0], hkl.shape[0]])
    )
    for ii in prange(nref):

        II = Iobs[ii]
        t = tth[ii]
        d = dsp[ii]
        g = hkl[ii]
        xs = xy_sf[ii]

        pv = pvoight_pink_beam(
            alpha, beta, uvw, p, xy, xs, shkl, eta_mixing, t, d, g, tth_list
        )

        spec += II * pv
    return spec


@njit(cache=True, nogil=True)
def calc_Iobs_pvfcj(
    uvw,
    p,
    xy,
    xy_sf,
    shkl,
    eta_mixing,
    HL,
    SL,
    xn,
    wn,
    tth,
    dsp,
    hkl,
    tth_list,
    Icalc,
    spectrum_expt,
    spectrum_sim,
):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/31/2021 SS 1.0 original
    @details compute Iobs for each reflection given all parameters.
    moved outside of the class to allow numba implementation
    this is called for multiple wavelengths and phases to compute
    the final intensities
    """
    Iobs = np.zeros(tth.shape)
    nref = np.min(
        np.array([Icalc.shape[0], tth.shape[0], dsp.shape[0], hkl.shape[0]])
    )

    yo = spectrum_expt[:, 1]
    yc = spectrum_sim[:, 1]
    mask = yc != 0.0
    yo = yo[mask]
    yc = yc[mask]
    tth_list_mask = spectrum_expt[:, 0]
    tth_list_mask = tth_list_mask[mask]

    for ii in np.arange(nref):
        Ic = Icalc[ii]
        t = tth[ii]
        d = dsp[ii]
        g = hkl[ii]
        xs = xy_sf[ii]

        pv = pvfcj(
            uvw,
            p,
            xy,
            xs,
            shkl,
            eta_mixing,
            t,
            d,
            g,
            tth_list_mask,
            HL,
            SL,
            xn,
            wn,
        )

        y = Ic * pv
        y = y[mask]

        Iobs[ii] = np.trapz(yo * y / yc, tth_list_mask)

    return Iobs


@njit(cache=True, nogil=True)
def calc_Iobs_pvtch(
    uvw,
    p,
    xy,
    xy_sf,
    shkl,
    eta_mixing,
    tth,
    dsp,
    hkl,
    tth_list,
    Icalc,
    spectrum_expt,
    spectrum_sim,
):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/31/2021 SS 1.0 original
    @details compute Iobs for each reflection given all parameters.
    moved outside of the class to allow numba implementation
    this is called for multiple wavelengths and phases to compute
    the final intensities
    """
    Iobs = np.zeros(tth.shape)
    nref = np.min(
        np.array([Icalc.shape[0], tth.shape[0], dsp.shape[0], hkl.shape[0]])
    )

    yo = spectrum_expt[:, 1]
    yc = spectrum_sim[:, 1]
    mask = yc != 0.0
    yo = yo[mask]
    yc = yc[mask]
    tth_list_mask = spectrum_expt[:, 0]
    tth_list_mask = tth_list_mask[mask]

    for ii in np.arange(nref):
        Ic = Icalc[ii]
        t = tth[ii]
        d = dsp[ii]
        g = hkl[ii]
        xs = xy_sf[ii]

        pv = pvoight_wppf(
            uvw, p, xy, xs, shkl, eta_mixing, t, d, g, tth_list_mask
        )

        y = Ic * pv
        y = y[mask]

        Iobs[ii] = np.trapz(yo * y / yc, tth_list_mask)

    return Iobs


@njit(cache=True, nogil=True)
def calc_Iobs_pvpink(
    alpha,
    beta,
    uvw,
    p,
    xy,
    xy_sf,
    shkl,
    eta_mixing,
    tth,
    dsp,
    hkl,
    tth_list,
    Icalc,
    spectrum_expt,
    spectrum_sim,
):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/31/2021 SS 1.0 original
    @details compute Iobs for each reflection given all parameters.
    moved outside of the class to allow numba implementation
    this is called for multiple wavelengths and phases to compute
    the final intensities
    """
    Iobs = np.zeros(tth.shape)
    nref = np.min(
        np.array([Icalc.shape[0], tth.shape[0], dsp.shape[0], hkl.shape[0]])
    )

    yo = spectrum_expt[:, 1]
    yc = spectrum_sim[:, 1]
    mask = yc != 0.0
    yo = yo[mask]
    yc = yc[mask]
    tth_list_mask = spectrum_expt[:, 0]
    tth_list_mask = tth_list_mask[mask]

    for ii in prange(nref):
        Ic = Icalc[ii]
        t = tth[ii]
        d = dsp[ii]
        g = hkl[ii]
        xs = xy_sf[ii]

        pv = pvoight_pink_beam(
            alpha,
            beta,
            uvw,
            p,
            xy,
            xs,
            shkl,
            eta_mixing,
            t,
            d,
            g,
            tth_list_mask,
        )

        y = Ic * pv
        y = y[mask]

        Iobs[ii] = np.trapz(yo * y / yc, tth_list_mask)

    return Iobs


@njit(cache=True, nogil=True)
def calc_rwp(spectrum_sim, spectrum_expt, weights, P):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/31/2021 SS 1.0 original
    @details calculate the rwp given all the input parameters.
    moved outside of the class to allow numba implementation
    P : number of independent parameters in fitting
    """
    err = weights[:, 1] * (spectrum_sim[:, 1] - spectrum_expt[:, 1]) ** 2

    weighted_expt = weights[:, 1] * spectrum_expt[:, 1] ** 2

    errvec = np.sqrt(err)

    """ weighted sum of square """
    wss = np.sum(err)
    den = np.sum(weighted_expt)

    """ standard Rwp i.e. weighted residual """
    if den > 0.0:
        if wss / den > 0.0:
            Rwp = np.sqrt(wss / den)
        else:
            Rwp = np.inf
    else:
        Rwp = np.inf

    """ number of observations to fit i.e. number of data points """
    N = spectrum_sim.shape[0]

    if den > 0.0:
        if (N - P) / den > 0:
            Rexp = np.sqrt((N - P) / den)
        else:
            Rexp = 0.0
    else:
        Rexp = np.inf

    # Rwp and goodness of fit parameters
    if Rexp > 0.0:
        gofF = Rwp / Rexp
    else:
        gofF = np.inf

    return errvec, Rwp, gofF
