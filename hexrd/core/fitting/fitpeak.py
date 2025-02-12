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
# the terms of the GNU Lesser General Public License (as published by the
# Free Software Foundation) version 2.1 dated February 1999.
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
# from numpy.polynomial import chebyshev

from scipy import integrate
from scipy import ndimage as imgproc
from scipy import optimize

from hexrd.core import constants
from hexrd.core.imageutil import snip1d
from hexrd.core.fitting import peakfunctions as pkfuncs

import matplotlib.pyplot as plt


# =============================================================================
# Helper Functions and Module Vars
# =============================================================================

ftol = constants.sqrt_epsf
xtol = constants.sqrt_epsf

inf = np.inf
minf = -inf

# dcs param values
# !!! converted from deg^-1 in Von Dreele's paper
alpha0, alpha1, beta0, beta1 = np.r_[14.4, 0., 3.016, -7.94]


def cnst_fit_obj(x, b):
    return np.ones_like(x)*b


def cnst_fit_jac(x, b):
    return np.vstack([np.ones_like(x)]).T


def lin_fit_obj(x, m, b):
    return m*np.asarray(x) + b


def lin_fit_jac(x, m, b):
    return np.vstack([x, np.ones_like(x)]).T


def quad_fit_obj(x, a, b, c):
    x = np.asarray(x)
    return a*x**2 + b*x + c


def quad_fit_jac(x, a, b, c):
    x = np.asarray(x)
    return a*x**2 + b*x + c
    return np.vstack([x**2, x, np.ones_like(x)]).T


def _amplitude_guess(x, x0, y, fwhm):
    pt_l = np.argmin(np.abs(x - (x0 - 0.5*fwhm)))
    pt_h = np.argmin(np.abs(x - (x0 + 0.5*fwhm)))
    return np.max(y[pt_l:pt_h + 1])


# =============================================================================
# 1-D Peak Fitting
# =============================================================================


def estimate_pk_parms_1d(x, f, pktype='pvoigt'):
    """
    Gives initial guess of parameters for analytic fit of one dimensional peak
    data.

    Required Arguments:
    x -- (n) ndarray of coordinate positions
    f -- (n) ndarray of intensity measurements at coordinate positions x
    pktype -- string, type of analytic function that will be used to fit the
    data, current options are "gaussian", "lorentzian",
    "pvoigt" (psuedo voigt), and "split_pvoigt" (split psuedo voigt)

    Outputs:
    p -- (m) ndarray containing initial guesses for parameters for the input
    peaktype
    (see peak function help for what each parameters corresponds to)

    Notes
    -----
    !!! LINEAR BACKGROUND ONLY
    !!! ASSUMES ANGULAR SPECTRA IN RADIANS (DCS PARAMS)
    """
    npts = len(x)
    assert len(f) == npts, "ordinate and data must be same length!"

    # handle background
    # ??? make kernel width a kwarg?
    bkg = snip1d(np.atleast_2d(f), w=int(2*npts/3.)).flatten()

    # fit linear bg and grab params
    bp, _ = optimize.curve_fit(lin_fit_obj, x, bkg, jac=lin_fit_jac)
    bg0 = bp[1]
    bg1 = bp[0]

    # set remaining params
    pint = f - lin_fit_obj(x, *bp)
    cen_index = np.argmax(pint)
    A = pint[cen_index]
    x0 = x[cen_index]

    # fix center index
    if cen_index > 0 and cen_index < npts - 1:
        left_hm = np.argmin(abs(pint[:cen_index] - 0.5*A))
        right_hm = np.argmin(abs(pint[cen_index:] - 0.5*A))
    elif cen_index == 0:
        right_hm = np.argmin(abs(pint[cen_index:] - 0.5*A))
        left_hm = right_hm
    elif cen_index == npts - 1:
        left_hm = np.argmin(abs(pint[:cen_index] - 0.5*A))
        right_hm = left_hm

    # FWHM estimation
    try:
        FWHM = x[cen_index + right_hm] - x[left_hm]
    except(IndexError):
        FWHM = 0
    if FWHM <= 0 or FWHM > 0.75*npts:
        # something is weird, so punt...
        FWHM = 0.25*(x[-1] - x[0])

    # set params
    if pktype in ['gaussian', 'lorentzian']:
        p = [A, x0, FWHM, bg0, bg1]
    elif pktype == 'pvoigt':
        p = [A, x0, FWHM, 0.5, bg0, bg1]
    elif pktype == 'split_pvoigt':
        p = [A, x0, FWHM, FWHM, 0.5, 0.5, bg0, bg1]
    elif pktype == 'pink_beam_dcs':
        # A, x0, alpha0, alpha1, beta0, beta1, fwhm_g, fwhm_l
        p = [A, x0, alpha0, alpha1, beta0, beta1, FWHM, FWHM, bg0, bg1]
    else:
        raise RuntimeError("pktype '%s' not understood" % pktype)

    return np.r_[p]


def fit_pk_parms_1d(p0, x, f, pktype='pvoigt'):
    """
    Performs least squares fit to find parameters for 1d analytic functions fit
    to diffraction data

    Required Arguments:
    p0 -- (m) ndarray containing initial guesses for parameters
              for the input peaktype
    x -- (n) ndarray of coordinate positions
    f -- (n) ndarray of intensity measurements at coordinate positions x
    pktype -- string, type of analytic function that will be used to
                      fit the data,
    current options are "gaussian","lorentzian","pvoigt" (psuedo voigt), and
    "split_pvoigt" (split psuedo voigt)


    Outputs:
    p -- (m) ndarray containing fit parameters for the input peaktype
    (see peak function help for what each parameters corresponds to)


    Notes:
    1. Currently no checks are in place to make sure that the guess of
    parameters has a consistent number of parameters with the requested
    peak type
    """

    weight = np.max(f)*10.  # hard coded should be changed
    fitArgs = (x, f, pktype)
    if pktype == 'gaussian':
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d, p0,
            args=fitArgs, Dfun=eval_pk_deriv_1d,
            ftol=ftol, xtol=xtol
        )
    elif pktype == 'lorentzian':
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d, p0,
            args=fitArgs, Dfun=eval_pk_deriv_1d,
            ftol=ftol, xtol=xtol
        )
    elif pktype == 'pvoigt':
        lb = [p0[0]*0.5, np.min(x), 0., 0., 0., None]
        ub = [p0[0]*2.0, np.max(x), 4.*p0[2], 1., 2.*p0[4], None]

        fitArgs = (x, f, pktype, weight, lb, ub)
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d_bnded, p0,
            args=fitArgs,
            ftol=ftol, xtol=xtol
        )
    elif pktype == 'split_pvoigt':
        lb = [p0[0]*0.5, np.min(x), 0., 0., 0., 0., 0., None]
        ub = [p0[0]*2.0, np.max(x), 4.*p0[2], 4.*p0[2], 1., 1., 2.*p0[4], None]
        fitArgs = (x, f, pktype, weight, lb, ub)
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d_bnded, p0,
            args=fitArgs,
            ftol=ftol, xtol=xtol
        )
    elif pktype == 'tanh_stepdown':
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d, p0,
            args=fitArgs,
            ftol=ftol, xtol=xtol)

    elif pktype == 'dcs_pinkbeam':
        # !!!: for some reason the 'trf' method was not behaving well,
        #      so switched to 'lm'
        lb = np.array([0.0, x.min(), -100., -100.,
                       -100., -100., 0., 0.,
                       -np.inf, -np.inf, -np.inf])
        ub = np.array([np.inf, x.max(), 100., 100.,
                       100., 100., 10., 10.,
                       np.inf, np.inf, np.inf])
        res = optimize.least_squares(
            fit_pk_obj_1d, p0,
            jac='2-point',
            # bounds=(),  # (lb, ub),
            method='lm',
            args=fitArgs,
            ftol=ftol,
            xtol=xtol)
        p = res['x']
        # outflag = res['success']
    else:
        p = p0
        print('non-valid option, returning guess')

    if np.any(np.isnan(p)):
        p = p0
        print('failed fitting, returning guess')

    return p


def fit_mpk_parms_1d(
        p0, x, f0, pktype, num_pks,
        bgtype=None, bnds=None
        ):
    """
    Fit MULTIPLE 1d analytic functions to diffraction data.

    Parameters
    ----------
    p0 : array_like (m x u + v, )
        list of peak parameters for number of peaks where m is the number of
        parameters per peak ("gaussian" and "lorentzian" - 3, "pvoigt" - 4,
        "split_pvoigt" - 5), v is the number of parameters for chosen bgtype.
    x : array_like
        (n, ) ndarray of evaluation coordinate positions.
    f0 : TYPE
        DESCRIPTION.
    pktype : TYPE
        DESCRIPTION.
    num_pks : int
        The number of peaks in the interval defined by x.
    bgtype : string, optional
        bBckground function flag. Available options are "constant",
        "linear", and "quadratic". The default is None.
    bnds : array_like, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    array
        (m x u, ) fit peak parameters where m is the number of
        parameters per peak ("gaussian" and "lorentzian" - 3, "pvoigt" - 4,
        "split_pvoigt" - 5).

    """
    fitArgs = (x, f0, pktype, num_pks, bgtype)

    if bnds is None:
        p = optimize.least_squares(
            fit_mpk_obj_1d, p0,
            args=fitArgs, ftol=ftol, xtol=xtol
        )
    else:
        p = optimize.least_squares(
            fit_mpk_obj_1d, p0,
            bounds=bnds, args=fitArgs, ftol=ftol, xtol=xtol
        )
    return p.x


def estimate_mpk_parms_1d(
        pk_pos_0, x, f,
        pktype='pvoigt', bgtype='linear',
        fwhm_guess=None, center_bnd=0.02,
        amp_lim_mult=[0.1, 10.], fwhm_lim_mult=[0.5, 2.]
        ):
    """
    Generate function-specific estimate for multi-peak parameters.

    Parameters
    ----------
    pk_pos_0 : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    pktype : TYPE, optional
        DESCRIPTION. The default is 'pvoigt'.
    bgtype : TYPE, optional
        DESCRIPTION. The default is 'linear'.
    fwhm_guess : TYPE, optional
        DESCRIPTION. The default is 0.07.
    center_bnd : TYPE, optional
        DESCRIPTION. The default is 0.02.
    amp_lim_mult : TYPE, optional
        Multiplier for bounds for AMP fitting [LB, UB].
        The default is [0.1, 10.0].
    fwhm_lim_mult : TYPE, optional
        Multiplier for bounds for FWHM fitting [LB, UB].
        The default is [0.5, 2.0].

    Returns
    -------
    p0 : TYPE
        DESCRIPTION.
    bnds : TYPE
        DESCRIPTION.

    """
    npts = len(x)
    assert len(f) == npts, "ordinate and data must be same length!"

    num_pks = len(pk_pos_0)

    center_bnd = np.atleast_1d(center_bnd)
    if(len(center_bnd) < 2):
        center_bnd = center_bnd*np.ones(num_pks)

    if fwhm_guess is None:
        fwhm_guess = (np.max(x) - np.min(x))/(20.*num_pks)
    fwhm_guess = np.atleast_1d(fwhm_guess)
    if(len(fwhm_guess) < 2):
        fwhm_guess = fwhm_guess*np.ones(num_pks)

    min_val = np.min(f)

    # estimate background with SNIP1d
    bkg = snip1d(np.atleast_2d(f),
                 w=int(np.floor(0.25*len(f)))).flatten()

    # fit linear bg and grab params
    bp, _ = optimize.curve_fit(lin_fit_obj, x, bkg, jac=lin_fit_jac)
    bg0 = bp[1]
    bg1 = bp[0]

    '''
    # TODO: In case we want to switch to chebyshev
    bkg_mod = chebyshev.Chebyshev(
        [0., 0.], domain=(min(x), max(x))
    )
    fit_bkg = bkg_mod.fit(x, bkg, 1)
    coeff = fit_bkg.coef
    bg0, bg1 = coeff
    '''

    # make lin bkg subtracted spectrum
    fsubtr = f - lin_fit_obj(x, *bp)
    # !!! for chebyshev
    # fsubtr = f - fit_bkg(x)

    # number of parmaters from reference dict
    npp = pkfuncs.mpeak_nparams_dict[pktype]

    p0tmp = np.zeros([num_pks, npp])
    p0tmp_lb = np.zeros([num_pks, npp])
    p0tmp_ub = np.zeros([num_pks, npp])

    # case processing
    # !!! used to use (f[pt] - min_val) for ampl
    if pktype == 'gaussian' or pktype == 'lorentzian':
        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            amp_guess = _amplitude_guess(
                x, pk_pos_0[ii], fsubtr, fwhm_guess[ii]
            )
            p0tmp[ii, :] = [
                amp_guess,
                pk_pos_0[ii],
                fwhm_guess[ii]
            ]
            p0tmp_lb[ii, :] = [
                amp_guess*amp_lim_mult[0],
                pk_pos_0[ii] - center_bnd[ii],
                fwhm_guess[ii]*fwhm_lim_mult[0]
            ]
            p0tmp_ub[ii, :] = [
                amp_guess*amp_lim_mult[1],
                pk_pos_0[ii] + center_bnd[ii],
                fwhm_guess[ii]*fwhm_lim_mult[1]
            ]
    elif pktype == 'pvoigt':
        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            amp_guess = _amplitude_guess(
                x, pk_pos_0[ii], fsubtr, fwhm_guess[ii]
            )
            p0tmp[ii, :] = [
                amp_guess,
                pk_pos_0[ii],
                fwhm_guess[ii],
                0.5
            ]
            p0tmp_lb[ii, :] = [
                amp_guess*amp_lim_mult[0],
                pk_pos_0[ii] - center_bnd[ii],
                fwhm_guess[ii]*fwhm_lim_mult[0],
                0.0
            ]
            p0tmp_ub[ii, :] = [
                (amp_guess - min_val + 1.)*amp_lim_mult[1],
                pk_pos_0[ii] + center_bnd[ii],
                fwhm_guess[ii]*fwhm_lim_mult[1],
                1.0
            ]
    elif pktype == 'split_pvoigt':
        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            amp_guess = _amplitude_guess(
                x, pk_pos_0[ii], fsubtr, fwhm_guess[ii]
            )
            p0tmp[ii, :] = [
                amp_guess,
                pk_pos_0[ii],
                fwhm_guess[ii],
                fwhm_guess[ii],
                0.5,
                0.5
            ]
            p0tmp_lb[ii, :] = [
                amp_guess*amp_lim_mult[0],
                pk_pos_0[ii] - center_bnd[ii],
                fwhm_guess[ii]*fwhm_lim_mult[0],
                fwhm_guess[ii]*fwhm_lim_mult[0],
                0.0,
                0.0
            ]
            p0tmp_ub[ii, :] = [
                amp_guess*amp_lim_mult[1],
                pk_pos_0[ii] + center_bnd[ii],
                fwhm_guess[ii]*fwhm_lim_mult[1],
                fwhm_guess[ii]*fwhm_lim_mult[1],
                1.0,
                1.0
            ]
    elif pktype == 'pink_beam_dcs':
        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            amp_guess = _amplitude_guess(
                x, pk_pos_0[ii], fsubtr, fwhm_guess[ii]
            )
            p0tmp[ii, :] = [
                amp_guess,
                pk_pos_0[ii],
                alpha0,
                alpha1,
                beta0,
                beta1,
                fwhm_guess[ii],
                fwhm_guess[ii],
            ]
            p0tmp_lb[ii, :] = [
                amp_guess*amp_lim_mult[0],
                pk_pos_0[ii] - center_bnd[ii],
                -1e5,
                -1e5,
                -1e5,
                -1e5,
                fwhm_guess[ii]*fwhm_lim_mult[0],
                fwhm_guess[ii]*fwhm_lim_mult[0],
            ]
            p0tmp_ub[ii, :] = [
                amp_guess*amp_lim_mult[1],
                pk_pos_0[ii] + center_bnd[ii],
                1e5,
                1e5,
                1e5,
                1e5,
                fwhm_guess[ii]*fwhm_lim_mult[1],
                fwhm_guess[ii]*fwhm_lim_mult[1],
            ]

    num_pk_parms = len(p0tmp.ravel())
    if bgtype == 'constant':
        p0 = np.zeros(num_pk_parms + 1)
        lb = np.zeros(num_pk_parms + 1)
        ub = np.zeros(num_pk_parms + 1)
        p0[:num_pk_parms] = p0tmp.ravel()
        lb[:num_pk_parms] = p0tmp_lb.ravel()
        ub[:num_pk_parms] = p0tmp_ub.ravel()

        p0[-1] = np.average(bkg)
        lb[-1] = minf
        ub[-1] = inf

    elif bgtype == 'linear':
        p0 = np.zeros(num_pk_parms + 2)
        lb = np.zeros(num_pk_parms + 2)
        ub = np.zeros(num_pk_parms + 2)
        p0[:num_pk_parms] = p0tmp.ravel()
        lb[:num_pk_parms] = p0tmp_lb.ravel()
        ub[:num_pk_parms] = p0tmp_ub.ravel()

        p0[-2] = bg0
        p0[-1] = bg1
        lb[-2:] = minf
        ub[-2:] = inf

    elif bgtype == 'quadratic':
        p0 = np.zeros(num_pk_parms + 3)
        lb = np.zeros(num_pk_parms + 3)
        ub = np.zeros(num_pk_parms + 3)
        p0[:num_pk_parms] = p0tmp.ravel()
        lb[:num_pk_parms] = p0tmp_lb.ravel()
        ub[:num_pk_parms] = p0tmp_ub.ravel()

        p0[-3] = bg0
        p0[-2] = bg1
        lb[-3:] = minf
        ub[-3:] = inf

    elif bgtype == 'cubic':
        p0 = np.zeros(num_pk_parms + 4)
        lb = np.zeros(num_pk_parms + 4)
        ub = np.zeros(num_pk_parms + 4)
        p0[:num_pk_parms] = p0tmp.ravel()
        lb[:num_pk_parms] = p0tmp_lb.ravel()
        ub[:num_pk_parms] = p0tmp_ub.ravel()

        p0[-4] = bg0
        p0[-3] = bg1
        lb[-4:] = minf
        ub[-4:] = inf

    return p0, (lb, ub)


def eval_pk_deriv_1d(p, x, y0, pktype):

    if pktype == 'gaussian':
        d_mat = pkfuncs.gaussian1d_deriv(p, x)
    elif pktype == 'lorentzian':
        d_mat = pkfuncs.lorentzian1d_deriv(p, x)

    return d_mat.T


def fit_pk_obj_1d(p, x, f0, pktype):
    """
    Return residual between specified peak function and data.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    f0 : TYPE
        DESCRIPTION.
    pktype : TYPE
        DESCRIPTION.

    Returns
    -------
    resd : TYPE
        DESCRIPTION.

    Notes
    -----
    !!! These objective functions all have a linear background added in their
        definition in peakfuncs
    """

    ww = np.ones(f0.shape)
    if pktype == 'gaussian':
        f = pkfuncs.gaussian1d(p, x)
    elif pktype == 'lorentzian':
        f = pkfuncs.lorentzian1d(p, x)
    elif pktype == 'pvoigt':
        f = pkfuncs.pvoigt1d(p, x)
    elif pktype == 'split_pvoigt':
        f = pkfuncs.split_pvoigt1d(p, x)
    elif pktype == 'tanh_stepdown':
        f = pkfuncs.tanh_stepdown_nobg(p, x)
    elif pktype == 'dcs_pinkbeam':
        f = pkfuncs.pink_beam_dcs(p, x)
        ww = 1./np.sqrt(f0)
        ww[np.isnan(ww)] = 0.0

    resd = (f - f0)*ww
    return resd


def fit_pk_obj_1d_bnded(p, x, f0, pktype, weight, lb, ub):
    if pktype == 'gaussian':
        f = pkfuncs.gaussian1d(p, x)
    elif pktype == 'lorentzian':
        f = pkfuncs.lorentzian1d(p, x)
    elif pktype == 'pvoigt':
        f = pkfuncs.pvoigt1d(p, x)
    elif pktype == 'split_pvoigt':
        f = pkfuncs.split_pvoigt1d(p, x)
    elif pktype == 'dcs_pinkbeam':
        f = pkfuncs.pink_beam_dcs(p, x)
        ww = 1./np.sqrt(f0)
        ww[np.isnan(ww)] = 0.0

    num_data = len(f)
    num_parm = len(p)
    resd = np.zeros(num_data + num_parm)
    # tub bnds implementation

    resd[:num_data] = f - f0
    for ii in range(num_parm):
        if lb[ii] is not None:
            resd[num_data + ii] = \
                weight*np.max([-(p[ii] - lb[ii]), 0., (p[ii] - ub[ii])])

    return resd


def fit_mpk_obj_1d(p, x, f0, pktype, num_pks, bgtype):
    f = pkfuncs.mpeak_1d(p, x, pktype, num_pks, bgtype=bgtype)
    resd = f - f0
    return resd


# =============================================================================
# 2-D Peak Fitting
# =============================================================================


def estimate_pk_parms_2d(x, y, f, pktype):
    """
    Calculate initial parameter values for 2-dimensional peak fitting.

    Parameters
    ----------
    x : array_like
        (n, ) ndarray of coordinate positions for dimension 1
        (numpy.meshgrid formatting).
    y : array_like
        (n, ) ndarray of coordinate positions for dimension 2
        (numpy.meshgrid formatting).
    f : array_like
        (n, ) ndarray of intensity measurements at coordinate
        positions x and y.
    pktype : str
        type of analytic function that will be used to fit the data; current
        options are "gaussian", "gaussian_rot" (gaussian with arbitrary axes)
        and "split_pvoigt_rot" (split psuedo voigt with arbitrary axes).

    Returns
    -------
    p -- (m) ndarray containing initial guesses for parameters for the input
    peak type (see peakfunction help for more information).
    """

    bg0 = np.mean([f[0, 0], f[-1, 0], f[-1, -1], f[0, -1]])
    bg1x = (np.mean([f[-1, -1], f[0, -1]]) - np.mean([f[0, 0], f[-1, 0]])) \
        / (x[0, -1] - x[0, 0])
    bg1y = (np.mean([f[-1, -1], f[-1, 0]]) - np.mean([f[0, 0], f[0, -1]])) \
        / (y[-1, 0] - y[0, 0])

    fnobg = f - (bg0 + bg1x * x + bg1y * y)

    labels, numlabels = imgproc.label(fnobg > 0.5*np.max(fnobg))

    # looks for the largest peak
    areas = np.zeros(numlabels)
    for ii in np.arange(1, numlabels + 1, 1):
        areas[ii - 1] = np.sum(labels == ii)

    peakIndex = np.argmax(areas) + 1

    FWHMx = np.max(x[labels == peakIndex]) - np.min(x[labels == peakIndex])
    FWHMy = np.max(y[labels == peakIndex]) - np.min(y[labels == peakIndex])

    coords = imgproc.maximum_position(fnobg, labels=labels, index=peakIndex)
    A = imgproc.maximum(fnobg, labels=labels, index=peakIndex)
    x0 = x[coords]
    y0 = y[coords]

    if pktype == 'gaussian':
        p = [A, x0, y0, FWHMx, FWHMy, bg0, bg1x, bg1y]
    elif pktype == 'gaussian_rot':
        p = [A, x0, y0, FWHMx, FWHMy, 0., bg0, bg1x, bg1y]
    elif pktype == 'split_pvoigt_rot':
        p = [A, x0, y0, FWHMx, FWHMx, FWHMy, FWHMy,
             0.5, 0.5, 0.5, 0.5, 0., bg0, bg1x, bg1y]
    p = np.array(p)
    return p


def fit_pk_parms_2d(p0, x, y, f, pktype):
    """
    Do least squares fit for 2-d profile functions.

    Parameters
    ----------
    p0 : array_like
        (m, ) ndarray containing initial guesses for parameters for the
        input peak type.
    x : array_like
        (n, ) ndarray of coordinate positions for dimension 1
        (numpy.meshgrid formatting).
    y : array_like
        (n, ) ndarray of coordinate positions for dimension 2
        (numpy.meshgrid formatting).
    f : array_like
        (n, ) ndarray of intensity measurements at coordinate
        positions x and y.
    pktype : str
        type of analytic function that will be used to fit the data; current
        options are "gaussian", "gaussian_rot" (gaussian with arbitrary axes)
        and "split_pvoigt_rot" (split psuedo voigt with arbitrary axes).

    Returns
    -------
    p : array_like
        (m, ) ndarray containing optimal parameters for the input peak type.

    Notes
    -----
    See peak function help for what each parameter corresponds to.

    !!! Currently no checks are in place to make sure that the guess of
    !!! parameters has a consisten number of parameters with the requested
    !!! peak type

    """

    fitArgs = (x, y, f, pktype)

    if pktype == 'gaussian':
        p, outflag = optimize.leastsq(
            fit_pk_obj_2d, p0, args=fitArgs, ftol=ftol, xtol=xtol
        )
    elif pktype == 'gaussian_rot':
        p, outflag = optimize.leastsq(
            fit_pk_obj_2d, p0, args=fitArgs, ftol=ftol, xtol=xtol
        )
    elif pktype == 'split_pvoigt_rot':
        p, outflag = optimize.leastsq(
            fit_pk_obj_2d, p0, args=fitArgs, ftol=ftol, xtol=xtol
        )

    if np.any(np.isnan(p)):
        p = p0

    return p


def fit_pk_obj_2d(p, x, y, f0, pktype):
    if pktype == 'gaussian':
        f = pkfuncs.gaussian2d(p, x, y)
    elif pktype == 'gaussian_rot':
        f = pkfuncs.gaussian2d_rot(p, x, y)
    elif pktype == 'split_pvoigt_rot':
        f = pkfuncs.split_pvoigt2d_rot(p, x, y)
    resd = f - f0
    return resd.flatten()


# =============================================================================
# Extra Utilities
# =============================================================================


def goodness_of_fit(f, f0):
    """
    Calculate two scalar measures of goodness of fit.

    Parameters
    ----------
    f : array_like
        (n, ) ndarray of intensity measurements at coordinate positions.
    f0 : array_like
        (n,) ndarray of fit intensity values at coordinate positions.

    Returns
    -------
    R : float
        goodness of fit measure which is sum(error^2)/sum(meas^2).
    Rw : float
        goodness of fit measure weighted by intensity
        sum(meas*error^2)/sum(meas^3).

    """

    R = np.sum((f - f0)**2) / np.sum(f0**2)
    Rw = np.sum(np.abs(f0 * (f - f0)**2)) / np.sum(np.abs(f0**3))

    return R, Rw


def direct_pk_analysis(x, f,
                       remove_bg=True, low_int=1.,
                       edge_pts=3, pts_per_meas=100):
    """
    Analyze a single peak that is not well matched to any analytic functions


    Parameters
    ----------
    x : array_like
        (n, ) ndarray of coordinate positions.
    f : arrauy_like
        (n, ) ndarray of intensity measurements at coordinate positions x.
    remove_bg : bool, optional
        flag for linear background subtraction. The default is True.
    low_int : float, optional
        Value for area under a peak that defines a lower bound on S/N.
        The default is 1..
    edge_pts : int, optional
        number of points at the edges of the data to use to calculated
        background. The default is 3.
    pts_per_meas : int, optional
        number of interpolated points to place between measurement values.
        The default is 100.

    Returns
    -------
    p : array_like
        array of values containing the integrated intensity, center of mass,
        and FWHM of the peak.
    """
    # TODO: should probably remove the plotting calls here
    plt.plot(x, f)

    # subtract background,  assumed linear
    if remove_bg:
        bg_data = np.hstack((f[:(edge_pts+1)], f[-edge_pts:]))
        bg_pts = np.hstack((x[:(edge_pts+1)], x[-edge_pts:]))

        bg_parm = np.polyfit(bg_pts, bg_data, 1)

        f = f - (bg_parm[0]*x + bg_parm[1])  # pull out high background

        f = f - np.min(f)  # set the minimum to 0

    plt.plot(bg_pts, bg_data, 'x')
    plt.plot(x, f, 'r')

    # make a fine grid of points
    spacing = np.diff(x)[0]/pts_per_meas
    xfine = np.arange(np.min(x), np.max(x) + spacing, spacing)
    ffine = np.interp(xfine, x, f)

    # find max intensity values
    data_max = np.max(f)

    # numerically integrate the peak using the simpson rule
    total_int = integrate.simps(ffine, xfine)

    cen_index = np.argmax(ffine)
    A = data_max

    # center of mass calculation
    # !!! this cutoff value is arbitrary,  maybe set higher?
    if(total_int < low_int):
        com = float('NaN')
        FWHM = float('NaN')
        total_int = total_int
        print('Analysis Failed... Intensity too low')
    else:
        com = np.sum(xfine*ffine)/np.sum(ffine)

        a = np.abs(ffine[cen_index+1:]-A/2.)
        b = np.abs(ffine[:cen_index]-A/2.)

        # this is a check to see if the peak is falling out of the bnds
        if a.size == 0 or b.size == 0:
            com = float('NaN')
            FWHM = float('NaN')
            total_int = total_int
            print('Analysis Failed... Peak is not well defined')
        else:
            """
            calculate positions on the left and right half of peaks at half
            maximum;
            !!! think about changing to full width @ 10% max?
            """
            FWHM = xfine[cen_index + np.argmin(a)] - xfine[np.argmin(b)]

    p = [total_int, com, FWHM]
    p = np.array(p)
    return p


def calc_pk_integrated_intensities(p, x, pktype, num_pks):
    """
    Calculate the area under the curve (integrated intensities) for fit peaks.

    Parameters
    ----------
    p : array_like
        (m x u + v) peak parameters for u peaks, m is the number of
        parameters per peak:
            "gaussian" and "lorentzian" = 3;
            "pvoigt" = 4;
            "split_pvoigt" = 5.
        v is the number of parameters for chosen bgtype
    x : array_like
        (n, ) ndarray of abcissa coordinates.
    pktype : str
        type of analytic function that will be used to fit the data; current
        options are "gaussian", "gaussian_rot" (gaussian with arbitrary axes)
        and "split_pvoigt_rot" (split psuedo voigt with arbitrary axes).
    num_pks : int
        the number of peaks in the specified interval.

    Returns
    -------
    ints : array_like
        (m, ) array of integrated intensity values.
    """

    ints = np.zeros(num_pks)

    if pktype == 'gaussian' or pktype == 'lorentzian':
        p_fit = np.reshape(p[:3*num_pks], [num_pks, 3])
    elif pktype == 'pvoigt':
        p_fit = np.reshape(p[:4*num_pks], [num_pks, 4])
    elif pktype == 'split_pvoigt':
        p_fit = np.reshape(p[:6*num_pks], [num_pks, 6])

    for ii in np.arange(num_pks):
        if pktype == 'gaussian':
            ints[ii] = integrate.simps(
                pkfuncs._gaussian1d_no_bg(p_fit[ii], x),
                x
            )
        elif pktype == 'lorentzian':
            ints[ii] = integrate.simps(
                pkfuncs._lorentzian1d_no_bg(p_fit[ii], x),
                x
            )
        elif pktype == 'pvoigt':
            ints[ii] = integrate.simps(
                pkfuncs._pvoigt1d_no_bg(p_fit[ii], x),
                x
            )
        elif pktype == 'split_pvoigt':
            ints[ii] = integrate.simps(
                pkfuncs._split_pvoigt1d_no_bg(p_fit[ii], x),
                x
            )

    return ints
