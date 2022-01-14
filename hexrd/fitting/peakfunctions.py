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
from hexrd.utils.decorators import numba_njit_if_available
from hexrd.constants import \
    c_erf, cnum_exp1exp, cden_exp1exp, c_coeff_exp1exp

gauss_width_fact = constants.sigma_to_fwhm
lorentz_width_fact = 2.

# FIXME: we need this for the time being to be able to parse multipeak fitting
# results; need to wrap all this up in a class in the future!
mpeak_nparams_dict = {
    'gaussian': 3,
    'lorentzian': 3,
    'pvoigt': 4,
    'split_pvoigt': 6,
    'pink_beam_dcs': 8
}

"""
cutom function to compute the complementary error function
based on rational approximation of the convergent Taylor
series. coefficients found in
Formula 7.1.26
Handbook of Mathematical Functions,
Abramowitz and Stegun
Error is < 1.5e-7 for all x
"""


@numba_njit_if_available(cache=True, nogil=True)
def erfc(x):
    # save the sign of x
    sign = np.sign(x)
    x = np.abs(x)

    # constants
    a1, a2, a3, a4, a5, p = c_erf

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1. - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    erf = sign*y  # erf(-x) = -erf(x)
    return 1. - erf


"""
cutom function to compute the exponential integral
based on Padé approximation of exponential integral
function. coefficients found in pg. 231 Abramowitz
and Stegun, eq. 5.1.53
"""


@numba_njit_if_available(cache=True, nogil=True)
def exp1exp_under1(x):
    f = np.zeros(x.shape).astype(np.complex128)
    for i in range(6):
        xx = x**(i+1)
        f += c_coeff_exp1exp[i]*xx

    return (f - np.log(x) - np.euler_gamma)*np.exp(x)


"""
cutom function to compute the exponential integral
based on Padé approximation of exponential integral
function. coefficients found in pg. 415 Y. Luke, The
special functions and their approximations, vol 2
(1969) Elsevier
"""


@numba_njit_if_available(cache=True, nogil=True)
def exp1exp_over1(x):
    num = np.zeros(x.shape).astype(np.complex128)
    den = np.zeros(x.shape).astype(np.complex128)

    for i in range(11):
        p = 10-i
        if p != 0:
            xx = x**p
            num += cnum_exp1exp[i]*xx
            den += cden_exp1exp[i]*xx
        else:
            num += cnum_exp1exp[i]
            den += cden_exp1exp[i]

    return (num/den)*(1./x)


@numba_njit_if_available(cache=True, nogil=True)
def exp1exp(x):
    mask = np.sign(x.real)*np.abs(x) > 1.

    f = np.zeros(x.shape).astype(np.complex128)
    f[mask] = exp1exp_over1(x[mask])
    f[~mask] = exp1exp_under1(x[~mask])

    return f

# =============================================================================
# 1-D Gaussian Functions
# =============================================================================
# Split the unit gaussian so this can be called for 2d and 3d functions


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
    sigma = FWHM/gauss_width_fact

    f = np.exp(-(x - x0)**2/(2.*sigma**2.))
    return f


def _gaussian1d_no_bg(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    A = p[0]
    f = A*_unit_gaussian(p[[1, 2]], x)
    return f


def gaussian1d(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM,c0,c1]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    bg0 = p[3]
    bg1 = p[4]

    f = _gaussian1d_no_bg(p[:3], x) + bg0 + bg1*x

    return f


def _gaussian1d_no_bg_deriv(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM]
    x -- (n) ndarray of coordinate positions

    Outputs:
    d_mat -- (3 x n) ndarray of derivative values at positions x
    """

    x0 = p[1]
    FWHM = p[2]

    sigma = FWHM/gauss_width_fact

    dydx0 = _gaussian1d_no_bg(p, x)*((x - x0)/(sigma**2.))
    dydA = _unit_gaussian(p[[1, 2]], x)
    dydFWHM = _gaussian1d_no_bg(p, x) \
        * ((x - x0)**2./(sigma**3.))/gauss_width_fact

    d_mat = np.zeros((len(p), len(x)))

    d_mat[0, :] = dydA
    d_mat[1, :] = dydx0
    d_mat[2, :] = dydFWHM

    return d_mat


def gaussian1d_deriv(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM,c0,c1]
    x -- (n) ndarray of coordinate positions

    Outputs:
    d_mat -- (5 x n) ndarray of derivative values at positions x
    """

    d_mat = np.zeros((len(p), len(x)))
    d_mat[0:3, :] = _gaussian1d_no_bg_deriv(p[0:3], x)
    d_mat[3, :] = 1.
    d_mat[4, :] = x

    return d_mat


# =============================================================================
# 1-D Lorentzian Functions
# =============================================================================
# Split the unit function so this can be called for 2d and 3d functions
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
    gamma = FWHM/lorentz_width_fact

    f = gamma**2 / ((x-x0)**2 + gamma**2)
    return f


def _lorentzian1d_no_bg(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    A = p[0]
    f = A*_unit_lorentzian(p[[1, 2]], x)

    return f


def lorentzian1d(p, x):
    """
    Required Arguments:
    p -- (m) [x0,FWHM,c0,c1]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    bg0 = p[3]
    bg1 = p[4]

    f = _lorentzian1d_no_bg(p[:3], x)+bg0+bg1*x

    return f


def _lorentzian1d_no_bg_deriv(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM]
    x -- (n) ndarray of coordinate positions

    Outputs:
    d_mat -- (3 x n) ndarray of derivative values at positions x
    """

    x0 = p[1]
    FWHM = p[2]

    gamma = FWHM/lorentz_width_fact

    dydx0 = _lorentzian1d_no_bg(p, x)*((2.*(x-x0))/((x-x0)**2 + gamma**2))
    dydA = _unit_lorentzian(p[[1, 2]], x)
    dydFWHM = _lorentzian1d_no_bg(p, x) \
        * ((2.*(x-x0)**2.)/(gamma*((x-x0)**2 + gamma**2)))/lorentz_width_fact

    d_mat = np.zeros((len(p), len(x)))
    d_mat[0, :] = dydA
    d_mat[1, :] = dydx0
    d_mat[2, :] = dydFWHM

    return d_mat


def lorentzian1d_deriv(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM,c0,c1]
    x -- (n) ndarray of coordinate positions

    Outputs:
    d_mat -- (5 x n) ndarray of derivative values at positions x
    """

    d_mat = np.zeros((len(p), len(x)))
    d_mat[0:3, :] = _lorentzian1d_no_bg_deriv(p[0:3], x)
    d_mat[3, :] = 1.
    d_mat[4, :] = x

    return d_mat


# =============================================================================
# 1-D Psuedo Voigt Functions
# =============================================================================

# Split the unit function so this can be called for 2d and 3d functions
def _unit_pvoigt1d(p, x):
    """
    Required Arguments:
    p -- (m) [x0,FWHM,n]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    n = p[2]

    f = (n*_unit_gaussian(p[:2], x)+(1.-n)*_unit_lorentzian(p[:2], x))
    return f


def _pvoigt1d_no_bg(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM,n]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    A = p[0]
    f = A*_unit_pvoigt1d(p[[1, 2, 3]], x)
    return f


def pvoigt1d(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM,n,c0,c1]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    bg0 = p[4]
    bg1 = p[5]

    f = _pvoigt1d_no_bg(p[:4], x) + bg0 + bg1*x

    return f


# =============================================================================
# 1-D Split Psuedo Voigt Functions
# =============================================================================

def _split_pvoigt1d_no_bg(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM-,FWHM+,n-,n+]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    A = p[0]
    x0 = p[1]

    f = np.zeros(x.shape[0])

    # Define halves, using gthanorequal and lthan, choice is arbitrary
    xr = x >= x0
    xl = x < x0

    # +
    right = np.where(xr)[0]

    f[right] = A*_unit_pvoigt1d(p[[1, 3, 5]], x[right])

    # -
    left = np.where(xl)[0]
    f[left] = A*_unit_pvoigt1d(p[[1, 2, 4]], x[left])

    return f


def split_pvoigt1d(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,FWHM-,FWHM+,n-,n+,c0,c1]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    bg0 = p[6]
    bg1 = p[7]

    f = _split_pvoigt1d_no_bg(p[:6], x) + bg0 + bg1*x

    return f


"""
================================================================
================================================================
@AUTHOR:    Saransh Singh, Lawrence Livermore National Lab,
            saransh1@llnl.gov
@DATE:      10/18/2021 SS 1.0 original

@DETAILS:   the following functions will be used for single
            peak fits for the DCS pink beam profile function.
            the collection includes the profile function for
            calculating the peak shape as well as derivatives
            w.r.t. the parameters
================================================================
================================================================
"""


@numba_njit_if_available(cache=True, nogil=True)
def _calc_alpha(alpha, x0):
    a0, a1 = alpha
    return (a0 + a1*np.tan(np.radians(0.5*x0)))


@numba_njit_if_available(cache=True, nogil=True)
def _calc_beta(beta, x0):
    b0, b1 = beta
    return b0 + b1*np.tan(np.radians(0.5*x0))


@numba_njit_if_available(cache=True, nogil=True)
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
    fwhm = fwhm_g**5 + 2.69269 * fwhm_g**4 * fwhm_l + \
        2.42843 * fwhm_g**3 * fwhm_l**2 + \
        4.47163 * fwhm_g**2 * fwhm_l**3 +\
        0.07842 * fwhm_g * fwhm_l**4 +\
        fwhm_l**5

    fwhm = fwhm**0.20
    eta = 1.36603 * (fwhm_l/fwhm) - \
        0.47719 * (fwhm_l/fwhm)**2 + \
        0.11116 * (fwhm_l/fwhm)**3
    if eta < 0.:
        eta = 0.
    elif eta > 1.:
        eta = 1.

    return eta, fwhm


@numba_njit_if_available(nogil=True)
def _gaussian_pink_beam(p, x):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/22/2021 SS 1.0 original
    @details the gaussian component of the pink beam peak profile
    obtained by convolution of gaussian with normalized back to back
    exponentials. more details can be found in
    Von Dreele et. al., J. Appl. Cryst. (2021). 54, 3–6

    p has the following parameters
    p = [A,x0,alpha0,alpha1,beta0,beta1,fwhm_g,bkg_c0,bkg_c1,bkg_c2]
    """

    A, x0, alpha, beta, fwhm_g = p

    del_tth = x - x0
    sigsqr = fwhm_g**2

    f1 = alpha*sigsqr + 2.0*del_tth
    f2 = beta*sigsqr - 2.0*del_tth
    f3 = np.sqrt(2.0)*fwhm_g

    u = 0.5*alpha*f1
    v = 0.5*beta*f2

    y = (f1-del_tth)/f3
    z = (f2+del_tth)/f3

    t1 = erfc(y)
    t2 = erfc(z)

    g = np.zeros(x.shape)
    zmask = np.abs(del_tth) > 5.0

    g[~zmask] = \
        (0.5*(alpha*beta)/(alpha + beta)) * np.exp(u[~zmask])*t1[~zmask] \
        + np.exp(v[~zmask])*t2[~zmask]
    mask = np.isnan(g)
    g[mask] = 0.
    gmax = g.max()
    g *= A/gmax

    return g


@numba_njit_if_available(nogil=True)
def _lorentzian_pink_beam(p, x):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 03/22/2021 SS 1.0 original
    @details the lorentzian component of the pink beam peak profile
    obtained by convolution of gaussian with normalized back to back
    exponentials. more details can be found in
    Von Dreele et. al., J. Appl. Cryst. (2021). 54, 3–6

    p has the following parameters
    p = [A,x0,alpha0,alpha1,beta0,beta1,fwhm_l]
    """

    A, x0, alpha, beta, fwhm_l = p

    del_tth = x - x0

    p = -alpha*del_tth + 1j*0.5*alpha*fwhm_l
    q = -beta*del_tth + 1j*0.5*beta*fwhm_l

    y = np.zeros(x.shape)
    f1 = exp1exp(p)
    f2 = exp1exp(q)

    y = -(alpha*beta)/(np.pi*(alpha + beta))*(f1 + f2).imag

    mask = np.isnan(y)
    y[mask] = 0.
    ymax = y.max()
    y *= A/ymax

    return y


@numba_njit_if_available(nogil=True)
def _pink_beam_dcs_no_bg(p, x):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 10/18/2021 SS 1.0 original
    @details pink beam profile for DCS data for calibration.
    more details can be found in
    Von Dreele et. al., J. Appl. Cryst. (2021). 54, 3–6

    p has the following 10 parameters
    p = [A, x0, alpha0, alpha1, beta0, beta1, fwhm_g, fwhm_l]
    """
    alpha = _calc_alpha((p[2], p[3]), p[1])
    beta = _calc_beta((p[4], p[5]), p[1])

    arg1 = np.array([alpha, beta, p[6]]).astype(np.float64)
    arg2 = np.array([alpha, beta, p[7]]).astype(np.float64)

    p_g = np.hstack((p[0:2], arg1))
    p_l = np.hstack((p[0:2], arg2))

    # bkg = p[8] + p[9]*x + p[10]*(2.*x**2 - 1.)
    # bkg = p[8] + p[9]*x  # !!! make like the other peak funcs here

    eta, fwhm = _mixing_factor_pv(p[6], p[7])

    G = _gaussian_pink_beam(p_g, x)
    L = _lorentzian_pink_beam(p_l, x)

    return eta*L + (1. - eta)*G


def pink_beam_dcs(p, x):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 10/18/2021 SS 1.0 original
    @details pink beam profile for DCS data for calibration.
    more details can be found in
    Von Dreele et. al., J. Appl. Cryst. (2021). 54, 3–6

    p has the following 10 parameters
    p = [A, x0, alpha0, alpha1, beta0, beta1, fwhm_g, fwhm_l, bkg_c0, bkg_c1]
    """
    return _pink_beam_dcs_no_bg(p[:-2], x) + p[-2] + p[-1]*x


def pink_beam_dcs_lmfit(
        x, A, x0, alpha0, alpha1, beta0, beta1, fwhm_g, fwhm_l):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 10/18/2021 SS 1.0 original
    @details pink beam profile for DCS data for calibration.
    more details can be found in
    Von Dreele et. al., J. Appl. Cryst. (2021). 54, 3–6
    """
    alpha = _calc_alpha((alpha0, alpha1), x0)
    beta = _calc_beta((beta0, beta1), x0)

    arg1 = np.array([alpha, beta, fwhm_g], dtype=np.float64)
    arg2 = np.array([alpha, beta, fwhm_l], dtype=np.float64)

    p_g = np.hstack([[A, x0], arg1]).astype(np.float64, order='C')
    p_l = np.hstack([[A, x0], arg2]).astype(np.float64, order='C')

    eta, fwhm = _mixing_factor_pv(fwhm_g, fwhm_l)

    G = _gaussian_pink_beam(p_g, x)
    L = _lorentzian_pink_beam(p_l, x)

    return eta*L + (1. - eta)*G


"""
================================================================
======================FINISHED==================================
================================================================
"""

# =============================================================================
# Tanh Step Down
# =============================================================================


def tanh_stepdown_nobg(p, x):
    """
    Required Arguments:
    p -- (m) [A,x0,w]
    x -- (n) ndarray of coordinate positions

    Outputs:
    f -- (n) ndarray of function values at positions x
    """

    A = p[0]
    x0 = p[1]
    w = p[2]

    f = A*(0.5*(1.-np.tanh((x-x0)/w)))

    return f


# =============================================================================
# 2-D Rotation Coordinate Transform
# =============================================================================

def _2d_coord_transform(theta, x0, y0, x, y):
    xprime = np.cos(theta)*x+np.sin(theta)*y
    yprime = -np.sin(theta)*x+np.cos(theta)*y

    x0prime = np.cos(theta)*x0+np.sin(theta)*y0
    y0prime = -np.sin(theta)*x0+np.cos(theta)*y0

    return x0prime, y0prime, xprime, yprime


# =============================================================================
# 2-D Gaussian Function
# =============================================================================

def _gaussian2d_no_bg(p, x, y):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,FWHMx,FWHMy]
    x -- (n x o) ndarray of coordinate positions for dimension 1
    y -- (n x o) ndarray of coordinate positions for dimension 1

    Outputs:
    f -- (n x 0) ndarray of function values at positions (x,y)
    """

    A = p[0]
    f = A*_unit_gaussian(p[[1, 3]], x)*_unit_gaussian(p[[2, 4]], y)
    return f


def _gaussian2d_rot_no_bg(p, x, y):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,FWHMx,FWHMy,theta]
    x -- (n x o) ndarray of coordinate positions for dimension 1
    y -- (n x o) ndarray of coordinate positions for dimension 2

    Outputs:
    f -- (n x o) ndarray of function values at positions (x,y)
    """

    theta = p[5]

    x0prime, y0prime, xprime, yprime = _2d_coord_transform(
        theta, p[1], p[2], x, y)

    # this copy was needed so original parameters set isn't changed
    newp = copy.copy(p)

    newp[1] = x0prime
    newp[2] = y0prime

    f = _gaussian2d_no_bg(newp[:5], xprime, yprime)

    return f


def gaussian2d_rot(p, x, y):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,FWHMx,FWHMy,theta,c0,c1x,c1y]
    x -- (n x o) ndarray of coordinate positions for dimension 1
    y -- (n x o) ndarray of coordinate positions for dimension 2

    Outputs:
    f -- (n x o) ndarray of function values at positions (x,y)
    """

    bg0 = p[6]
    bg1x = p[7]
    bg1y = p[8]

    f = _gaussian2d_rot_no_bg(p[:6], x, y)+(bg0+bg1x*x+bg1y*y)
    return f


def gaussian2d(p, x, y):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,FWHMx,FWHMy,c0,c1x,c1y]
    x -- (n x o) ndarray of coordinate positions for dimension 1
    y -- (n x o) ndarray of coordinate positions for dimension 2

    Outputs:
    f -- (n x o) ndarray of function values at positions (x,y)
    """

    bg0 = p[5]
    bg1x = p[6]
    bg1y = p[7]

    f = _gaussian2d_no_bg(p[:5], x, y)+(bg0+bg1x*x+bg1y*y)
    return f


# =============================================================================
# 2-D Split Psuedo Voigt Function
# =============================================================================

def _split_pvoigt2d_no_bg(p, x, y):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,FWHMx-,FWHMx+,FWHMy-,FWHMy+,nx-,nx+,ny-,ny+]
    x -- (n x o) ndarray of coordinate positions for dimension 1
    y -- (n x o) ndarray of coordinate positions for dimension 2

    Outputs:
    f -- (n x o) ndarray of function values at positions (x,y)
    """

    A = p[0]
    x0 = p[1]
    y0 = p[2]

    f = np.zeros([x.shape[0], x.shape[1]])

    # Define quadrants, using gthanorequal and lthan, choice is arbitrary
    xr = x >= x0
    xl = x < x0
    yr = y >= y0
    yl = y < y0

    # ++
    q1 = np.where(xr & yr)
    f[q1] = A*_unit_pvoigt1d(p[[1, 4, 8]], x[q1]) * \
        _unit_pvoigt1d(p[[2, 6, 10]], y[q1])

    # +-
    q2 = np.where(xr & yl)
    f[q2] = A*_unit_pvoigt1d(p[[1, 4, 8]], x[q2]) * \
        _unit_pvoigt1d(p[[2, 5, 9]], y[q2])

    # -+
    q3 = np.where(xl & yr)
    f[q3] = A*_unit_pvoigt1d(p[[1, 3, 7]], x[q3]) * \
        _unit_pvoigt1d(p[[2, 6, 10]], y[q3])

    # --
    q4 = np.where(xl & yl)
    f[q4] = A*_unit_pvoigt1d(p[[1, 3, 7]], x[q4]) * \
        _unit_pvoigt1d(p[[2, 5, 9]], y[q4])

    return f


def _split_pvoigt2d_rot_no_bg(p, x, y):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,FWHMx-,FWHMx+,FWHMy-,FWHMy+,nx-,nx+,ny-,ny+,theta]
    x -- (n x o) ndarray of coordinate positions for dimension 1
    y -- (n x o) ndarray of coordinate positions for dimension 2

    Outputs:
    f -- (n x o) ndarray of function values at positions (x,y)
    """

    theta = p[11]

    x0prime, y0prime, xprime, yprime = _2d_coord_transform(
        theta, p[1], p[2], x, y)

    # this copy was needed so original parameters set isn't changed
    newp = copy.copy(p)

    newp[1] = x0prime
    newp[2] = y0prime

    f = _split_pvoigt2d_no_bg(newp[:11], xprime, yprime)

    return f


def split_pvoigt2d_rot(p, x, y):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,FWHMx-,FWHMx+,FWHMy-,FWHMy+,
              nx-,nx+,ny-,ny+,theta,c0,c1x,c1y]
    x -- (n x o) ndarray of coordinate positions for dimension 1
    y -- (n x o) ndarray of coordinate positions for dimension 2

    Outputs:
    f -- (n x o) ndarray of function values at positions (x,y)
    """

    bg0 = p[12]
    bg1x = p[13]
    bg1y = p[14]

    f = _split_pvoigt2d_rot_no_bg(p[:12], x, y)+(bg0+bg1x*x+bg1y*y)

    return f


# =============================================================================
# 3-D Gaussian Function
# =============================================================================

def _gaussian3d_no_bg(p, x, y, z):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,z0,FWHMx,FWHMy,FWHMz]
    x -- (n x o x q) ndarray of coordinate positions for dimension 1
    y -- (n x o x q) ndarray of coordinate positions for dimension 2
    y -- (z x o x q) ndarray of coordinate positions for dimension 3

    Outputs:
    f -- (n x o x q) ndarray of function values at positions (x,y)
    """

    A = p[0]
    f = A * _unit_gaussian(p[[1, 4]], x) \
        * _unit_gaussian(p[[2, 5]], y) \
        * _unit_gaussian(p[[3, 6]], z)
    return f


def gaussian3d(p, x, y, z):
    """
    Required Arguments:
    p -- (m) [A,x0,y0,z0,FWHMx,FWHMy,FWHMz,c0,c1x,c1y,c1z]
    x -- (n x o x q) ndarray of coordinate positions for dimension 1
    y -- (n x o x q) ndarray of coordinate positions for dimension 2
    y -- (z x o x q) ndarray of coordinate positions for dimension 3

    Outputs:
    f -- (n x o x q) ndarray of function values at positions (x,y,z)
    """

    bg0 = p[7]
    bg1x = p[8]
    bg1y = p[9]
    bg1z = p[10]

    f = _gaussian3d_no_bg(p[:5], x, y)+(bg0+bg1x*x+bg1y*y+bg1z*z)
    return f


# =============================================================================
# Mutlipeak
# =============================================================================

def _mpeak_1d_no_bg(p, x, pktype, num_pks):
    """
    Required Arguments:
    p -- (m x u) list of peak parameters for number of peaks
         where m is the number of parameters per peak
             - "gaussian" and "lorentzian" - 3
             - "pvoigt" - 4
             - "split_pvoigt" - 6
    x -- (n) ndarray of coordinate positions for dimension 1
    pktype -- string, type of analytic function; current options are
        "gaussian","lorentzian","pvoigt" (psuedo voigt), and
        "split_pvoigt" (split psuedo voigt)
    num_pks -- integer 'u' indicating the number of pks, must match length of p

    Outputs:
    f -- (n) ndarray of function values at positions (x)
    """

    f = np.zeros(len(x))

    npp = mpeak_nparams_dict[pktype]

    p_fit = np.reshape(p[:npp*num_pks], [num_pks, npp])

    for ii in np.arange(num_pks):
        if pktype == 'gaussian':
            f += _gaussian1d_no_bg(p_fit[ii], x)
        elif pktype == 'lorentzian':
            f += _lorentzian1d_no_bg(p_fit[ii], x)
        elif pktype == 'pvoigt':
            f += _pvoigt1d_no_bg(p_fit[ii], x)
        elif pktype == 'split_pvoigt':
            f += _split_pvoigt1d_no_bg(p_fit[ii], x)
        elif pktype == 'pink_beam_dcs':
            f += _pink_beam_dcs_no_bg(p_fit[ii], x)

    return f


def mpeak_1d(p, x, pktype, num_pks, bgtype=None):
    """
    Required Arguments:
    p -- (m x u) list of peak parameters for number of peaks where m is the
         number of parameters per peak
             "gaussian" and "lorentzian" - 3
             "pvoigt" - 4
             "split_pvoigt" - 6
    x -- (n) ndarray of coordinate positions for dimension 1
    pktype -- string, type of analytic function that will be used;
    current options are "gaussian","lorentzian","pvoigt" (psuedo voigt), and
    "split_pvoigt" (split psuedo voigt)
    num_pks -- integer 'u' indicating the number of pks, must match length of p
    pktype -- string, background functions, available options are "constant",
    "linear", and "quadratic"

    Outputs:
    f -- (n) ndarray of function values at positions (x)
    """
    f = _mpeak_1d_no_bg(p, x, pktype, num_pks)

    if bgtype == 'linear':
        f = f+p[-2]+p[-1]*x  # c0=p[-2], c1=p[-1]
    elif bgtype == 'constant':
        f = f+p[-1]  # c0=p[-1]
    elif bgtype == 'quadratic':
        f = f+p[-3]+p[-2]*x+p[-1]*x**2  # c0=p[-3], c1=p[-2], c2=p[-1],

    return f
