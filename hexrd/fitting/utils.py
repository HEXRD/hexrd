import numpy as np

from hexrd.constants import (
    c_erf, cnum_exp1exp, cden_exp1exp, c_coeff_exp1exp, sqrt_epsf
)
from hexrd.matrixutil import uniqueVectors
from hexrd.utils.decorators import numba_njit_if_available


# =============================================================================
# LMFIT Parameter munging utilities
# =============================================================================


def _parameter_arg_constructor(pdict, pargs):
    return [i + pargs for i in pdict.items()]


def _extract_parameters_by_name(params, pname_root):
    return [s for s in params.keys() if s.__contains__(pname_root)]


def _set_refinement_by_name(params, pname_root, vary=True):
    target_pnames = _extract_parameters_by_name(params, pname_root)
    if len(target_pnames) > 0:
        for pname in target_pnames:
            params[pname].vary = vary
    else:
        raise RuntimeWarning("Only 1 parameter found; exiting")


def _set_equality_constraints(params, pname_spec):
    if isinstance(pname_spec, str):
        target_pnames = _extract_parameters_by_name(params, pname_spec)
        if len(target_pnames) > 0:
            for pname in target_pnames[1:]:
                params[pname].expr = target_pnames[0]
        else:
            raise RuntimeWarning("Only 1 parameter found; exiting")
    else:
        for name_pair in pname_spec:
            assert len(name_pair) == 2, \
                "entries in name spec must be 2-tuples"
            params[name_pair[0]].expr = name_pair[1]


def _set_bound_constraints(params, pname_spec,
                           min_val=-np.inf, max_val=np.inf,
                           box=None, percentage=False):
    target_pnames = _extract_parameters_by_name(params, pname_spec)
    for pname in target_pnames:
        if box is None:
            params[pname].min = min_val
            params[pname].max = max_val
        else:
            hval = 0.5*box
            if percentage:
                hval = 0.5*abs(params[pname].value*(box/100.))
            params[pname].min = params[pname].value - hval
            params[pname].max = params[pname].value + hval



def _set_width_mixing_bounds(params, min_w=0.01, max_w=np.inf):
    for pname, param in params.items():
        if 'fwhm' in pname:
            param.min = min_w
            param.max = max_w
        if 'mixing' in pname:
            param.min = 0.
            param.max = 1.


def _set_peak_center_bounds(params, window_range, min_sep=0.01):
    target_pnames = _extract_parameters_by_name(params, 'cen')
    npks = len(target_pnames)
    if npks > 1:
        center_values = []
        for pname in target_pnames:
            # will need these to sort the peaks with increasing centers
            center_values.append(params[pname].value)

            # force peaks to be in window (with buffer)
            params[pname].min = window_range[0] + min_sep
            params[pname].max = window_range[1] - min_sep

        # make sure peak list does not include any peaks closer than min_sep
        uvec = uniqueVectors(
            np.atleast_2d(center_values), tol=min_sep
        ).squeeze()
        if len(uvec) < npks:
            raise RuntimeError(
                "Params contain peaks separated by <="
                + " the specified min, %d" % min_sep
            )

        # get the sorted indices
        peak_order = np.argsort(center_values)

        # sort parameter names
        sorted_pnames = np.asarray(target_pnames)[peak_order].tolist()

        # add new parameter to fit peak separations
        prev_peak = params[sorted_pnames[0]]

        # add parameters to fit subsequent centers as separations
        for ip, pname in enumerate(sorted_pnames[1:]):
            curr_peak = params[pname]
            new_pname = 'pksep%d' % ip
            params.add(name=new_pname,
                       value=curr_peak.value - prev_peak.value,
                       min=min_sep,
                       max=window_range[1] - window_range[0],
                       vary=True)
            curr_peak.expr = '+'.join([prev_peak.name, new_pname])
            prev_peak = curr_peak
    else:
        msg = "Found only 1 peak; setting no bounds"
        # print(msg)
        # raise RuntimeWarning(msg)


# =============================================================================
# DCS-related utilities
# =============================================================================

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
    g *= A

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
    y *= A

    return y

# =============================================================================
# pseudo-Voigt
# =============================================================================
