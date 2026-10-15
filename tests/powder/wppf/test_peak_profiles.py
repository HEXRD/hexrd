import numpy as np
from scipy.signal import fftconvolve
from scipy.special import roots_legendre

from hexrd.powder.wppf import peakfunctions as wpf


def _normalized(y, x):
    return y / np.trapezoid(y, x)


def _convolve(x, a, b):
    dx = x[1] - x[0]
    return _normalized(fftconvolve(a, b, mode='same') * dx, x)


def _rel_err(y, ref, x):
    y = _normalized(y, x)
    return np.max(np.abs(y - ref)) / ref.max()


def test_pink_beam_matches_numeric_convolution():
    # Both components of the pink beam profile must be the convolution of
    # a Gaussian or Lorentzian with the same back to back exponential
    x = np.arange(-8, 8, 2e-4)
    alpha, beta, fwhm = 8.0, 2.0, 0.15
    b2b = np.where(x < 0, np.exp(alpha * x), np.exp(-beta * x))

    gamma = 0.5 * fwhm
    lorentzian = gamma / np.pi / (x**2 + gamma**2)
    ref = _convolve(x, lorentzian, b2b)
    y = wpf._lorentzian_pink_beam(alpha, beta, fwhm, 0.0, x)
    assert _rel_err(y, ref, x) < 1e-3

    sigma = fwhm / 2.354820045
    ref = _convolve(x, np.exp(-(x**2) / (2 * sigma**2)), b2b)
    y = wpf._gaussian_pink_beam(alpha, beta, sigma, 0.0, x)
    assert _rel_err(y, ref, x) < 1e-3



def test_asymmetric_profiles_match_convolved_pvtch():
    # The Gaussian and Lorentzian parts of pvexponential and pvheating must
    # be those of pvtch (same widths) convolved with the asymmetry kernel
    x = np.linspace(-6, 6, 120001)
    tau, sigma = 0.1, 0.1
    kernels = [
        (wpf.pvoight_exponential, [tau, 0.0, 0.0], np.exp(-x / tau)),
        (wpf.pvoight_heating, [sigma, 0.0], np.exp(-(x**2) / (2 * sigma**2))),
    ]
    widths = [
        # Pure Gaussian
        (np.array([0.0, 0.0, 25.0]), np.array([0.0, 0.0])),
        # Pure Lorentzian
        (np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0])),
    ]
    for uvw, xy in widths:
        args = (np.float64(0.0), xy, np.float64(0.0), np.zeros(15), np.float64(0.5))
        args += (np.float64(0.0), np.float64(0.2), np.array([1.0, 1.0, 1.0]), x)
        pvtch = wpf.pvoight_wppf(uvw, *args)
        for func, coeffs, kernel in kernels:
            ref = _convolve(x, pvtch, np.where(x > 0, kernel, 0.0))
            y = func(np.array(coeffs), uvw, *args)
            assert _rel_err(y, ref, x) < 2e-3, func.__name__


def test_pvfcj_low_angle():
    # With large H/L and S/L at low 2theta, the FCJ arccos argument
    # exceeds 1. The profile must stay finite and normalized.
    xn, wn = roots_legendre(16)
    xn, wn = xn[8:], wn[8:]
    x = np.linspace(0.5, 20, 4000)
    args = (np.array([0.0, 0.0, 100.0]), 0.0, np.array([1.0, 1.0]), 0.0)
    args += (np.zeros(15), 0.5)
    for tth in [3.0, 8.0, 15.0]:
        y = wpf.pvfcj(*args, tth, 0.2, np.array([1.0, 1.0, 1.0]), x, 0.05, 0.05, xn, wn)
        assert np.all(np.isfinite(y))
        assert np.isclose(np.trapezoid(y, x), 1.0)
        assert tth - 2 < x[np.argmax(y)] <= tth
