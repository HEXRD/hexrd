import numpy as np
from scipy.signal import fftconvolve

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
