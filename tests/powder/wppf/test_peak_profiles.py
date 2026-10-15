import numpy as np
from scipy.signal import fftconvolve

from hexrd.powder.wppf import peakfunctions as wpf


def normalized(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return y / np.trapezoid(y, x)


def convolve(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalized(fftconvolve(a, b, mode='same'), x)


def test_pink_beam_lorentzian_is_a_convolution() -> None:
    # The Lorentzian part should be a Lorentzian convolved with the same
    # back-to-back exponential as the Gaussian part
    x = np.arange(-8, 8, 2e-4)
    alpha, beta, hwhm = 8.0, 2.0, 0.075
    kernel = np.where(x < 0, np.exp(alpha * x), np.exp(-beta * x))
    expected = convolve(x, hwhm / np.pi / (x**2 + hwhm**2), kernel)
    y = normalized(wpf._lorentzian_pink_beam(alpha, beta, 2 * hwhm, 0.0, x), x)
    assert np.abs(y - expected).max() < 1e-3 * expected.max()
