import numpy as np
import pytest
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


@pytest.mark.parametrize(
    'uvw, xy',
    [
        # A pure Gaussian, and a pure Lorentzian
        (np.array([0.0, 0.0, 25.0]), np.zeros(2)),
        (np.zeros(3), np.array([5.0, 0.0])),
    ],
)
def test_asymmetric_profiles_are_convolved_pvtch(
    uvw: np.ndarray, xy: np.ndarray
) -> None:
    # pvexponential and pvheating are pvtch, with the same widths,
    # convolved with their asymmetry kernel
    x = np.linspace(-6, 6, 120001)
    args = (0.0, xy, 0.0, np.zeros(15), 0.5, 0.0, 0.2, np.ones(3), x)
    pvtch = wpf.pvoight_wppf(uvw, *args)
    profiles = [
        (wpf.pvoight_exponential, [0.1, 0.0, 0.0], np.exp(-x / 0.1)),
        (wpf.pvoight_heating, [0.1, 0.0], np.exp(-(x**2) / 0.02)),
    ]
    for func, coeffs, kernel in profiles:
        expected = convolve(x, pvtch, np.where(x > 0, kernel, 0.0))
        y = normalized(func(np.array(coeffs), uvw, *args), x)
        assert np.abs(y - expected).max() < 2e-3 * expected.max(), func.__name__
