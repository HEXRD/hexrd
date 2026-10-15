import numpy as np
import pytest
from scipy.signal import fftconvolve
from scipy.special import roots_legendre

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


@pytest.mark.parametrize('tth, hol, sol', [(60.0, 0.3, 0.3), (120.0, 0.3, 0.2)])
def test_pvfcj_matches_direct_integration(tth: float, hol: float, sol: float) -> None:
    # The FCJ profile is the pseudo-Voigt integrated over the observed
    # angle 2phi, with the weight W / (h |cos(2phi)|)
    args = (np.array([0.0, 0.0, 2500.0]), 0.0, np.zeros(2), 0.0, np.zeros(15), 0.5)
    x = np.linspace(tth - 12, tth + 12, 3001)
    xn, wn = roots_legendre(64)
    y = wpf.pvfcj(*args, tth, 0.2, np.ones(3), x, hol, sol, xn[32:], wn[32:])

    # Sampling the offset from 2theta as tau_max * u^2 removes the
    # 1 / sqrt(tau) singularity of 1 / h
    u = np.linspace(0, 1, 20001)[1:]
    tau = np.radians(12) * u**2
    phi = np.radians(tth) - tau if tth < 90 else np.radians(tth) + tau
    h = np.sqrt(np.clip((np.cos(phi) / np.cos(np.radians(tth))) ** 2 - 1, 0, None))
    W = np.where(h <= abs(hol - sol), 2 * min(hol, sol), hol + sol - h)
    expected = np.zeros_like(x)
    for p, w in zip(phi[W > 0], (W / h / np.abs(np.cos(phi)) * u)[W > 0]):
        expected += w * wpf.pvoight_wppf(*args, np.degrees(p), 0.2, np.ones(3), x)

    expected = normalized(expected, x)
    assert np.abs(normalized(y, x) - expected).max() < 1e-2 * expected.max()
