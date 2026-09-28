import numpy as np
import pytest
from scipy.special import roots_legendre

from hexrd.powder.wppf import peakfunctions as wpf


@pytest.mark.parametrize("tth", [10.0, 120.0])
def test_fcj_sloped_weight_matches_equation_7a(tth):
    hol = 0.01
    sol = 0.008
    tth_r = np.radians(tth)
    ctth = np.cos(tth_r)
    tau_min = tth_r - np.arccos(ctth * np.sqrt((hol + sol) ** 2 + 1.0))
    tau_infl = tth_r - np.arccos(ctth * np.sqrt((hol - sol) ** 2 + 1.0))
    tau = 0.5 * (tau_min + tau_infl)

    expected = hol + sol - wpf._func_h(tau, tth_r)
    assert np.isclose(
        wpf._func_W(hol, sol, tau, tau_min, tau_infl, tth_r), expected
    )


def test_pvfcj_matches_equation_9_centroid():
    # Use a purely Gaussian intrinsic profile so its centroid is the
    # quadrature-weighted mean of the FCJ integration angles.
    uvw = np.array([0.0, 0.0, 4.0])
    xy = np.zeros(2)
    shkl = np.zeros(15)
    hkl = np.ones(3)
    tth = 80.0
    hol = 0.1
    sol = 0.08
    tth_list = np.linspace(79.0, 81.0, 1001)

    xn, wn = roots_legendre(32)
    profile = wpf.pvfcj(
        uvw,
        0.0,
        xy,
        0.0,
        shkl,
        0.5,
        tth,
        2.0,
        hkl,
        tth_list,
        hol,
        sol,
        xn[16:],
        wn[16:],
    )

    # Independently evaluate equations (4)-(9) with a high-order,
    # conventionally mapped Gauss-Legendre rule.
    ref_xn, ref_wn = roots_legendre(2048)
    ref_xn = 0.5 * (ref_xn + 1.0)
    ref_wn *= 0.5
    tth_r = np.radians(tth)
    ctth = np.cos(tth_r)
    tau_min = tth_r - np.arccos(ctth * np.sqrt((hol + sol) ** 2 + 1.0))
    tau_infl = tth_r - np.arccos(ctth * np.sqrt((hol - sol) ** 2 + 1.0))
    tau = tau_min * ref_xn
    integration_angle = tth_r - tau
    h = np.sqrt((np.cos(integration_angle) / ctth) ** 2 - 1.0)
    weight = np.where(tau <= tau_infl, 2.0 * min(hol, sol), hol + sol - h)
    quadrature_weight = ref_wn * weight / h / np.cos(integration_angle)
    expected_centroid = np.degrees(
        np.sum(quadrature_weight * integration_angle) / np.sum(quadrature_weight)
    )

    assert np.isclose(np.trapezoid(profile, tth_list), 1.0)
    centroid = np.trapezoid(profile * tth_list, tth_list)
    assert np.isclose(centroid, expected_centroid, atol=2e-5)
