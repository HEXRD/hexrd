import numpy as np
from scipy.special import roots_legendre

from hexrd.powder.wppf.peakfunctions import (
    calc_Iobs_pvfcj,
    computespectrum_pvfcj,
    pvfcj,
)


def test_pvfcj_coincident_reflections_match_individual_profiles():
    uvw = np.array([81.5, 1.0337, 5.18275])
    p = 0.0
    xy = np.array([0.5665, 1.90994])
    xy_sf = np.zeros(3)
    shkl = np.zeros(15)
    eta_mixing = 0.5
    hl = sl = 0.001
    tth = np.array([30.0, 30.0, 45.0])
    dsp = np.array([0.6, 0.6, 0.4])
    hkl = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    intensity = np.array([2.0, 3.0, 4.0])
    grid = np.linspace(20.0, 55.0, 351)
    xn, wn = roots_legendre(16)
    xn = xn[8:]
    wn = wn[8:]

    expected = np.zeros_like(grid)
    for i in range(tth.size):
        expected += intensity[i] * pvfcj(
            uvw,
            p,
            xy,
            xy_sf[i],
            shkl,
            eta_mixing,
            tth[i],
            dsp[i],
            hkl[i],
            grid,
            hl,
            sl,
            xn,
            wn,
        )

    actual = computespectrum_pvfcj(
        uvw,
        p,
        xy,
        xy_sf,
        shkl,
        eta_mixing,
        hl,
        sl,
        tth,
        dsp,
        hkl,
        grid,
        intensity,
        xn,
        wn,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    observed = 10.0 + np.sin(grid) ** 2
    calculated = 9.0 + np.cos(grid) ** 2
    calculated[::17] = 0.0
    spectrum_expt = np.column_stack((grid, observed))
    spectrum_sim = np.column_stack((grid, calculated))
    mask = calculated != 0.0

    expected_iobs = np.empty_like(intensity)
    for i in range(tth.size):
        profile = pvfcj(
            uvw,
            p,
            xy,
            xy_sf[i],
            shkl,
            eta_mixing,
            tth[i],
            dsp[i],
            hkl[i],
            grid[mask],
            hl,
            sl,
            xn,
            wn,
        )
        expected_iobs[i] = intensity[i] * np.trapezoid(
            observed[mask] * profile / calculated[mask], grid[mask]
        )

    actual_iobs = calc_Iobs_pvfcj(
        uvw,
        p,
        xy,
        xy_sf,
        shkl,
        eta_mixing,
        hl,
        sl,
        xn,
        wn,
        tth,
        dsp,
        hkl,
        grid,
        intensity,
        spectrum_expt,
        spectrum_sim,
    )

    np.testing.assert_allclose(
        actual_iobs, expected_iobs, rtol=1e-12, atol=1e-12
    )
