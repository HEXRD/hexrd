import numpy as np

from hexrd.powder.wppf.WPPF import _unique_profiles
from hexrd.powder.wppf.peakfunctions import (
    calc_Iobs_pvtch,
    computespectrum_pvtch,
    pvoight_wppf,
)


def test_unique_profiles_match_individual_reflections():
    # (100) and (010) have identical profiles; (110) does not
    uvw = np.array([81.5, 1.0337, 5.18275])
    xy = np.array([0.5665, 1.90994])
    shkl = np.zeros(15)
    tth = np.array([30.0, 30.0, 45.0])
    dsp = np.array([0.6, 0.6, 0.4])
    hkls = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    xs = np.zeros(3)
    Ic = np.array([2.0, 3.0, 4.0])
    grid = np.linspace(20.0, 55.0, 351)

    ut, ud, uh, ux, uI, inv = _unique_profiles(tth, dsp, hkls, xs, shkl, Ic)
    assert len(ut) == 2
    np.testing.assert_array_equal(ut[inv], tth)

    profiles = [
        pvoight_wppf(uvw, 0.0, xy, xs[i], shkl, 0.5, tth[i], dsp[i], hkls[i], grid)
        for i in range(3)
    ]
    unique_args = (uvw, 0.0, xy, ux, shkl, 0.5, ut, ud, uh, grid)
    spec = computespectrum_pvtch(*unique_args, uI)
    np.testing.assert_allclose(spec, np.dot(Ic, profiles), rtol=1e-12)

    # zeros in the simulated spectrum are masked out of the Iobs integral
    observed = 10.0 + np.sin(grid) ** 2
    calculated = 9.0 + np.cos(grid) ** 2
    calculated[::17] = 0.0
    mask = calculated != 0.0
    expected = [
        Ic[i]
        * np.trapezoid(
            observed[mask] * profiles[i][mask] / calculated[mask], grid[mask]
        )
        for i in range(3)
    ]
    expt = np.column_stack((grid, observed))
    sim = np.column_stack((grid, calculated))
    scale = calc_Iobs_pvtch(*unique_args, np.ones(2), expt, sim)
    np.testing.assert_allclose(Ic * scale[inv], expected, rtol=1e-12)
