import numpy as np
import lmfit
import pytest

from hexrd.powder.wppf import peakfunctions as wpf
from hexrd.powder.wppf.wppfsupport import (
    _add_pvheating_parameters,
    _generate_default_parameters_LeBail,
)
from hexrd.powder.wppf.WPPF import peakshape_dict


# --------------- wppf/peakfunctions.py ---------------


class TestWppfPeakFunctions:
    def test_calc_tau_matches_core(self):
        from hexrd.core.fitting.peakfunctions import _calc_tau as core_calc_tau

        tau_vals = (1.58, -1.35, 0.36)
        tau_arr = np.array(tau_vals)
        for tth in [0.0, 5.0, 15.0, 45.0, 90.0]:
            result = wpf._calc_tau(tau_arr, tth)
            tan_val = np.tan(np.radians(0.5 * tth))
            expected = tau_vals[0] + tau_vals[1] * tan_val + tau_vals[2] * tan_val**2
            assert np.isclose(result, expected, rtol=1e-12)
            assert np.isclose(result, core_calc_tau(tau_vals, tth), rtol=1e-12)

    def test_gaussian_heating_matches_core(self):
        from hexrd.core.fitting.peakfunctions import (
            _gaussian_heating as core_gh,
        )

        tth_list = np.linspace(8, 12, 200)
        tau, fwhm_g, tth = 1.5, 0.1, 10.0
        wppf_result = wpf._gaussian_heating(tau, fwhm_g, tth, tth_list)
        assert wppf_result.shape == tth_list.shape
        assert np.all(np.isfinite(wppf_result))
        assert abs(tth_list[np.argmax(wppf_result)] - tth) < 1.0

        p = np.array([1.0, tth, tau, fwhm_g])
        np.testing.assert_allclose(wppf_result, core_gh(p, tth_list), rtol=1e-10)

    def test_gaussian_heating_far_from_center(self):
        g = wpf._gaussian_heating(1.5, 0.1, 10.0, np.array([100.0, 200.0, -100.0]))
        assert np.allclose(g, 0.0)

    def test_lorentzian_heating_matches_core(self):
        from hexrd.core.fitting.peakfunctions import (
            _lorentzian_heating as core_lh,
        )

        tth_list = np.linspace(8, 12, 200)
        tau, fwhm_l, tth = 1.5, 0.1, 10.0
        wppf_result = wpf._lorentzian_heating(tau, fwhm_l, tth, tth_list)
        assert wppf_result.shape == tth_list.shape
        assert np.all(np.isfinite(wppf_result))

        p = np.array([1.0, tth, tau, fwhm_l])
        np.testing.assert_allclose(wppf_result, core_lh(p, tth_list), rtol=1e-10)


class TestPvoightAndSpectrum:
    @pytest.fixture
    def standard_params(self):
        tau = np.array([1.58, -1.35, 0.36])
        uvw = np.array([10.0, -2.0, 0.5])
        p = np.float64(0.0)
        xy = np.array([0.5, 1.9])
        xy_sf = np.float64(0.0)
        shkl = np.zeros(15)
        eta_mixing = np.float64(0.5)
        tth = np.float64(20.0)
        dsp = np.float64(2.0)
        hkl = np.array([1.0, 1.0, 1.0])
        tth_list = np.linspace(18, 22, 200)
        return tau, uvw, p, xy, xy_sf, shkl, eta_mixing, tth, dsp, hkl, tth_list

    def test_pvoight_shape_and_normalized(self, standard_params):
        result = wpf.pvoight_heating(*standard_params)
        tth_list = standard_params[-1]
        assert result.shape == tth_list.shape
        assert np.all(np.isfinite(result))
        area = np.trapezoid(result, tth_list)
        assert np.isclose(area, 1.0, atol=0.05)

    def test_pvoight_peak_near_tth(self, standard_params):
        result = wpf.pvoight_heating(*standard_params)
        tth_list = standard_params[-1]
        tth = standard_params[7]
        peak_x = tth_list[np.argmax(np.abs(result))]
        assert abs(peak_x - tth) < 2.0

    def test_computespectrum_single_and_multiple(self):
        tau = np.array([1.58, -1.35, 0.36])
        uvw = np.array([10.0, -2.0, 0.5])
        p = np.float64(0.0)
        xy = np.array([0.5, 1.9])
        shkl = np.zeros(15)
        eta_mixing = np.float64(0.5)

        # Single reflection
        tth_list = np.linspace(15, 25, 300)
        spec = wpf.computespectrum_pvheating(
            tau, uvw, p, xy, np.array([0.0]), shkl, eta_mixing,
            np.array([20.0]), np.array([2.0]),
            np.array([[1.0, 1.0, 1.0]]), tth_list, np.array([100.0]),
        )
        assert spec.shape == tth_list.shape
        assert np.all(np.isfinite(spec))
        assert np.max(spec) > 0

        # Multiple reflections
        tth_list2 = np.linspace(15, 35, 500)
        spec2 = wpf.computespectrum_pvheating(
            tau, uvw, p, xy, np.array([0.0, 0.0]), shkl, eta_mixing,
            np.array([20.0, 30.0]), np.array([2.0, 1.5]),
            np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]),
            tth_list2, np.array([100.0, 50.0]),
        )
        assert spec2.shape == tth_list2.shape
        assert np.all(np.isfinite(spec2))

    def test_computespectrum_zero_intensity(self):
        tau = np.array([1.58, -1.35, 0.36])
        uvw = np.array([10.0, -2.0, 0.5])
        tth_list = np.linspace(15, 25, 300)
        spec = wpf.computespectrum_pvheating(
            tau, uvw, np.float64(0.0), np.array([0.5, 1.9]),
            np.array([0.0]), np.zeros(15), np.float64(0.5),
            np.array([20.0]), np.array([2.0]),
            np.array([[1.0, 1.0, 1.0]]), tth_list, np.array([0.0]),
        )
        assert np.allclose(spec, 0.0)

    def test_calc_iobs(self):
        tau = np.array([1.58, -1.35, 0.36])
        uvw = np.array([10.0, -2.0, 0.5])
        p = np.float64(0.0)
        xy = np.array([0.5, 1.9])
        xy_sf = np.array([0.0])
        shkl = np.zeros(15)
        eta_mixing = np.float64(0.5)
        tth = np.array([20.0])
        dsp = np.array([2.0])
        hkl = np.array([[1.0, 1.0, 1.0]])
        tth_list = np.linspace(15, 25, 300)
        Icalc = np.array([100.0])

        spec_sim_y = wpf.computespectrum_pvheating(
            tau, uvw, p, xy, xy_sf, shkl, eta_mixing,
            tth, dsp, hkl, tth_list, Icalc,
        ) + 0.01
        spectrum_expt = np.column_stack([tth_list, spec_sim_y])
        spectrum_sim = np.column_stack([tth_list, spec_sim_y])

        Iobs = wpf.calc_Iobs_pvheating(
            tau, uvw, p, xy, xy_sf, shkl, eta_mixing,
            tth, dsp, hkl, tth_list, Icalc,
            spectrum_expt, spectrum_sim,
        )
        assert Iobs.shape == tth.shape
        assert np.all(np.isfinite(Iobs))
        assert Iobs[0] > 0


# --------------- wppf/wppfsupport.py ---------------


class TestWppfSupport:
    def test_pvheating_parameters(self):
        params = lmfit.Parameters()
        _add_pvheating_parameters(params)

        assert params['tau0'].value == 1.58
        assert params['tau1'].value == -1.35
        assert params['tau2'].value == 0.36

        assert params['tau0'].min == -2.0
        assert params['tau0'].max == 2.0
        assert params['tau2'].min == -1.0
        assert params['tau2'].max == 1.0

        for name in ['tau0', 'tau1', 'tau2']:
            assert not params[name].vary

    def test_default_lebail_params_include_tau(self):
        from hexrd.core.material import Material

        mat = Material()
        params = _generate_default_parameters_LeBail(
            mat, peakshape=3, bkgmethod={'chebyshev': 3}
        )
        for name in ['tau0', 'tau1', 'tau2', 'U', 'V', 'W']:
            assert name in params


# --------------- wppf/WPPF.py dispatch ---------------


class TestPeakshapeDict:
    def test_pvheating_in_dict(self):
        assert 'pvheating' in peakshape_dict
