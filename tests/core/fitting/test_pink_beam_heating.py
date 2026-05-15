import numpy as np
from unittest.mock import patch

import hexrd.core.fitting.peakfunctions as pf
from hexrd.core.fitting.fitpeak import (
    tau0_DFLT,
    tau1_DFLT,
    tau2_DFLT,
    estimate_pk_parms_1d,
    fit_pk_parms_1d,
    estimate_mpk_parms_1d,
    fit_pk_obj_1d,
    fit_pk_obj_1d_bnded,
    calc_pk_integrated_intensities,
)
from hexrd.core.fitting.spectrum import (
    pink_beam_heating as spectrum_pink_beam_heating,
    _initial_guess,
    _build_composite_model,
    SpectrumModel,
    num_func_params,
)


# --------------- core/fitting/peakfunctions.py ---------------


class TestPeakFunctions:
    def test_calc_tau(self):
        tau0, tau1, tau2 = 1.58, -1.35, 0.36
        for x0 in [0.0, 10.0, 30.0, 60.0]:
            result = pf._calc_tau((tau0, tau1, tau2), x0)
            tan_val = np.tan(np.radians(0.5 * x0))
            expected = tau0 + tau1 * tan_val + tau2 * tan_val**2
            assert np.isclose(result, expected, rtol=1e-12)

    def test_gaussian_heating(self):
        p = np.array([1.0, 10.0, 1.5, 0.1])
        x = np.linspace(8, 12, 200)
        g = pf._gaussian_heating(p, x)
        assert g.shape == x.shape
        assert np.all(np.isfinite(g))
        assert abs(x[np.argmax(g)] - 10.0) < 1.0

    def test_gaussian_heating_far_from_center(self):
        p = np.array([1.0, 10.0, 1.5, 0.1])
        g = pf._gaussian_heating(p, np.array([100.0, 200.0, -100.0]))
        assert np.allclose(g, 0.0)

    def test_lorentzian_heating(self):
        p = np.array([1.0, 10.0, 1.5, 0.1])
        x = np.linspace(8, 12, 200)
        l_val = pf._lorentzian_heating(p, x)
        assert l_val.shape == x.shape
        assert np.all(np.isfinite(l_val))

    def test_pink_beam_heating_no_bg(self):
        p = np.array([100.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1])
        x = np.linspace(8, 12, 200)
        result = pf._pink_beam_heating_no_bg(p, x)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0

    def test_amplitude_scaling(self):
        x = np.linspace(8, 12, 200)
        p1 = np.array([100.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1])
        p2 = np.array([200.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1])
        r1 = pf._pink_beam_heating_no_bg(p1, x)
        r2 = pf._pink_beam_heating_no_bg(p2, x)
        np.testing.assert_allclose(r2, 2.0 * r1, rtol=1e-10)

    def test_zero_tau_returns_zeros(self):
        x = np.linspace(8, 12, 200)
        p = np.array([100.0, 10.0, 0.0, 0.0, 0.0, 0.1, 0.1])
        assert np.allclose(pf._pink_beam_heating_no_bg(p, x), 0.0)

    def test_background_added(self):
        x = np.linspace(8, 12, 200)
        p_no_bg = np.array([100.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1])
        bkg_c0, bkg_c1 = 0.5, 0.02
        p_full = np.hstack([p_no_bg, [bkg_c0, bkg_c1]])
        np.testing.assert_allclose(
            pf.pink_beam_heating(p_full, x) - pf._pink_beam_heating_no_bg(p_no_bg, x),
            bkg_c0 + bkg_c1 * x,
            rtol=1e-12,
        )

    def test_lmfit_matches_no_bg(self):
        x = np.linspace(8, 12, 200)
        A, x0 = 100.0, 10.0
        tau0, tau1, tau2 = 1.58, -1.35, 0.36
        fwhm_g, fwhm_l = 0.1, 0.1
        lmfit_result = pf.pink_beam_heating_lmfit(
            x, A, x0, tau0, tau1, tau2, fwhm_g, fwhm_l
        )
        p = np.array([A, x0, tau0, tau1, tau2, fwhm_g, fwhm_l])
        np.testing.assert_allclose(
            lmfit_result, pf._pink_beam_heating_no_bg(p, x), rtol=1e-10
        )

    def test_mpeak_single(self):
        x = np.linspace(8, 12, 200)
        p = np.array([100.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1])
        np.testing.assert_allclose(
            pf._mpeak_1d_no_bg(p, x, 'pink_beam_heating', 1),
            pf._pink_beam_heating_no_bg(p, x),
            rtol=1e-12,
        )

    def test_mpeak_two_peaks(self):
        x = np.linspace(8, 14, 300)
        p1 = np.array([100.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1])
        p2 = np.array([80.0, 12.0, 1.58, -1.35, 0.36, 0.1, 0.1])
        result = pf._mpeak_1d_no_bg(np.hstack([p1, p2]), x, 'pink_beam_heating', 2)
        expected = (
            pf._pink_beam_heating_no_bg(p1, x)
            + pf._pink_beam_heating_no_bg(p2, x)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)


# --------------- core/fitting/fitpeak.py ---------------


class TestFitpeak:
    @patch('hexrd.core.fitting.fitpeak.snip1d')
    @patch('hexrd.core.fitting.fitpeak.optimize.curve_fit')
    def test_estimate_pk_parms(self, mock_curve_fit, mock_snip):
        x = np.linspace(8, 12, 101)
        y = 10.0 * np.exp(-0.5 * ((x - 10.0) / 0.2) ** 2) + 1.0
        mock_snip.return_value = np.ones_like(y)
        mock_curve_fit.return_value = ([0.0, 1.0], None)

        p = estimate_pk_parms_1d(x, y, 'pink_beam_heating')
        assert len(p) == 9
        assert p[2] == tau0_DFLT
        assert p[3] == tau1_DFLT
        assert p[4] == tau2_DFLT

    @patch('hexrd.core.fitting.fitpeak.optimize.least_squares')
    def test_fit_pk_parms(self, mock_lsq):
        x = np.linspace(8, 12, 101)
        mock_lsq.return_value = {'x': np.zeros(9), 'success': True}
        fit_pk_parms_1d(np.zeros(9), x, np.ones_like(x), 'pink_beam_heating')
        assert mock_lsq.called

    def test_fit_pk_obj_residual(self):
        x = np.linspace(8, 12, 101)
        p = np.array([100.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1, 0.5, 0.02])
        f0 = pf.pink_beam_heating(p, x)
        resd = fit_pk_obj_1d(p, x, f0, 'pink_beam_heating')
        np.testing.assert_allclose(resd, 0.0, atol=1e-10)

    @patch('hexrd.core.fitting.fitpeak.pkfuncs')
    def test_fit_pk_obj_bounded(self, mock_pkfuncs):
        x = np.linspace(8, 12, 101)
        f0 = np.ones_like(x)
        mock_pkfuncs.pink_beam_heating.return_value = f0
        res = fit_pk_obj_1d_bnded([10.0], x, f0, 'pink_beam_heating', 1.0, [0.0], [5.0])
        assert res[-1] == 5.0

    @patch('hexrd.core.fitting.fitpeak.snip1d')
    @patch('hexrd.core.fitting.fitpeak.optimize.curve_fit')
    def test_estimate_mpk_parms(self, mock_cf, mock_snip):
        x = np.linspace(8, 14, 201)
        mock_snip.return_value = np.zeros_like(x)
        mock_cf.return_value = ([0, 0], None)
        p0, bnds = estimate_mpk_parms_1d(
            [10.0, 12.0], x, np.ones_like(x), 'pink_beam_heating', bgtype='linear'
        )
        assert len(p0) == 16
        assert len(bnds[0]) == 16

    def test_integrated_intensities(self):
        x = np.linspace(8, 12, 200)
        p = np.array([100.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1])
        ints = calc_pk_integrated_intensities(p, x, 'pink_beam_heating', 1)
        assert len(ints) == 1
        assert ints[0] > 0


# --------------- core/fitting/spectrum.py ---------------


class TestSpectrum:
    def test_wrapper_matches_lmfit(self):
        x = np.linspace(8, 12, 200)
        args = (100.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1)
        np.testing.assert_allclose(
            spectrum_pink_beam_heating(x, *args),
            pf.pink_beam_heating_lmfit(x, *args),
            rtol=1e-12,
        )

    def test_composite_model(self):
        model = _build_composite_model(
            npeaks=2, pktype='pink_beam_heating', bgtype='linear'
        )
        for i in range(2):
            for k in ['amp', 'cen', 'tau0', 'tau1', 'tau2', 'fwhm_g', 'fwhm_l']:
                assert f'pk{i}_{k}' in model.param_names
        assert 'c0' in model.param_names
        assert 'c1' in model.param_names

        params = model.make_params()
        for i in range(2):
            params[f'pk{i}_amp'].set(value=100.0)
            params[f'pk{i}_cen'].set(value=10.0 + i)
            params[f'pk{i}_tau0'].set(value=1.58)
            params[f'pk{i}_tau1'].set(value=-1.35)
            params[f'pk{i}_tau2'].set(value=0.36)
            params[f'pk{i}_fwhm_g'].set(value=0.1)
            params[f'pk{i}_fwhm_l'].set(value=0.1)
        params['c0'].set(value=0.0)
        params['c1'].set(value=0.0)

        y = model.eval(params=params, x=np.linspace(8, 13, 200))
        assert isinstance(y, np.ndarray)
        assert np.all(np.isfinite(y))

    def test_initial_guess_shape(self):
        x = np.linspace(8, 14, 301)
        centers = np.array([10.0, 12.0])
        y = (
            pf.pink_beam_heating_lmfit(x, 5.0, 10.0, 1.58, -1.35, 0.36, 0.1, 0.1)
            + pf.pink_beam_heating_lmfit(x, 3.0, 12.0, 1.58, -1.35, 0.36, 0.1, 0.1)
            + 0.2
            + 0.01 * x
        )
        p0 = _initial_guess(
            centers, x, y,
            pktype='pink_beam_heating', bgtype='linear', fwhm_guess=0.1,
        )
        n_pk = num_func_params['pink_beam_heating']
        n_bg = num_func_params['linear']
        assert p0.size == len(centers) * n_pk + n_bg

    def test_spectrum_model_fit_recovers_center(self):
        x = np.linspace(8, 12, 301)
        amp, cen = 10.0, 10.0
        tau0, tau1, tau2 = 1.58, -1.35, 0.36
        fwhm_g = fwhm_l = 0.1
        y = (
            pf.pink_beam_heating_lmfit(
                x, amp, cen, tau0, tau1, tau2, fwhm_g, fwhm_l
            )
            + 0.01
        )
        data = np.vstack([x, y]).T
        sm = SpectrumModel(
            data, [cen],
            pktype='pink_beam_heating', bgtype='linear',
            fwhm_init=0.1, min_ampl=1e-8,
        )
        assert sm.pktype == 'pink_beam_heating'
        assert any('tau' in n for n in sm.peak_params.keys())
        res = sm.fit()
        assert hasattr(res, 'params') and hasattr(res, 'success')
        if res.success:
            assert abs(res.params['pk0_cen'].value - cen) < 0.5

    def test_tau_constraints_independent(self):
        x = np.linspace(8, 14, 301)
        cen1, cen2 = 10.0, 12.0
        tau0, tau1, tau2 = 1.58, -1.35, 0.36
        y = (
            pf.pink_beam_heating_lmfit(x, 5.0, cen1, tau0, tau1, tau2, 0.1, 0.1)
            + pf.pink_beam_heating_lmfit(x, 3.0, cen2, tau0, tau1, tau2, 0.1, 0.1)
            + 0.01
        )
        data = np.vstack([x, y]).T
        sm = SpectrumModel(
            data, [cen1, cen2],
            pktype='pink_beam_heating', bgtype='linear',
            fwhm_init=0.1, min_ampl=1e-8,
        )
        res = sm.fit()
        if res.success:
            t0 = res.params['pk0_tau0'].value
            t1 = res.params['pk0_tau1'].value
            t2 = res.params['pk0_tau2'].value
            assert not (np.isclose(t0, t1) and np.isclose(t1, t2))
            assert np.isclose(res.params['pk1_tau0'].value, t0, rtol=1e-10)
            assert np.isclose(res.params['pk1_tau1'].value, t1, rtol=1e-10)
            assert np.isclose(res.params['pk1_tau2'].value, t2, rtol=1e-10)


# --------------- core/fitting/utils.py ---------------


class TestFitRing:
    def test_successful_fit(self):
        from hexrd.core.fitting.utils import fit_ring

        x = np.linspace(8, 12, 301)
        cen = 10.0
        tau0, tau1, tau2 = 1.58, -1.35, 0.36
        y = (
            pf.pink_beam_heating_lmfit(x, 10.0, cen, tau0, tau1, tau2, 0.1, 0.1)
            + 0.01
        )
        spectrum_kwargs = {
            'pktype': 'pink_beam_heating',
            'bgtype': 'linear',
            'fwhm_init': 0.1,
            'min_ampl': 1e-8,
        }
        result = fit_ring(x, y, np.array([cen]), spectrum_kwargs, 0.001, 1.0)
        if result is not None:
            assert len(result) == 1
            assert abs(result[0] - cen) < 0.5

    def test_nan_fit_does_not_crash(self):
        from hexrd.core.fitting.utils import fit_ring

        x = np.linspace(0, 1, 50)
        y = np.zeros_like(x)
        spectrum_kwargs = {
            'pktype': 'pink_beam_heating',
            'bgtype': 'linear',
            'fwhm_init': 0.1,
        }
        result = fit_ring(x, y, np.array([0.5]), spectrum_kwargs, 0.001, 1.0)
        assert result is None
