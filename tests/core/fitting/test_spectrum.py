# tests/test_spectrum_full_coverage.py
import numpy as np
import pytest

import hexrd.core.fitting.spectrum as s
from hexrd.core.fitting.spectrum import (
    gaussian_1d,
    linear_bkg,
    quadratic_bkg,
    cubic_bkg,
    quartic_bkg,
    quintic_bkg,
    chebyshev_bkg,
    _amplitude_guess,
    _initial_guess,
    _build_composite_model,
    SpectrumModel,
    pink_beam_dcs,
    num_func_params,
)


Cheb = np.polynomial.chebyshev.Chebyshev


# ---- backgrounds: verify Chebyshev-based implementation and varargs ----
@pytest.mark.parametrize(
    "fn, coeffs",
    [
        (linear_bkg, (0.7, -0.25)),
        (quadratic_bkg, (0.7, -0.25, 0.1)),
        (cubic_bkg, (0.7, -0.25, 0.1, -0.05)),
    ],
)
def test_small_backgrounds_match_chebyshev(fn, coeffs):
    x = np.linspace(-1.0, 1.0, 7)
    ref = Cheb(coeffs, domain=(min(x), max(x)))(x)
    assert np.allclose(fn(x, *coeffs), ref)


def test_quartic_quintic_and_varargs():
    x = np.linspace(2.0, 5.0, 13)
    c0, c1, c2, c3, c4 = 0.5, -0.2, 0.07, -0.01, 0.003
    c5 = -0.0005
    assert np.allclose(quartic_bkg(x, c0, c1, c2, c3, c4), Cheb((c0, c1, c2, c3, c4), domain=(min(x), max(x)))(x))
    assert np.allclose(quintic_bkg(x, c0, c1, c2, c3, c4, c5), Cheb((c0, c1, c2, c3, c4, c5), domain=(min(x), max(x)))(x))

    # quintic with zero last coefficient equals quartic
    x2 = np.linspace(-3.0, 1.0, 17)
    coeffs = (1.1, -0.3, 0.05, 0.0, 0.001)
    assert np.allclose(quartic_bkg(x2, *coeffs), quintic_bkg(x2, *coeffs, 0.0))

    # chebyshev_bkg varargs
    x3 = np.linspace(0.0, 2.0, 9)
    vals = (0.2, -0.1, 0.03, -0.005, 0.0007, -9e-05)
    assert np.allclose(chebyshev_bkg(x3, *vals[:2]), linear_bkg(x3, *vals[:2]))
    assert np.allclose(chebyshev_bkg(x3, *vals[:3]), quadratic_bkg(x3, *vals[:3]))
    assert np.allclose(chebyshev_bkg(x3, *vals), Cheb(vals, domain=(min(x3), max(x3)))(x3))


# ---- amplitude guess & initial_guess ----
def test_amplitude_guess_centered_peak():
    x = np.linspace(0.0, 4.0, 41)
    amp, cen, fwhm = 5.0, 2.0, 0.2
    y = gaussian_1d(x, amp, cen, fwhm)
    assert _amplitude_guess(x, cen, y, fwhm) == pytest.approx(np.max(y), rel=1e-6)


def test_initial_guess_pk_and_bg_slicing_and_constant_bg():
    x = np.linspace(10.0, 12.0, 101)
    centers = np.array([10.5, 11.4])
    amp1, amp2 = 4.0, 2.0
    fwhm_guess = 0.05
    y = gaussian_1d(x, amp1, centers[0], fwhm_guess) + gaussian_1d(x, amp2, centers[1], fwhm_guess) + (0.2 + 0.05 * x)

    p0 = _initial_guess(centers, x, y, pktype="pvoigt", bgtype="linear", fwhm_guess=fwhm_guess, min_ampl=0.1)
    n_pk = s.num_func_params["pvoigt"]
    n_bg = s.num_func_params["linear"]
    pk_part = p0[:-n_bg]
    assert pk_part.size == len(centers) * n_pk
    pkparams = pk_part.reshape(len(centers), n_pk)
    assert np.all(pkparams[:, 0] >= 0.1)

    # constant bg branch returns single bg param
    y_const = gaussian_1d(x, amp1, centers[0], fwhm_guess) + gaussian_1d(x, amp2, centers[1], fwhm_guess) + 3.14
    p0_const = _initial_guess(centers, x, y_const, pktype="gaussian", bgtype="constant", fwhm_guess=[fwhm_guess, fwhm_guess])
    assert p0_const.size == (len(centers) * s.num_func_params["gaussian"] + 1)


# ---- composite model builder: parameter names + eval with filled defaults ----
PK_TYPES = ['gaussian', 'lorentzian', 'pvoigt', 'split_pvoigt', 'pink_beam_dcs']
BG_TYPES = ['constant', 'linear', 'quadratic', 'cubic', 'quartic', 'quintic']


def _fill_params_with_defaults(params):
    for name, par in params.items():
        ln = name.lower()
        if ln.endswith('_amp') or ln == 'amp' or ln.endswith('amp'):
            par.set(value=1.0)
        elif ln.endswith('_cen') or ln == 'cen':
            par.set(value=0.0)
        elif 'fwhm' in ln:
            par.set(value=0.1)
        elif 'mixing' in ln:
            par.set(value=0.5)
        elif ln in ('c0', 'c1', 'c2', 'c3', 'c4', 'c5'):
            par.set(value=0.01 if ln != 'c0' else 0.1)
        elif 'alpha' in ln or 'beta' in ln:
            par.set(value=0.1)
        else:
            try:
                par.set(value=0.0)
            except Exception:
                pass
    return params


@pytest.mark.parametrize("pktype", PK_TYPES)
@pytest.mark.parametrize("bgtype", BG_TYPES)
def test_build_composite_model_all_variants_eval(pktype, bgtype):
    model = _build_composite_model(npeaks=2, pktype=pktype, bgtype=bgtype)
    assert model is not None
    # expected prefixed peak params
    for i in range(2):
        for k in s._function_dict_1d[pktype]:
            assert f"pk{i}_{k}" in model.param_names
    for bk in s._function_dict_1d[bgtype]:
        assert bk in model.param_names

    params = _fill_params_with_defaults(model.make_params())
    x = np.linspace(-0.5, 0.5, 31)
    y = model.eval(params=params, x=x)
    assert isinstance(y, np.ndarray) and y.shape == x.shape and np.all(np.isfinite(y))


def test_multi_peak_prefixing_and_output_behavior():
    model = _build_composite_model(npeaks=3, pktype='gaussian', bgtype='constant')
    for i in range(3):
        assert f"pk{i}_amp" in model.param_names
    params = model.make_params()
    for i in range(3):
        params[f"pk{i}_amp"].set(value=1.0 + 0.5 * i)
        params[f"pk{i}_cen"].set(value=-0.2 + 0.2 * i)
        params[f"pk{i}_fwhm"].set(value=0.08 + 0.01 * i)
    params['c0'].set(value=0.0)
    x = np.linspace(-1.0, 1.0, 201)
    y = model.eval(params=params, x=x)
    assert np.all(np.isfinite(y))
    argmax = x[np.argmax(y)]
    centers = [-0.2 + 0.2 * i for i in range(3)]
    assert np.min(np.abs(np.array(centers) - argmax)) < 0.3


# ---- SpectrumModel constructor + fit paths (gaussian, split_pvoigt, pink_beam_dcs) ----
def test_spectrummodel_gaussian_fit_recovers_amp_and_center():
    x = np.linspace(20.0, 25.0, 301)
    amp_true, cen_true, fwhm_true = 7.25, 22.3, 0.12
    c0_true, c1_true = 0.5, -0.02
    y = gaussian_1d(x, amp_true, cen_true, fwhm_true) + linear_bkg(x, c0_true, c1_true)
    data = np.vstack([x, y]).T
    sm = SpectrumModel(data, [cen_true], pktype="gaussian", bgtype="linear", fwhm_init=0.1)
    assert sm.pktype == "gaussian" and sm.bgtype == "linear" and sm.num_peaks == 1
    assert "pk0_amp" in sm.params and "c0" in sm.params and "c1" in sm.params
    res = sm.fit()
    assert res.success
    assert res.params["pk0_amp"].value == pytest.approx(amp_true, rel=0.05, abs=0.2)
    assert res.params["pk0_cen"].value == pytest.approx(cen_true, rel=1e-3, abs=0.05)


def test_spectrummodel_split_pvoigt_constructor_and_fit_runs():
    x = np.linspace(-1.0, 1.0, 201)
    centers = np.array([-0.3, 0.4])
    p1 = s.split_pvoigt_1d(x, 1.0, centers[0], 0.1, 0.2, 0.5, 0.5)
    p2 = s.split_pvoigt_1d(x, 0.6, centers[1], 0.08, 0.16, 0.5, 0.5)
    y = p1 + p2 + 0.1
    data = np.vstack([x, y]).T
    sm = SpectrumModel(data, centers, pktype="split_pvoigt", bgtype="constant", fwhm_init=0.1)
    names = list(sm.params.keys())
    assert any("mixing_l" in n or "mixing_h" in n for n in names)
    res = sm.fit()
    assert hasattr(res, "success")


def test_spectrummodel_pink_beam_dcs_refit_flow_executes():
    x = np.linspace(0.5, 1.5, 201)
    amp, cen = 2.0, 1.0
    a0, a1, b0, b1 = s.alpha0_DFLT, s.alpha1_DFLT, s.beta0_DFLT, s.beta1_DFLT
    fwhm_g = fwhm_l = 0.05
    y = pink_beam_dcs(x, amp, cen, a0, a1, b0, b1, fwhm_g, fwhm_l) + 0.01
    data = np.vstack([x, y]).T
    sm = SpectrumModel(data, [cen], pktype="pink_beam_dcs", bgtype="linear", fwhm_init=0.05, min_ampl=1e-8)
    # constructor should create alpha/beta/fwhm params (possibly with vary=False initially)
    assert any("alpha" in n or "beta" in n for n in sm.peak_params.keys())
    res = sm.fit()
    assert hasattr(res, "params") and hasattr(res, "success")


def test_invalid_pktype_and_bgtype_assertions():
    x = np.linspace(0.0, 1.0, 11)
    y = np.zeros_like(x)
    data = np.vstack([x, y]).T
    with pytest.raises(AssertionError):
        SpectrumModel(data, [0.5], pktype="__no_such_type__")
    with pytest.raises(AssertionError):
        SpectrumModel(data, [0.5], pktype="gaussian", bgtype="__no_such_bg__")

def test_spectrummodel_background_params_and_none_fwhm_guess():
    # synthetic single peak spectrum
    x = np.linspace(0.0, 1.0, 21)
    y = gaussian_1d(x, 2.0, 0.5, 0.1) + s.linear_bkg(x, 0.1, 0.05)
    data = np.vstack([x, y]).T
    centers = [0.5]

    sm = s.SpectrumModel(data, centers, pktype="gaussian", bgtype="linear", fwhm_init=None)

    bparams = sm.background_params
    for bk in s._function_dict_1d['linear']:
        assert bk in bparams

    y_simple = gaussian_1d(x, 1.0, 0.5, 0.1) + 0.2
    p0 = _initial_guess(centers, x, y_simple, pktype="gaussian", bgtype="linear")
    n_pk = num_func_params["gaussian"]
    n_bg = num_func_params["linear"]
    assert p0.size == len(centers) * n_pk + n_bg
