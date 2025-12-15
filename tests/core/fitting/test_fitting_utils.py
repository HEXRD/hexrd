import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from lmfit import Parameters

from hexrd.core.fitting import utils

# --- Fixtures ---


@pytest.fixture
def params():
    p = Parameters()
    p.add_many(
        ('cen_1', 10.0, True),
        ('cen_2', 20.0, True),
        ('fwhm_1', 0.1, True),
        ('mixing_1', 0.5, True),
        ('other', 1.0, True),
    )
    return p


# --- Parameter Munging ---


def test_param_helpers(params):
    assert utils._parameter_arg_constructor({'a': 1}, (2,))[0] == ('a', 1, 2)
    assert len(utils._extract_parameters_by_name(params, 'cen')) == 2
    utils._set_refinement_by_name(params, 'cen', vary=False)
    assert not params['cen_1'].vary
    with pytest.raises(RuntimeWarning):
        utils._set_refinement_by_name(params, 'bad')


def test_equality_constraints(params):
    utils._set_equality_constraints(params, 'cen')
    assert params['cen_2'].expr == 'cen_1'

    utils._set_equality_constraints(params, [('fwhm_1', 'other')])
    assert params['fwhm_1'].expr == 'other'

    with pytest.raises(RuntimeWarning):
        utils._set_equality_constraints(params, 'bad')


def test_bound_constraints(params):
    utils._set_bound_constraints(params, 'cen_1', min_val=0, max_val=100)
    assert params['cen_1'].min == 0

    utils._set_bound_constraints(params, 'cen_2', box=2.0)
    assert params['cen_2'].min == 19.0

    utils._set_bound_constraints(params, 'other', box=50.0, percentage=True)
    assert params['other'].min == 0.75
    assert params['other'].max == 1.25


def test_width_mixing_bounds(params):
    utils._set_width_mixing_bounds(params, min_w=0.05)
    assert params['fwhm_1'].min == 0.05
    assert params['mixing_1'].max == 1.0


@patch('hexrd.core.fitting.utils.uniqueVectors')
def test_peak_center_bounds(mock_uniq, params):
    mock_uniq.return_value = np.array([[10], [20]])
    utils._set_peak_center_bounds(params, [0, 30], min_sep=2.0)
    assert 'pksep0' in params
    assert params['cen_2'].expr == 'cen_1+pksep0'

    mock_uniq.return_value = MagicMock()
    mock_uniq.return_value.squeeze.return_value = [10.0]
    with pytest.raises(RuntimeError):
        utils._set_peak_center_bounds(params, [0, 30], min_sep=1.0)

    p = Parameters()
    p.add('cen_1', 10.0)
    utils._set_peak_center_bounds(p, [0, 30])


# --- Numba Math Utilities () ---


def test_math_funcs():
    assert np.isclose(utils.erfc(np.array([0.0, 100.0]))[0], 1.0)
    assert np.isclose(utils.erfc(np.array([-100.0]))[0], 2.0)

    assert not np.isnan(utils.exp1exp_under1(np.array([0.1 + 0j]))[0])
    assert not np.isnan(utils.exp1exp_over1(np.array([2.0 + 0j]))[0])
    assert not np.any(
        np.isnan(utils.exp1exp(np.array([0.1, 2.0], dtype=np.complex128)))
    )

    assert utils._calc_alpha(np.array([1.0, 2.0]), 0.0) == 1.0
    assert utils._calc_beta(np.array([3.0, 4.0]), 0.0) == 3.0

    assert np.isclose(utils._mixing_factor_pv(1.0, 0.0)[0], 0.0)
    assert np.isclose(utils._mixing_factor_pv(0.0, 1.0)[0], 1.0)
    assert 0 < utils._mixing_factor_pv(1.0, 1.0)[0] < 1


def test_pink_beam_funcs():
    p = np.array([10.0, 5.0, 1.0, 1.0, 0.5])
    x = np.array([5.0, 15.0])

    res_g = utils._gaussian_pink_beam(p, x)
    assert res_g[0] > 1.0 and res_g[1] == 0.0
    assert utils._gaussian_pink_beam(p, np.array([np.nan]))[0] == 0.0

    res_l = utils._lorentzian_pink_beam(p, x)
    assert not np.any(np.isnan(res_l))
    assert utils._lorentzian_pink_beam(p, np.array([np.nan]))[0] == 0.0


# --- Spectrum Fitting Wrapper ---


@patch('hexrd.core.fitting.spectrum.SpectrumModel')
def test_fit_ring(mock_sm_cls):
    data = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 50.0, 10.0]), [2.0])
    mock_res = MagicMock()
    mock_sm_cls.return_value.fit.return_value = mock_res

    mock_res.success = True
    mock_res.best_values = {'pk0_amp': 100.0, 'pk0_cen': 2.0}
    res = utils.fit_ring(*data, {}, 10, 0.1)
    assert np.isclose(res[0], 2.0)

    mock_res.success = False
    assert utils.fit_ring(*data, {}, 10, 0.1) is None

    mock_res.success = True
    mock_res.best_values['pk0_amp'] = 5.0
    assert utils.fit_ring(*data, {}, 10, 0.1) is None

    mock_res.best_values['pk0_amp'] = 100.0
    mock_res.best_values['pk0_cen'] = 2.5
    assert utils.fit_ring(*data, {}, 10, 0.1) is None
