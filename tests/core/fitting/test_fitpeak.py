import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from hexrd.core.fitting import fitpeak


# --- Fixtures ---


@pytest.fixture
def x_1d():
    """Standard 1D x-axis."""
    return np.linspace(0, 10, 101)


@pytest.fixture
def y_gauss_1d(x_1d):
    """Simple 1D Gaussian data."""
    return 10.0 * np.exp(-0.5 * ((x_1d - 5.0) / 1.0) ** 2) + 1.0


@pytest.fixture
def xy_2d():
    """2D grid."""
    x = np.arange(0, 10, 1)
    y = np.arange(0, 10, 1)
    X, Y = np.meshgrid(x, y)
    return X, Y


@pytest.fixture
def z_gauss_2d(xy_2d):
    """2D Gaussian data."""
    X, Y = xy_2d
    Z = 10.0 * np.exp(-((X - 5) ** 2 + (Y - 5) ** 2) / 4.0) + 1.0
    return Z


# --- Helper Function Tests ---


def test_helper_functions():
    x = np.array([1, 2, 3])

    assert np.allclose(fitpeak.cnst_fit_obj(x, 5), [5, 5, 5])
    assert np.allclose(fitpeak.cnst_fit_jac(x, 5), [[1], [1], [1]])

    assert np.allclose(fitpeak.lin_fit_obj(x, 2, 1), [3, 5, 7])
    jac_lin = fitpeak.lin_fit_jac(x, 2, 1)
    assert np.allclose(jac_lin, [[1, 1], [2, 1], [3, 1]])

    assert np.allclose(fitpeak.quad_fit_obj(x, 1, 0, 0), [1, 4, 9])

    res = fitpeak.quad_fit_jac(x, 1, 0, 0)
    assert np.allclose(res, [1, 4, 9])

    y = np.array([0, 5, 10, 5, 0])
    x_grid = np.array([0, 1, 2, 3, 4])
    amp = fitpeak._amplitude_guess(x_grid, 2, y, 2.0)
    assert amp == 10


# --- 1-D Peak Estimation Tests ---


@patch('hexrd.core.fitting.fitpeak.snip1d')
@patch('hexrd.core.fitting.fitpeak.optimize.curve_fit')
def test_estimate_pk_parms_1d_logic(
    mock_curve_fit, mock_snip, x_1d, y_gauss_1d
):
    mock_snip.return_value = np.ones_like(y_gauss_1d)
    mock_curve_fit.return_value = ([0.0, 1.0], None)

    fitpeak.estimate_pk_parms_1d(x_1d, y_gauss_1d, 'gaussian')
    fitpeak.estimate_pk_parms_1d(x_1d, y_gauss_1d, 'pvoigt')
    fitpeak.estimate_pk_parms_1d(x_1d, y_gauss_1d, 'split_pvoigt')
    fitpeak.estimate_pk_parms_1d(x_1d, y_gauss_1d, 'pink_beam_dcs')

    with pytest.raises(RuntimeError):
        fitpeak.estimate_pk_parms_1d(x_1d, y_gauss_1d, 'invalid_type')


@patch('hexrd.core.fitting.fitpeak.snip1d')
@patch('hexrd.core.fitting.fitpeak.optimize.curve_fit')
def test_estimate_pk_parms_1d_edge_cases(mock_curve_fit, mock_snip):
    x = np.linspace(0, 10, 10)
    mock_snip.return_value = np.zeros(10)
    mock_curve_fit.return_value = ([0, 0], None)

    y_start = np.zeros(10)
    y_start[0] = 10
    fitpeak.estimate_pk_parms_1d(x, y_start, 'gaussian')

    y_end = np.zeros(10)
    y_end[-1] = 10
    fitpeak.estimate_pk_parms_1d(x, y_end, 'gaussian')

    y_flat = np.ones(10) * 10
    p = fitpeak.estimate_pk_parms_1d(x, y_flat, 'gaussian')
    assert np.isclose(p[2], 2.5)


# --- 1-D Peak Fitting Tests ---


@patch('hexrd.core.fitting.fitpeak.optimize.leastsq')
def test_fit_pk_parms_1d_leastsq(mock_leastsq, x_1d, y_gauss_1d):
    """Test fitters that use leastsq (unbounded)."""
    p0 = [10, 5, 1, 0, 0]
    mock_leastsq.return_value = (np.array(p0), 1)

    for pt in ['gaussian', 'lorentzian', 'tanh_stepdown']:
        fitpeak.fit_pk_parms_1d(p0, x_1d, y_gauss_1d, pt)
        assert mock_leastsq.call_args[0][0] == fitpeak.fit_pk_obj_1d


@patch('hexrd.core.fitting.fitpeak.optimize.leastsq')
def test_fit_pk_parms_1d_bounded(mock_leastsq, x_1d, y_gauss_1d):
    """Test fitters that use leastsq with manual bounds."""
    p0 = [10, 5, 1, 0.5, 0, 0]
    mock_leastsq.return_value = (np.array(p0), 1)

    for pt in ['pvoigt', 'split_pvoigt']:
        fitpeak.fit_pk_parms_1d(p0, x_1d, y_gauss_1d, pt)
        assert mock_leastsq.call_args[0][0] == fitpeak.fit_pk_obj_1d_bnded


@patch('hexrd.core.fitting.fitpeak.optimize.least_squares')
def test_fit_pk_parms_1d_least_squares(mock_lsq, x_1d, y_gauss_1d):
    p0 = np.zeros(11)
    mock_lsq.return_value = {'x': p0, 'success': True}
    fitpeak.fit_pk_parms_1d(p0, x_1d, y_gauss_1d, 'dcs_pinkbeam')
    assert mock_lsq.called


def test_fit_pk_parms_1d_invalid_and_nan():
    res = fitpeak.fit_pk_parms_1d([1, 2], [0], [0], 'invalid')
    assert res == [1, 2]

    with patch('hexrd.core.fitting.fitpeak.optimize.leastsq') as m_lsq:
        m_lsq.return_value = (np.array([np.nan, np.nan]), 1)
        res = fitpeak.fit_pk_parms_1d([1, 2], [0], [0], 'gaussian')
        assert res == [1, 2]


# --- 1-D Objective and Deriv Functions (Full Coverage) ---


@pytest.mark.parametrize(
    "pktype",
    [
        'gaussian',
        'lorentzian',
        'pvoigt',
        'split_pvoigt',
        'tanh_stepdown',
        'dcs_pinkbeam',
    ],
)
@patch('hexrd.core.fitting.fitpeak.pkfuncs')
def test_fit_pk_obj_1d_coverage(mock_pkfuncs, pktype, x_1d):
    """Cover all branches in fit_pk_obj_1d."""
    f0 = np.ones_like(x_1d)
    p = [1, 2, 3]

    func_map = {
        'gaussian': 'gaussian1d',
        'lorentzian': 'lorentzian1d',
        'pvoigt': 'pvoigt1d',
        'split_pvoigt': 'split_pvoigt1d',
        'tanh_stepdown': 'tanh_stepdown_nobg',
        'dcs_pinkbeam': 'pink_beam_dcs',
    }

    target_func = getattr(mock_pkfuncs, func_map[pktype])
    target_func.return_value = f0 + 1.0

    res = fitpeak.fit_pk_obj_1d(p, x_1d, f0, pktype)

    if pktype == 'dcs_pinkbeam':
        assert np.allclose(res, 1.0)

        f0_nan = np.ones_like(x_1d) * -1
        target_func.return_value = f0_nan + 1
        res_nan = fitpeak.fit_pk_obj_1d(p, x_1d, f0_nan, pktype)
        assert np.allclose(res_nan, 0.0)
    else:
        assert np.allclose(res, 1.0)


@pytest.mark.parametrize(
    "pktype",
    ['gaussian', 'lorentzian', 'pvoigt', 'split_pvoigt', 'dcs_pinkbeam'],
)
@patch('hexrd.core.fitting.fitpeak.pkfuncs')
def test_fit_pk_obj_1d_bnded_coverage(mock_pkfuncs, pktype, x_1d):
    """Cover all branches in fit_pk_obj_1d_bnded."""
    f0 = np.ones_like(x_1d)
    p = [10.0]
    weight = 1.0
    lb, ub = [0.0], [5.0]

    func_map = {
        'gaussian': 'gaussian1d',
        'lorentzian': 'lorentzian1d',
        'pvoigt': 'pvoigt1d',
        'split_pvoigt': 'split_pvoigt1d',
        'dcs_pinkbeam': 'pink_beam_dcs',
    }

    target_func = getattr(mock_pkfuncs, func_map[pktype])
    target_func.return_value = f0

    res = fitpeak.fit_pk_obj_1d_bnded(p, x_1d, f0, pktype, weight, lb, ub)

    assert res[-1] == 5.0

    lb_none = [None]
    res_none = fitpeak.fit_pk_obj_1d_bnded(
        p, x_1d, f0, pktype, weight, lb_none, ub
    )
    assert res_none[-1] == 0.0


@patch('hexrd.core.fitting.fitpeak.pkfuncs')
def test_eval_pk_deriv_1d(mock_pkfuncs, x_1d):
    p = [1, 2, 3]
    mock_pkfuncs.gaussian1d_deriv.return_value = np.zeros((3, len(x_1d)))
    mock_pkfuncs.lorentzian1d_deriv.return_value = np.zeros((3, len(x_1d)))

    fitpeak.eval_pk_deriv_1d(p, x_1d, None, 'gaussian')
    fitpeak.eval_pk_deriv_1d(p, x_1d, None, 'lorentzian')


@patch('hexrd.core.fitting.fitpeak.pkfuncs')
def test_fit_mpk_obj_1d_direct(mock_pkfuncs, x_1d):
    """Directly test multi-peak objective."""
    f0 = np.zeros_like(x_1d)
    mock_pkfuncs.mpeak_1d.return_value = f0 + 5.0

    res = fitpeak.fit_mpk_obj_1d([1, 2], x_1d, f0, 'gaussian', 1, 'linear')
    assert np.allclose(res, 5.0)


# --- Multi-Peak Tests ---


@patch('hexrd.core.fitting.fitpeak.optimize.least_squares')
def test_fit_mpk_parms_1d(mock_lsq, x_1d):
    mock_lsq.return_value = MagicMock(x=np.array([1, 2]))
    fitpeak.fit_mpk_parms_1d([1, 2], x_1d, np.zeros_like(x_1d), 'gaussian', 1)
    fitpeak.fit_mpk_parms_1d(
        [1, 2], x_1d, np.zeros_like(x_1d), 'gaussian', 1, bnds=([0], [1])
    )


@patch('hexrd.core.fitting.fitpeak.snip1d')
@patch('hexrd.core.fitting.fitpeak.optimize.curve_fit')
def test_estimate_mpk_parms_1d(mock_cf, mock_snip, x_1d, y_gauss_1d):
    mock_snip.return_value = np.zeros_like(y_gauss_1d)
    mock_cf.return_value = ([0, 0], None)
    pk_pos = [5.0]

    p0, _ = fitpeak.estimate_mpk_parms_1d(
        pk_pos, x_1d, y_gauss_1d, 'gaussian', bgtype='linear'
    )
    assert len(p0) == 5

    p0, _ = fitpeak.estimate_mpk_parms_1d(
        pk_pos, x_1d, y_gauss_1d, 'pvoigt', bgtype='constant'
    )
    assert len(p0) == 5

    p0, _ = fitpeak.estimate_mpk_parms_1d(
        pk_pos, x_1d, y_gauss_1d, 'split_pvoigt', bgtype='quadratic'
    )
    assert len(p0) == 9

    p0, _ = fitpeak.estimate_mpk_parms_1d(
        pk_pos, x_1d, y_gauss_1d, 'pink_beam_dcs', bgtype='cubic'
    )
    assert len(p0) > 4


# --- 2-D Peak Fitting Tests ---


def test_estimate_pk_parms_2d(xy_2d, z_gauss_2d):
    X, Y = xy_2d
    fitpeak.estimate_pk_parms_2d(X, Y, z_gauss_2d, 'gaussian')
    fitpeak.estimate_pk_parms_2d(X, Y, z_gauss_2d, 'gaussian_rot')
    fitpeak.estimate_pk_parms_2d(X, Y, z_gauss_2d, 'split_pvoigt_rot')


@pytest.mark.parametrize(
    "pktype", ['gaussian', 'gaussian_rot', 'split_pvoigt_rot']
)
@patch('hexrd.core.fitting.fitpeak.optimize.leastsq')
def test_fit_pk_parms_2d_coverage(mock_leastsq, pktype, xy_2d, z_gauss_2d):
    """Cover all types in fit_pk_parms_2d."""
    X, Y = xy_2d
    p0 = np.zeros(8)
    mock_leastsq.return_value = (p0, 1)

    fitpeak.fit_pk_parms_2d(p0, X, Y, z_gauss_2d, pktype)
    assert mock_leastsq.called


def test_fit_pk_parms_2d_nan_check(xy_2d, z_gauss_2d):
    with patch('hexrd.core.fitting.fitpeak.optimize.leastsq') as m_lsq:
        m_lsq.return_value = (np.array([np.nan]), 1)
        res = fitpeak.fit_pk_parms_2d(
            [1], xy_2d[0], xy_2d[1], z_gauss_2d, 'gaussian'
        )
        assert res == [1]


@pytest.mark.parametrize(
    "pktype", ['gaussian', 'gaussian_rot', 'split_pvoigt_rot']
)
@patch('hexrd.core.fitting.fitpeak.pkfuncs')
def test_fit_pk_obj_2d_coverage(mock_pk, pktype, xy_2d):
    """Cover all types in fit_pk_obj_2d."""
    X, Y = xy_2d
    f0 = np.zeros_like(X)
    p = [1]

    func_map = {
        'gaussian': 'gaussian2d',
        'gaussian_rot': 'gaussian2d_rot',
        'split_pvoigt_rot': 'split_pvoigt2d_rot',
    }
    getattr(mock_pk, func_map[pktype]).return_value = f0 + 1.0

    res = fitpeak.fit_pk_obj_2d(p, X, Y, f0, pktype)
    assert np.all(res == 1.0)


# --- Utilities Tests ---


def test_goodness_of_fit():
    f = np.array([10, 10, 10])
    f0 = np.array([10, 9, 11])
    fitpeak.goodness_of_fit(f, f0)


@patch('hexrd.core.fitting.fitpeak.plt')
def test_direct_pk_analysis(mock_plt, x_1d, y_gauss_1d):
    fitpeak.direct_pk_analysis(x_1d, y_gauss_1d, remove_bg=True)
    fitpeak.direct_pk_analysis(x_1d, y_gauss_1d * 0.0001, low_int=100.0)
    y_edge = np.zeros_like(x_1d)
    y_edge[-1] = 100.0
    fitpeak.direct_pk_analysis(x_1d, y_edge, remove_bg=True)


@patch('hexrd.core.fitting.fitpeak.pkfuncs')
def test_calc_pk_integrated_intensities(mock_pk, x_1d):
    mock_pk._gaussian1d_no_bg.return_value = np.ones_like(x_1d)
    mock_pk._lorentzian1d_no_bg.return_value = np.ones_like(x_1d)
    mock_pk._pvoigt1d_no_bg.return_value = np.ones_like(x_1d)
    mock_pk._split_pvoigt1d_no_bg.return_value = np.ones_like(x_1d)

    p = np.zeros(100)
    for t in ['gaussian', 'lorentzian', 'pvoigt', 'split_pvoigt']:
        fitpeak.calc_pk_integrated_intensities(p, x_1d, t, 1)
