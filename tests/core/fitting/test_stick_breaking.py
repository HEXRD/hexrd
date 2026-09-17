import lmfit
import numpy as np
import pytest

from hexrd.core.fitting import stick_breaking

# A noisy mix of three patterns with fractions 0.6, 0.3 and 0.1
X = np.linspace(0, 1, 200)
PATTERNS = {'A': np.sin(3 * X) ** 2, 'B': X**2, 'C': np.exp(-(((X - 0.5) / 0.1) ** 2))}
MIX = 0.6 * PATTERNS['A'] + 0.3 * PATTERNS['B'] + 0.1 * PATTERNS['C']
DATA = 2 * MIX + 0.5 + np.random.default_rng(0).normal(0, 0.01, X.size)


def _residual(p: lmfit.Parameters) -> np.ndarray:
    model = sum(p[k].value * PATTERNS[k] for k in 'ABC')
    return p['scale'].value * model + p['bkg'].value - DATA


def _fit(params: lmfit.Parameters) -> lmfit.Parameters:
    fit_params = stick_breaking.add_params(params, [['A', 'B', 'C']])
    result = lmfit.Minimizer(_residual, fit_params).least_squares()
    return stick_breaking.strip_params(result.params, params)


def test_stick_breaking() -> None:
    params = lmfit.Parameters()
    params.add('scale', 1.0)
    params.add('bkg', 0.0)
    params.add('A', 1 / 3, min=0, max=1)
    params.add('B', 1 / 3, min=0, max=1)
    params.add('C', expr='1 - A - B', min=0, max=1)

    # With no constraint active, this matches plain lmfit
    fit = _fit(params)
    plain = lmfit.minimize(_residual, params, method='least_squares').params
    assert list(fit) == list(params) and set(fit['scale'].correl) <= set(params)
    for k in params:
        assert fit[k].value == pytest.approx(plain[k].value, rel=1e-4)
        assert fit[k].stderr == pytest.approx(plain[k].stderr, rel=1e-3)

    # A wrongly fixed B stays fixed. The data want A + B > 1, so the fit
    # stops at A + B = 1 instead of letting C go negative. A's narrowed
    # bound and C's stale expression are fixed up first.
    params['A'].set(value=0.6, max=0.6)
    params['B'].set(value=0.5, vary=False)
    params['C'].expr = '1 - A'
    fit = _fit(params)
    assert [fit[k].value for k in 'ABC'] == pytest.approx([0.5, 0.5, 0])
    assert params['A'].max == 1 and params['C'].expr == '1 - A - B'


def test_validate() -> None:
    params = lmfit.Parameters()
    params.add('A', 0.7, min=-1, vary=False)
    params.add('B', 0.6, vary=False)
    params.add('C', expr='1 - A - B')
    groups = [['A', 'B', 'C']]
    with pytest.raises(ValueError, match='sum to more than 1'):
        stick_breaking.validate(params, groups)

    params['A'].value = -0.2
    with pytest.raises(ValueError, match='within'):
        stick_breaking.validate(params, groups)

    params['B'].expr = '1 - A'
    with pytest.raises(ValueError, match='Only one'):
        stick_breaking.validate(params, groups)
