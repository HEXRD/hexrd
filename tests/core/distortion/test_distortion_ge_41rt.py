import numpy as np
import pytest

import hexrd.core.distortion as distortion
from hexrd.core.distortion.ge_41rt import RHO_MAX, inverse_distortion_numpy

# Canonical GE exponents are (2, 4, 2). These are the ge3 parameters from
# examples/state_examples/GE_WPPF/ge_wppf.h5
GE_WPPF_PARAMS = [
    -5.24127372e-07,
    -7.14809975e-05,
    -5.18806621e-04,
    2.0,
    4.0,
    2.0,
]

# Same magnitudes, but positive, so B(eta) > 0 (pincushion). The closed-form
# inverse takes sqrt(-B) here and returns NaN.
PINCUSHION_PARAMS = [5.24e-07, 7.15e-05, 5.19e-04, 2.0, 2.0, 2.0]


def _panel_points(n=64):
    """Points spanning a 2048x2048 GE panel, in mm.

    The corners reach rho ~ 290 mm, well past RHO_MAX (204.8), which is where
    the closed-form inverse's exponent assumption diverges the most.
    """
    c = np.linspace(-204.8, 204.8, n)
    x, y = np.meshgrid(c, c)
    return np.column_stack((x.ravel(), y.ravel()))


def test_newton_is_registered_and_shares_the_forward_map():
    assert 'GE_41RT_newton' in distortion.maptypes()

    xy = _panel_points()
    old = distortion.get_mapping('GE_41RT', GE_WPPF_PARAMS)
    new = distortion.get_mapping('GE_41RT_newton', GE_WPPF_PARAMS)

    assert new.maptype == 'GE_41RT_newton'
    assert len(new.params) == 6
    # Only the inverse differs; the forward model must be untouched.
    assert np.array_equal(old.apply(xy), new.apply(xy))

    # Trivial parameters still short-circuit.
    trivial = distortion.get_mapping('GE_41RT_newton', [0, 0, 0, 2, 4, 2])
    assert trivial.is_trivial
    assert trivial.apply_inverse(xy) is xy


@pytest.mark.parametrize('params', [GE_WPPF_PARAMS, PINCUSHION_PARAMS])
def test_newton_inverse_actually_inverts(params):
    """apply_inverse(apply(x)) == x, for exponents and signs the closed form
    cannot handle."""
    xy = _panel_points()
    dc = distortion.get_mapping('GE_41RT_newton', params)

    result = dc.apply_inverse(dc.apply(xy))
    assert not np.isnan(result).any()
    assert np.allclose(result, xy, atol=1e-9)

    # Equivalent to the legacy helper, up to the polar round trip's roundoff.
    fwd = dc.apply(xy)
    expected = inverse_distortion_numpy(
        np.hypot(fwd[:, 0], fwd[:, 1]),
        np.arctan2(fwd[:, 1], fwd[:, 0]),
        RHO_MAX,
        np.asarray(params, dtype=float),
    )
    assert np.allclose(np.hypot(result[:, 0], result[:, 1]), expected, atol=1e-12)


def test_newton_fixes_what_the_closed_form_gets_wrong():
    """Regression guard for hexrd#946: this is the reason the maptype exists."""
    xy = _panel_points()
    old = distortion.get_mapping('GE_41RT', GE_WPPF_PARAMS)
    new = distortion.get_mapping('GE_41RT_newton', GE_WPPF_PARAMS)
    fwd = old.apply(xy)

    def radial_error(dc):
        out = dc.apply_inverse(fwd)
        return np.hypot(out[:, 0], out[:, 1]) - np.hypot(xy[:, 0], xy[:, 1])

    # Exponents (2, 4, 2): the closed form assumes 2 and is off by tens of
    # microns in the corners, where Newton is exact.
    assert np.nanmax(np.abs(radial_error(old))) > 1e-2  # mm
    assert np.nanmax(np.abs(radial_error(new))) < 1e-9  # mm

    # Pincushion: the closed form is all NaN, Newton is finite everywhere.
    old_pin = distortion.get_mapping('GE_41RT', PINCUSHION_PARAMS)
    new_pin = distortion.get_mapping('GE_41RT_newton', PINCUSHION_PARAMS)
    fwd_pin = old_pin.apply(xy)
    assert np.isnan(old_pin.apply_inverse(fwd_pin)).all()
    assert np.isfinite(new_pin.apply_inverse(fwd_pin)).all()
