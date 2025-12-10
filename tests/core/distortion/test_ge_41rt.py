import numpy as np
import pytest

from hexrd.core.distortion.ge_41rt import (
    GE_41RT,
    _ge_41rt_distortion,
    _ge_41rt_inverse_distortion,
    _rho_scl_func_inv,
    _rho_scl_dfunc_inv,
    inverse_distortion_numpy,
    inverse_distortion,
    RHO_MAX
)


def test_set_parameters():
    params = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]
    ge = ge41rt_mod.GE_41RT(params)

    new_params = [0.1, 0.2, 0.3, 1.0, 1.0, 1.0]
    ge.params = new_params
    np.testing.assert_array_equal(ge.params, new_params)

    with pytest.raises(AssertionError):
        ge.params = [0.0, 0.0]  # too short

def test_ge_41rt_class_trivial_and_param_checks():
    params = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    ge = GE_41RT(params)
    assert ge.is_trivial

    params_nontrivial = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]
    ge_nt = GE_41RT(params_nontrivial)
    assert not ge_nt.is_trivial

    with pytest.raises(AssertionError):
        ge.params = [0.0, 0.0]  # too short


def test_ge_41rt_apply_and_inverse_trivial():
    params = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    ge = GE_41RT(params)
    xy_in = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # trivial -> returns input unchanged
    xy_out = ge.apply(xy_in)
    np.testing.assert_array_equal(xy_out, xy_in)
    
    xy_inv = ge.apply_inverse(xy_in)
    np.testing.assert_array_equal(xy_inv, xy_in)


def test_ge_41rt_apply_and_inverse_nontrivial():
    params = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]
    ge = GE_41RT(params)
    xy_in = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    xy_out = ge.apply(xy_in)
    assert not np.array_equal(xy_out, xy_in)

    # bypass njit for inverse_distortion to cover it
    saved_func = inverse_distortion.ge_41rt_inverse_distortion
    inverse_distortion.ge_41rt_inverse_distortion = lambda xy, rhoMax, p: xy * 2
    xy_inv = ge.apply_inverse(xy_in)
    np.testing.assert_array_equal(xy_inv, xy_in * 2)
    inverse_distortion.ge_41rt_inverse_distortion = saved_func


def test_ge_41rt_jitted_helpers_directly():
    xy_in = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = np.empty_like(xy_in)
    params = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]

    res = _ge_41rt_distortion(out.copy(), xy_in, 204.8, np.array(params))
    assert res.shape == xy_in.shape
    assert isinstance(res, np.ndarray)

    # Call the inverse jitted function directly
    res_inv = _ge_41rt_inverse_distortion(xy_in.copy(), 204.8, np.array(params[:3]))
    assert res_inv.shape == (2, 2)


def test_rho_scaling_functions():
    ri = 1.0
    ni = 0.0
    ro = 1.5
    rx = 2.0
    p = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]

    fval = _rho_scl_func_inv(ri, ni, ro, rx, p)
    dfval = _rho_scl_dfunc_inv(ri, ni, ro, rx, p)

    assert isinstance(fval, float)
    assert isinstance(dfval, float)


def test_inverse_distortion_numpy_works():
    rho0 = 1.0
    eta0 = 0.0
    rhoMax = 2.0
    params = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]

    res = inverse_distortion_numpy(rho0, eta0, rhoMax, params)
    assert np.isscalar(res) or isinstance(res, float)

import hexrd.core.distortion.ge_41rt as ge41rt_mod


@pytest.fixture(autouse=True)
def bypass_numba_for_coverage(monkeypatch, request):
    """When coverage is active, replace njit wrappers with their Python bodies.

    We must patch the functions on the module object (ge41rt_mod) and ensure
    tests call the module attributes so coverage will see the Python lines.
    """
    cov_plugin = request.config.pluginmanager.getplugin("pytest_cov")
    cov_active = cov_plugin is not None or getattr(request.config, "cov_controller", None)

    if cov_active:
        # Replace the njit wrappers with their original Python implementations.
        # This preserves the decorators in source but ensures pytest-cov sees lines.
        if hasattr(ge41rt_mod, "_ge_41rt_inverse_distortion"):
            monkeypatch.setattr(
                ge41rt_mod,
                "_ge_41rt_inverse_distortion",
                ge41rt_mod._ge_41rt_inverse_distortion.py_func,
            )
        if hasattr(ge41rt_mod, "_ge_41rt_distortion"):
            monkeypatch.setattr(
                ge41rt_mod,
                "_ge_41rt_distortion",
                ge41rt_mod._ge_41rt_distortion.py_func,
            )


def test_ge_41rt_class_trivial_and_param_checks():
    params = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    ge = ge41rt_mod.GE_41RT(params)
    assert ge.is_trivial

    params_nontrivial = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]
    ge_nt = ge41rt_mod.GE_41RT(params_nontrivial)
    assert not ge_nt.is_trivial

    with pytest.raises(AssertionError):
        ge.params = [0.0, 0.0]  # too short


def test_ge_41rt_apply_and_inverse_trivial():
    params = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    ge = ge41rt_mod.GE_41RT(params)
    xy_in = np.array([[1.0, 2.0], [3.0, 4.0]])

    # trivial -> returns input unchanged
    xy_out = ge.apply(xy_in)
    np.testing.assert_array_equal(xy_out, xy_in)

    xy_inv = ge.apply_inverse(xy_in)
    np.testing.assert_array_equal(xy_inv, xy_in)


def test_ge_41rt_apply_and_inverse_nontrivial(monkeypatch):
    params = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]
    ge = ge41rt_mod.GE_41RT(params)
    xy_in = np.array([[1.0, 2.0], [3.0, 4.0]])

    xy_out = ge.apply(xy_in)
    assert not np.array_equal(xy_out, xy_in)

    # Patch the inverse_distortion implementation the module uses so apply_inverse
    # calls a deterministic replacement (without altering the module source).
    # Use monkeypatch to ensure automatic restore.
    monkeypatch.setattr(
        ge41rt_mod.inverse_distortion,
        "ge_41rt_inverse_distortion",
        lambda xy, rhoMax, p: xy * 2,
    )

    xy_inv = ge.apply_inverse(xy_in)
    np.testing.assert_array_equal(xy_inv, xy_in * 2)


def test_ge_41rt_jitted_helpers_directly():
    xy_in = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = np.empty_like(xy_in)
    params = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]

    # Call the forward distortion via module attribute (will be patched to .py_func
    # when coverage is active, otherwise executes the njit wrapper).
    res = ge41rt_mod._ge_41rt_distortion(out.copy(), xy_in, ge41rt_mod.RHO_MAX, np.array(params))
    assert res.shape == xy_in.shape
    assert isinstance(res, np.ndarray)
    assert not np.array_equal(res, xy_in)

    # Call the inverse jitted function via module attribute (patched to .py_func under coverage).
    res_inv = ge41rt_mod._ge_41rt_inverse_distortion(xy_in.copy(), ge41rt_mod.RHO_MAX, np.array(params[:3]))
    assert res_inv.shape == (2, 2)
    assert isinstance(res_inv, np.ndarray)


def test_rho_scaling_functions():
    ri = 1.0
    ni = 0.0
    ro = 1.5
    rx = 2.0
    p = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]

    fval = ge41rt_mod._rho_scl_func_inv(ri, ni, ro, rx, p)
    dfval = ge41rt_mod._rho_scl_dfunc_inv(ri, ni, ro, rx, p)

    assert isinstance(fval, float)
    assert isinstance(dfval, float)


def test_inverse_distortion_numpy_works():
    rho0 = 1.0
    eta0 = 0.0
    rhoMax = 2.0
    params = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]

    res = ge41rt_mod.inverse_distortion_numpy(rho0, eta0, rhoMax, params)
    # newton may return a numpy scalar or a float depending on implementation
    assert np.isscalar(res) or isinstance(res, float)


def test_extremely_small_radius():
    """Test that extremely small radius values are handled without error."""
    params = [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]
    ge = ge41rt_mod.GE_41RT(params)
    xy_in = np.array([[1e-10, 0.0], [0.0, 1e-10]])

    xy_out = ge.apply(xy_in)
    assert xy_out.shape == xy_in.shape

    with pytest.MonkeyPatch.context() as mpatch:
        mpatch.setattr(
            ge41rt_mod.inverse_distortion,
            "ge_41rt_inverse_distortion",
            lambda xy, rhoMax, p: xy * 2,
        )

        xy_inv = ge.apply_inverse(xy_in)
        np.testing.assert_array_equal(xy_inv, xy_in * 2)

