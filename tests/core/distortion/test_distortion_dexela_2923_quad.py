import numpy as np
import pytest

from hexrd.core.distortion.dexela_2923_quad import Dexela_2923_quad, _find_quadrant as _find_quadrant_quad
import hexrd.core.distortion.dexela_2923_quad as dexela_2923_quad


@pytest.fixture(autouse=True)
def bypass_numba_for_coverage(monkeypatch, request):
    """When coverage is active, replace njit wrappers with their Python bodies.

    This keeps the @njit decorators in the source but ensures pytest-cov
    / Codecov see the Python lines by executing the original functions via
    .py_func during coverage runs.
    """
    cov_plugin = request.config.pluginmanager.getplugin("pytest_cov")
    cov_active = cov_plugin is not None or getattr(request.config, "cov_controller", None)

    if cov_active:
        monkeypatch.setattr(
            dexela_2923_quad,
            "_dexela_2923_quad_distortion",
            dexela_2923_quad._dexela_2923_quad_distortion.py_func,
        )
        monkeypatch.setattr(
            dexela_2923_quad,
            "_dexela_2923_quad_inverse_distortion",
            dexela_2923_quad._dexela_2923_quad_inverse_distortion.py_func
        )


def test_find_quadrant_quad():
    xy_in = np.array([[1.0, 1.0],   # Quadrant 1
                      [-1.0, 1.0],  # Quadrant 2
                      [-1.0, -1.0], # Quadrant 3
                      [1.0, -1.0]]) # Quadrant 4
    quadrants = _find_quadrant_quad(xy_in)
    expected_quadrants = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(quadrants, expected_quadrants)

def test_dexela_2923_quad_distortion_class():
    params = [0.0] * 6
    distortion_instance = Dexela_2923_quad(params)

    # Test properties
    assert distortion_instance.maptype == "Dexela_2923_quad"
    assert distortion_instance.is_trivial

    # Test invalid length raises
    with pytest.raises(AssertionError):
        distortion_instance.params = [0.0] * 5  # Invalid length

    # Test setting correct length
    distortion_instance.params = np.array([1.0] * 6)
    assert np.array_equal(distortion_instance.params, np.array([1.0] * 6))
    distortion_instance.params = np.array([0.0] * 6)  # Reset

    xy_in = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Trivial apply / apply_inverse
    xy_out = distortion_instance.apply(xy_in)
    np.testing.assert_array_equal(xy_out, xy_in)

    xy_out_inv = distortion_instance.apply_inverse(xy_in)
    np.testing.assert_array_equal(xy_out_inv, xy_in)

    # Non-trivial apply / apply_inverse
    distortion_instance.params = np.array([1.0] * 6)
    xy_out_nontrivial = distortion_instance.apply(xy_in)
    assert not np.array_equal(xy_out_nontrivial, xy_in)

    xy_out_inv_nontrivial = distortion_instance.apply_inverse(xy_in)
    assert not np.array_equal(xy_out_inv_nontrivial, xy_in)

def test_dexela_2923_quad_jitted_helpers():
    """Exercise the jitted forward/inverse helpers via the module attributes.

    The test computes the forward (inverse_distortion) mapping, then applies
    the algebraic solver (distortion) to recover the original coordinates.
    Calling the functions through the module ensures the coverage-time fixture
    can patch them to their .py_func bodies.
    """
    params = np.array([0.5, 1.2, 0.1, -0.3, 0.05, 0.8], dtype=float)

    # Points chosen to cover quadrants / generic coordinates
    sample_xy = np.array([
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [0.2, -0.7],   # extra arbitrary point
    ], dtype=float)

    # Compute forward mapping using the inverse_distortion function
    forward = np.empty_like(sample_xy)
    dexela_2923_quad._dexela_2923_quad_inverse_distortion(forward, sample_xy, params)

    # Sanity-check the forward mapping matches the expected linear form
    expected_forward = np.column_stack([
        params[0] + params[1] * sample_xy[:, 0] + params[2] * sample_xy[:, 1],
        params[3] + params[4] * sample_xy[:, 0] + params[5] * sample_xy[:, 1],
    ])
    np.testing.assert_allclose(forward, expected_forward, rtol=1e-9, atol=0.0)

    # Use the algebraic solver to recover original points from 'forward'
    recovered = np.empty_like(sample_xy)
    dexela_2923_quad._dexela_2923_quad_distortion(recovered, forward, params)

    # The solver should recover the original coordinates (within numerical tol)
    np.testing.assert_allclose(recovered, sample_xy, rtol=1e-6, atol=1e-9)