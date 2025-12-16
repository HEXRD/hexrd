import numpy as np
import pytest

from hexrd.core.distortion.dexela_2923_quad import (
    Dexela_2923_quad,
    _find_quadrant as _find_quadrant_quad,
)
import hexrd.core.distortion.dexela_2923_quad as dexela_2923_quad


def test_find_quadrant_quad():
    xy_in = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
        ]
    )
    quadrants = _find_quadrant_quad(xy_in)
    expected_quadrants = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(quadrants, expected_quadrants)


def test_dexela_2923_quad_distortion_class():
    params = [0.0] * 6
    distortion_instance = Dexela_2923_quad(params)

    assert distortion_instance.maptype == "Dexela_2923_quad"
    assert distortion_instance.is_trivial

    with pytest.raises(AssertionError):
        distortion_instance.params = [0.0] * 5

    distortion_instance.params = np.array([1.0] * 6)
    assert np.array_equal(distortion_instance.params, np.array([1.0] * 6))
    distortion_instance.params = np.array([0.0] * 6)

    xy_in = np.array([[1.0, 2.0], [3.0, 4.0]])

    xy_out = distortion_instance.apply(xy_in)
    np.testing.assert_array_equal(xy_out, xy_in)

    xy_out_inv = distortion_instance.apply_inverse(xy_in)
    np.testing.assert_array_equal(xy_out_inv, xy_in)

    distortion_instance.params = np.array([1.0] * 6)
    xy_out_nontrivial = distortion_instance.apply(xy_in)
    assert not np.array_equal(xy_out_nontrivial, xy_in)

    xy_out_inv_nontrivial = distortion_instance.apply_inverse(xy_in)
    assert not np.array_equal(xy_out_inv_nontrivial, xy_in)


def test_dexela_2923_quad_jitted_helpers():
    params = np.array([0.5, 1.2, 0.1, -0.3, 0.05, 0.8], dtype=float)

    sample_xy = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [0.2, -0.7],
        ],
        dtype=float,
    )

    forward = np.empty_like(sample_xy)
    dexela_2923_quad._dexela_2923_quad_inverse_distortion(
        forward, sample_xy, params
    )

    expected_forward = np.column_stack(
        [
            params[0]
            + params[1] * sample_xy[:, 0]
            + params[2] * sample_xy[:, 1],
            params[3]
            + params[4] * sample_xy[:, 0]
            + params[5] * sample_xy[:, 1],
        ]
    )
    np.testing.assert_allclose(forward, expected_forward, rtol=1e-9, atol=0.0)

    recovered = np.empty_like(sample_xy)
    dexela_2923_quad._dexela_2923_quad_distortion(recovered, forward, params)

    np.testing.assert_allclose(recovered, sample_xy, rtol=1e-6, atol=1e-9)
