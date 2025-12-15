import numpy as np
import pytest

from hexrd.core.distortion.dexela_2923 import Dexela_2923, _find_quadrant
import hexrd.core.distortion.dexela_2923 as dexela_2923


def test_dexela_2923_distortion_class():
    params = [0.0] * 8
    distortion_instance = Dexela_2923(params)

    # Test properties
    assert distortion_instance.maptype == "Dexela_2923"
    assert distortion_instance.is_trivial

    # Test invalid length raises
    with pytest.raises(AssertionError):
        distortion_instance.params = [0.0] * 5  # Invalid length

    # Test setting correct length
    distortion_instance.params = np.array([1.0] * 8)
    assert np.array_equal(distortion_instance.params, np.array([1.0] * 8))
    distortion_instance.params = np.array([0.0] * 8)  # Reset

    xy_in = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Trivial apply / apply_inverse
    xy_out = distortion_instance.apply(xy_in)
    np.testing.assert_array_equal(xy_out, xy_in)

    xy_out_inv = distortion_instance.apply_inverse(xy_in)
    np.testing.assert_array_equal(xy_out_inv, xy_in)

    # Non-trivial apply / apply_inverse
    distortion_instance.params = np.array([1.0] * 8)
    xy_out_nontrivial = distortion_instance.apply(xy_in)
    assert not np.array_equal(xy_out_nontrivial, xy_in)

    xy_out_inv_nontrivial = distortion_instance.apply_inverse(xy_in)
    assert not np.array_equal(xy_out_inv_nontrivial, xy_in)


def test_find_quadrant():
    xy_in = np.array([[1.0, 1.0],   # Quadrant 1
                      [-1.0, 1.0],  # Quadrant 2
                      [-1.0, -1.0], # Quadrant 3
                      [1.0, -1.0]]) # Quadrant 4
    quadrants = _find_quadrant(xy_in)
    expected_quadrants = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(quadrants, expected_quadrants)


def test_dexela_2923_distortion_edge_cases():
    params = [0.0] * 8
    distortion_instance = Dexela_2923(params)

    # Empty input
    xy_in_empty = np.empty((0, 2))
    xy_out_empty = distortion_instance.apply(xy_in_empty)
    assert xy_out_empty.shape == (0, 2)

    # Single-point input
    xy_in_single = np.array([[0.0, 0.0]])
    xy_out_single = distortion_instance.apply(xy_in_single)
    np.testing.assert_array_equal(xy_out_single, xy_in_single)


def test_dexela_2923_jitted_helpers():
    params = np.zeros(8, dtype=float)
    sample_xy = np.array([
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0]
    ])
    out = np.empty_like(sample_xy)

    dexela_2923._dexela_2923_distortion(out, sample_xy, params)
    assert np.array_equal(out[0], sample_xy[0] + params[0:2])
    assert np.array_equal(out[1], sample_xy[1] + params[2:4])
    assert np.array_equal(out[2], sample_xy[2] + params[4:6])
    assert np.array_equal(out[3], sample_xy[3] + params[6:8])

    out_inv = np.empty_like(sample_xy)
    dexela_2923._dexela_2923_inverse_distortion(out_inv, sample_xy, params)
    assert np.array_equal(out_inv[0], sample_xy[0] - params[0:2])
    assert np.array_equal(out_inv[1], sample_xy[1] - params[2:4])
    assert np.array_equal(out_inv[2], sample_xy[2] - params[4:6])
    assert np.array_equal(out_inv[3], sample_xy[3] - params[6:8])
