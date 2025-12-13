import numpy as np
import pytest

from hexrd.core.convolution.convolve import _copy_input_if_needed, convolve
from hexrd.core.valunits import valWUnit


def assert_basic_convolve_result(result, arr):
    """Common assertions used for successful convolve calls."""
    assert result.shape == arr.shape
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.floating)


def test_copy_input_valwunit_and_invalid_type():
    arr = np.array([1, 2, 3], dtype=np.float64)
    vwu = valWUnit("input_array", "length", arr, "mm")
    result = _copy_input_if_needed(vwu, np.float64)
    assert np.array_equal(result, arr)

    with pytest.raises(TypeError):
        _copy_input_if_needed({"1": 2}, np.float64)


def test_copy_input_contiguous_vs_noncontiguous():
    arr = np.array([1, 2, 3], dtype=np.float64)
    res = _copy_input_if_needed(arr, np.float64)
    assert res is arr

    arr2 = np.array([1, 2, 3, 4], dtype=np.float64)
    view = arr2[::2] 
    res_view = _copy_input_if_needed(view, np.float64)
    assert np.array_equal(res_view, view)
    assert res_view is not view

    int_arr = np.array([1, 2, 3], dtype=np.int32)
    res_conv = _copy_input_if_needed(int_arr, np.float64)
    assert np.array_equal(res_conv, int_arr)
    assert res_conv.dtype == np.float64
    assert res_conv is not int_arr


def test_copy_input_with_mask_argument_copies():
    arr_masked = np.ma.array([1, 2, 3], mask=[0, 1, 0], dtype=np.float64)
    res1 = _copy_input_if_needed(arr_masked, np.float64, mask=arr_masked.mask)
    assert res1 is not arr_masked

    arr_plain = np.array([1, 2, 3], dtype=np.float64)
    res2 = _copy_input_if_needed(arr_plain, np.float64, mask=[0, 1, 0])
    assert res2 is not arr_plain

# ---- convolve tests --------------

def test_convolve_smoke_and_basic_shapes():
    """A few representative successful calls are consolidated here."""
    arr1 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    kernel1d = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    assert_basic_convolve_result(convolve(arr1, kernel1d), arr1)

    arr2d = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=float)
    kernel2d = np.ones((3, 3), dtype=float)
    assert_basic_convolve_result(convolve(arr2d, kernel2d), arr2d)

    # 3D
    arr3 = np.random.rand(4, 4, 4).astype(np.float64)
    kernel3 = np.full((3, 3, 3), 1.0 / 27.0, dtype=np.float64)
    assert_basic_convolve_result(convolve(arr3, kernel3), arr3)


def test_convolve_even_kernel_axes_raises():
    arr = np.array([1, 2, 3], dtype=np.float64)
    kernel = np.array([[0.25, 0.25],
                       [0.25, 0.25]], dtype=np.float64)
    with pytest.raises(ValueError):
        convolve(arr, kernel)


def test_convolve_invalid_boundary_and_nan_treatment_options_raise():
    arr = np.array([1, 2, 3], dtype=np.float64)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)

    with pytest.raises(ValueError):
        convolve(arr, kernel, boundary="invalid_option")

    with pytest.raises(ValueError):
        convolve(arr, kernel, nan_treatment="invalid_option")


def test_convolve_nan_fill_and_interpolate_variants():
    arr = np.array([[1, 2, np.nan],
                    [4, np.nan, 6],
                    [7, 8, 9]], dtype=np.float64)
    kernel = np.array([[0.25, 0.5, 0.25],
                       [0.25, 0.5, 0.25],
                       [0.25, 0.5, 0.25]], dtype=np.float64)

    res_fill = convolve(arr, kernel, nan_treatment="fill", fill_value=0)
    assert_basic_convolve_result(res_fill, arr)

    res_interp_no_norm = convolve(arr, kernel, nan_treatment="interpolate", normalize_kernel=False)
    assert_basic_convolve_result(res_interp_no_norm, arr)

    res_interp_norm = convolve(arr, kernel, nan_treatment="interpolate", normalize_kernel=True)
    assert_basic_convolve_result(res_interp_norm, arr)

    res_interp_preserve = convolve(arr, kernel, nan_treatment="interpolate", preserve_nan=True)
    assert_basic_convolve_result(res_interp_preserve, arr)


def test_nan_interpolate_preserve_false_warns_when_all_nan_except_one():
    arr = np.full((10, 10), np.nan, dtype=np.float64)
    arr[5, 5] = 1.0
    kernel = np.array([[0.25, 0.5, 0.25],
                       [0.25, 0.5, 0.25],
                       [0.25, 0.5, 0.25]], dtype=np.float64)

    with pytest.warns(UserWarning):
        convolve(arr, kernel, nan_treatment="interpolate", preserve_nan=False)


def test_convolve_dimension_and_boundary_cases():
    arr4 = np.random.rand(2, 2, 2, 2).astype(np.float64)
    kernel1 = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    with pytest.raises(NotImplementedError):
        convolve(arr4, kernel1)

    for dim in (1, 2, 3):
        shape = (5,) * dim
        arr = np.random.rand(*shape).astype(np.float64)
        kshape = (3,) * dim
        kernel = np.ones(kshape, dtype=np.float64) / np.prod(kshape)
        res = convolve(arr, kernel, boundary="extend")
        assert_basic_convolve_result(res, arr)

    arr_wrap = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.float64)
    kernel_wrap = np.array([[0.25, 0.5, 0.25],
                            [0.25, 0.5, 0.25],
                            [0.25, 0.5, 0.25]], dtype=np.float64)
    assert_basic_convolve_result(convolve(arr_wrap, kernel_wrap, boundary="wrap"), arr_wrap)


def test_array_and_kernel_dimension_mismatch_and_boundary_none_small_array_and_kernel_sum_zero():
    arr2d = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float64)
    kernel1 = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    with pytest.raises(ValueError):
        convolve(arr2d, kernel1)

    arr_small = np.array([[1, 2],
                          [3, 4]], dtype=np.float64)
    kernel3 = np.array([[0.25, 0.5, 0.25],
                        [0.25, 0.5, 0.25],
                        [0.25, 0.5, 0.25]], dtype=np.float64)
    with pytest.raises(ValueError):
        convolve(arr_small, kernel3, boundary=None)

    kernel_zero_sum = np.array([[1, -2, 1],
                                [-2, 4, -2],
                                [1, -2, 1]], dtype=np.float64)
    with pytest.raises(ValueError):
        convolve(arr2d, kernel_zero_sum)


def test_convolve_with_float_arrays():
    arr_f = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float32)
    kernel_f = np.array([[0.25, 0.5, 0.25],
                         [0.25, 0.5, 0.25],
                         [0.25, 0.5, 0.25]], dtype=np.float32)
    res_f = convolve(arr_f, kernel_f)
    assert_basic_convolve_result(res_f, arr_f)
