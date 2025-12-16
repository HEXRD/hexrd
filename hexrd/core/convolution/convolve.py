# Based on code from astropy
# Licensed under a 3-clause BSD style license

import warnings

import os
import ctypes
from functools import partial

import numpy as np
from numpy.ctypeslib import ndpointer, load_library

from .utils import KernelSizeError, has_even_axis, raise_even_kernel_exception

LIBRARY_PATH = os.path.dirname(__file__)

try:
    _convolve = load_library("_convolve", LIBRARY_PATH)
except Exception:
    raise ImportError(
        "Convolution C extension is missing. Try re-building astropy."
    )

# The GIL is automatically released by default when calling functions imported
# from libraries loaded by ctypes.cdll.LoadLibrary(<path>)

# Declare prototypes
# Boundary None
_convolveNd_c = _convolve.convolveNd_c
_convolveNd_c.restype = None
_convolveNd_c.argtypes = [
    ndpointer(
        ctypes.c_double, flags={"C_CONTIGUOUS", "WRITEABLE"}
    ),  # return array
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # input array
    ctypes.c_uint,  # N dim
    # size array for input and result unless
    # embed_result_within_padded_region is False,
    # in which case the result array is assumed to be
    # input.shape - 2*(kernel.shape//2). Note: integer division.
    ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # kernel array
    ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),  # size array for kernel
    ctypes.c_bool,  # nan_interpolate
    ctypes.c_bool,  # embed_result_within_padded_region
    ctypes.c_uint,
]  # n_threads

# Disabling all doctests in this module until a better way of handling warnings
# in doctests can be determined
__doctest_skip__ = ['*']

BOUNDARY_OPTIONS = [None, 'fill', 'wrap', 'extend']

MAX_NORMALIZATION = 100


def _copy_input_if_needed(
    input,
    dtype=float,
    order='C',
    nan_treatment=None,
    mask=None,
    fill_value=None,
):
    # strip quantity attributes
    if hasattr(input, 'unit'):
        input = input.value
    output = input
    # Copy input
    try:
        # Anything that's masked must be turned into NaNs for the interpolation.
        # This requires copying. A copy is also needed for nan_treatment == 'fill'
        # A copy prevents possible function side-effects of the input array.
        if (
            nan_treatment == 'fill'
            or np.ma.is_masked(input)
            or mask is not None
        ):
            if np.ma.is_masked(input):
                # ``np.ma.maskedarray.filled()`` returns a copy, however there
                # is no way to specify the return type or order etc. In addition
                # ``np.nan`` is a ``float`` and there is no conversion to an
                # ``int`` type. Therefore, a pre-fill copy is needed for non
                # ``float`` masked arrays. ``subok=True`` is needed to retain
                # ``np.ma.maskedarray.filled()``. ``copy=False`` allows the fill
                # to act as the copy if type and order are already correct.
                output = np.array(
                    input, dtype=dtype, copy=False, order=order, subok=True
                )
                output = output.filled(fill_value)
            else:
                # Since we're making a copy, we might as well use `subok=False` to save,
                # what is probably, a negligible amount of memory.
                output = np.array(
                    input, dtype=dtype, copy=True, order=order, subok=False
                )

            if mask is not None:
                # mask != 0 yields a bool mask for all ints/floats/bool
                output[mask != 0] = fill_value
        else:
            # The call below is synonymous with np.asanyarray(array, ftype=float, order='C')
            # The advantage of `subok=True` is that it won't copy when array is an ndarray subclass. If it
            # is and `subok=False` (default), then it will copy even if `copy=False`. This uses less memory
            # when ndarray subclasses are passed in.
            output = np.array(
                input, dtype=dtype, copy=False, order=order, subok=True
            )
    except (TypeError, ValueError) as e:
        raise TypeError(
            'input should be a Numpy array or something '
            'convertible into a float array',
            e,
        )
    return output


# TODO: This function does not work when array or kernel are 32-bit float types.
def convolve(
    array,
    kernel,
    boundary='fill',
    fill_value=0.0,
    nan_treatment='interpolate',
    normalize_kernel=True,
    mask=None,
    preserve_nan=False,
    normalization_zero_tol=1e-8,
):
    """
    Convolve an array with a kernel.

    This routine differs from `scipy.ndimage.convolve` because
    it includes a special treatment for ``NaN`` values. Rather than
    including ``NaN`` values in the array in the convolution calculation, which
    causes large ``NaN`` holes in the convolved array, ``NaN`` values are
    replaced with interpolated values using the kernel as an interpolation
    function.

    Parameters
    ----------
    array : `numpy.ndarray`
        The array to convolve. This should be a 1, 2, or 3-dimensional array
        or a list or a set of nested lists representing a 1, 2, or
        3-dimensional array.
    kernel : `numpy.ndarray`
        The convolution kernel. The number of dimensions should match those for
        the array, and the dimensions should be odd in all directions.  If a
        masked array, the masked values will be replaced by ``fill_value``.
    boundary : str, optional
        A flag indicating how to handle boundaries:
            * `None`
                Set the ``result`` values to zero where the kernel
                extends beyond the edge of the array.
            * 'fill'
                Set values outside the array boundary to ``fill_value`` (default).
            * 'wrap'
                Periodic boundary that wrap to the other side of ``array``.
            * 'extend'
                Set values outside the array to the nearest ``array``
                value.
    fill_value : float, optional
        The value to use outside the array when using ``boundary='fill'``
    normalize_kernel : bool, optional
        Whether to normalize the kernel to have a sum of one.
    nan_treatment : {'interpolate', 'fill'}
        interpolate will result in renormalization of the kernel at each
        position ignoring (pixels that are NaN in the image) in both the image
        and the kernel.
        'fill' will replace the NaN pixels with a fixed numerical value (default
        zero, see ``fill_value``) prior to convolution
        Note that if the kernel has a sum equal to zero, NaN interpolation
        is not possible and will raise an exception.
    preserve_nan : bool
        After performing convolution, should pixels that were originally NaN
        again become NaN?
    mask : `None` or `numpy.ndarray`
        A "mask" array.  Shape must match ``array``, and anything that is masked
        (i.e., not 0/`False`) will be set to NaN for the convolution.  If
        `None`, no masking will be performed unless ``array`` is a masked array.
        If ``mask`` is not `None` *and* ``array`` is a masked array, a pixel is
        masked of it is masked in either ``mask`` *or* ``array.mask``.
    normalization_zero_tol: float, optional
        The absolute tolerance on whether the kernel is different than zero.
        If the kernel sums to zero to within this precision, it cannot be
        normalized. Default is "1e-8".

    Returns
    -------
    result : `numpy.ndarray`
        An array with the same dimensions and as the input array,
        convolved with kernel.  The data type depends on the input
        array type.  If array is a floating point type, then the
        return array keeps the same data type, otherwise the type
        is ``numpy.float``.

    Notes
    -----
    For masked arrays, masked values are treated as NaNs.  The convolution
    is always done at ``numpy.float`` precision.
    """

    if boundary not in BOUNDARY_OPTIONS:
        raise ValueError(
            "Invalid boundary option: must be one of {}".format(
                BOUNDARY_OPTIONS
            )
        )

    if nan_treatment not in ('interpolate', 'fill'):
        raise ValueError("nan_treatment must be one of 'interpolate','fill'")

    # OpenMP support is disabled at the C src code level, changing this will have
    # no effect.
    n_threads = 1

    # Keep refs to originals
    passed_kernel = kernel
    passed_array = array

    # The C routines all need float type inputs (so, a particular
    # bit size, endianness, etc.).  So we have to convert, which also
    # has the effect of making copies so we don't modify the inputs.
    # After this, the variables we work with will be array_internal, and
    # kernel_internal.  However -- we do want to keep track of what type
    # the input array was so we can cast the result to that at the end
    # if it's a floating point type.  Don't bother with this for lists --
    # just always push those as float.
    # It is always necessary to make a copy of kernel (since it is modified),
    # but, if we just so happen to be lucky enough to have the input array
    # have exactly the desired type, we just alias to array_internal
    # Convert kernel to ndarray if not already

    # Copy or alias array to array_internal
    array_internal = _copy_input_if_needed(
        passed_array,
        dtype=float,
        order='C',
        nan_treatment=nan_treatment,
        mask=mask,
        fill_value=np.nan,
    )
    array_dtype = getattr(passed_array, 'dtype', array_internal.dtype)
    # Copy or alias kernel to kernel_internal
    kernel_internal = _copy_input_if_needed(
        passed_kernel,
        dtype=float,
        order='C',
        nan_treatment=None,
        mask=None,
        fill_value=fill_value,
    )

    # Make sure kernel has all odd axes
    if has_even_axis(kernel_internal):
        raise ValueError("Kernel size must be odd in all axes.")

    # -----------------------------------------------------------------------
    # From this point onwards refer only to ``array_internal`` and
    # ``kernel_internal``.
    # Assume both are base np.ndarrays and NOT subclasses e.g. NOT
    # ``Kernel`` nor ``np.ma.maskedarray`` classes.
    # -----------------------------------------------------------------------

    if array_internal.ndim > 3:
        raise NotImplementedError(
            'convolve only supports 1, 2, and 3-dimensional '
            'arrays at this time'
        )
    elif array_internal.ndim != kernel_internal.ndim:
        raise ValueError('array and kernel have differing number of dimensions')

    array_shape = np.array(array_internal.shape)
    kernel_shape = np.array(kernel_internal.shape)
    pad_width = kernel_shape // 2

    # For boundary=None only the center space is convolved. All array indices within a
    # distance kernel.shape//2 from the edge are completely ignored (zeroed).
    # E.g. (1D list) only the indices len(kernel)//2 : len(array)-len(kernel)//2
    # are convolved. It is therefore not possible to use this method to convolve an
    # array by a kernel that is larger (see note below) than the array - as ALL pixels would be ignored
    # leaving an array of only zeros.
    # Note: For even kernels the correctness condition is array_shape > kernel_shape.
    # For odd kernels it is:
    # array_shape >= kernel_shape OR array_shape > kernel_shape-1 OR array_shape > 2*(kernel_shape//2).
    # Since the latter is equal to the former two for even lengths, the latter condition is complete.
    if boundary is None and not np.all(array_shape > 2 * pad_width):
        raise ValueError(
            "for boundary=None all kernel axes must be smaller than array's - "
            "use boundary in ['fill', 'extend', 'wrap'] instead."
        )

    # NaN interpolation significantly slows down the C convolution
    # computation. Since nan_treatment = 'interpolate', is the default
    # check whether it is even needed, if not, don't interpolate.
    # NB: np.isnan(array_internal.sum()) is faster than np.isnan(array_internal).any()
    nan_interpolate = (nan_treatment == 'interpolate') and np.isnan(
        array_internal.sum()
    )

    # Check if kernel is normalizable
    if normalize_kernel or nan_interpolate:
        kernel_sum = kernel_internal.sum()
        kernel_sums_to_zero = np.isclose(
            kernel_sum, 0, atol=normalization_zero_tol
        )

        if kernel_sum < 1.0 / MAX_NORMALIZATION or kernel_sums_to_zero:
            raise ValueError(
                "The kernel can't be normalized, because its sum is "
                "close to zero. The sum of the given kernel is < {}".format(
                    1.0 / MAX_NORMALIZATION
                )
            )

    # Mark the NaN values so we can replace them later if interpolate_nan is
    # not set
    if preserve_nan or nan_treatment == 'fill':
        initially_nan = np.isnan(array_internal)
        if nan_treatment == 'fill':
            array_internal[initially_nan] = fill_value

    # Avoid any memory allocation within the C code. Allocate output array
    # here and pass through instead.
    result = np.zeros(array_internal.shape, dtype=float, order='C')

    embed_result_within_padded_region = True
    array_to_convolve = array_internal
    if boundary in ('fill', 'extend', 'wrap'):
        embed_result_within_padded_region = False
        if boundary == 'fill':
            # This method is faster than using numpy.pad(..., mode='constant')
            array_to_convolve = np.full(
                array_shape + 2 * pad_width,
                fill_value=fill_value,
                dtype=float,
                order='C',
            )
            # Use bounds [pad_width[0]:array_shape[0]+pad_width[0]] instead of [pad_width[0]:-pad_width[0]]
            # to account for when the kernel has size of 1 making pad_width = 0.
            if array_internal.ndim == 1:
                array_to_convolve[
                    pad_width[0] : array_shape[0] + pad_width[0]
                ] = array_internal
            elif array_internal.ndim == 2:
                array_to_convolve[
                    pad_width[0] : array_shape[0] + pad_width[0],
                    pad_width[1] : array_shape[1] + pad_width[1],
                ] = array_internal
            else:
                array_to_convolve[
                    pad_width[0] : array_shape[0] + pad_width[0],
                    pad_width[1] : array_shape[1] + pad_width[1],
                    pad_width[2] : array_shape[2] + pad_width[2],
                ] = array_internal
        else:
            np_pad_mode_dict = {
                'fill': 'constant',
                'extend': 'edge',
                'wrap': 'wrap',
            }
            np_pad_mode = np_pad_mode_dict[boundary]
            pad_width = kernel_shape // 2

            if array_internal.ndim == 1:
                np_pad_width = (pad_width[0],)
            elif array_internal.ndim == 2:
                np_pad_width = ((pad_width[0],), (pad_width[1],))
            else:
                np_pad_width = (
                    (pad_width[0],),
                    (pad_width[1],),
                    (pad_width[2],),
                )

            array_to_convolve = np.pad(
                array_internal, pad_width=np_pad_width, mode=np_pad_mode
            )

    _convolveNd_c(
        result,
        array_to_convolve,
        array_to_convolve.ndim,
        np.array(array_to_convolve.shape, dtype=ctypes.c_size_t, order='C'),
        kernel_internal,
        np.array(kernel_shape, dtype=ctypes.c_size_t, order='C'),
        nan_interpolate,
        embed_result_within_padded_region,
        n_threads,
    )

    # So far, normalization has only occured for nan_treatment == 'interpolate'
    # because this had to happen within the C extension so as to ignore
    # any NaNs
    if normalize_kernel:
        if not nan_interpolate:
            result /= kernel_sum
    elif nan_interpolate:
        result *= kernel_sum

    if nan_interpolate and not preserve_nan and np.isnan(result.sum()):
        warnings.warn(
            "nan_treatment='interpolate', however, NaN values detected "
            "post convolution. A contiguous region of NaN values, larger "
            "than the kernel size, are present in the input array. "
            "Increase the kernel size to avoid this."
        )

    if preserve_nan:
        result[initially_nan] = np.nan

    # Convert result to original data type
    if array_dtype.kind == 'f':
        # Try to preserve the input type if it's a floating point type
        # Avoid making another copy if possible
        try:
            return result.astype(array_dtype, copy=False)
        except TypeError:
            return result.astype(array_dtype)
    else:
        return result
