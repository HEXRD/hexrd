# Based on code from astropy
# Licensed under a 3-clause BSD style license
import ctypes
import numpy as np


class DiscretizationError(Exception):
    """
    Called when discretization of models goes wrong.
    """


class KernelSizeError(Exception):
    """
    Called when size of kernels is even.
    """


def has_even_axis(array):
    if isinstance(array, (list, tuple)):
        return not len(array) % 2
    else:
        return any(not axes_size % 2 for axes_size in array.shape)


def raise_even_kernel_exception():
    raise KernelSizeError("Kernel size must be odd in all axes.")
