# encoding: utf-8
"""Decorators that don't go anywhere else.

This module contains decorators that don't really go with another module
in :mod:`hexrd.utils`. Before putting something here please see if it should
go into another topical module in :mod:`hexrd.utils`.
"""

import hashlib

import numpy as np

from hexrd.constants import USE_NUMBA


def undoc(func):
    """Mark a function or class as undocumented.
    This is found by inspecting the AST, so for now it must be used directly
    as @undoc, not as e.g. @decorators.undoc
    """
    return func


class memoize:
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        all_args = list(args) + sorted(kwargs.items())
        key = self.make_hashable(all_args)
        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)

        return self.cache[key]

    @staticmethod
    def make_hashable(items):

        def convert(x):
            # Perform any conversions here to make a variable hashable
            if isinstance(x, np.ndarray):
                # Create an sha1 of the data, and throw in a string
                # and the shape.
                return ('__type_np.ndarray', x.shape,
                        hashlib.sha1(x).hexdigest())
            elif isinstance(x, (list, tuple)):
                return memoize.make_hashable(x)
            elif isinstance(x, dict):
                return memoize.make_hashable(sorted(x.items()))
            return x

        return tuple(map(convert, items))


def numba_njit_if_available(func=None, *args, **kwargs):
    # Forwards decorator to numba.njit if numba is available
    # Otherwise, does nothing.

    def decorator(func):
        if USE_NUMBA:
            import numba
            return numba.njit(*args, **kwargs)(func)
        else:
            # Do nothing...
            return func

    if func is None:
        return decorator
    else:
        return decorator(func)
