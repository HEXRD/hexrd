# encoding: utf-8
"""Decorators that don't go anywhere else.

This module contains decorators that don't really go with another module
in :mod:`hexrd.utils`. Before putting something here please see if it should
go into another topical module in :mod:`hexrd.utils`.
"""

import collections

from hexrd.constants import USE_NUMBA


def undoc(func):
    """Mark a function or class as undocumented.
    This is found by inspecting the AST, so for now it must be used directly
    as @undoc, not as e.g. @decorators.undoc
    """
    return func


class memoized:
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kw):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args, **kw)
        key = (args, frozenset(kw.items()))
        if key in self.cache:
            return self.cache[key]
        else:
            value = self.func(*args, **kw)
            self.cache[key] = value
            return value


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
