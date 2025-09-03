# encoding: utf-8
"""Decorators that don't go anywhere else.

This module contains decorators that don't really go with another module
in :mod:`hexrd.utils`. Before putting something here please see if it should
go into another topical module in :mod:`hexrd.utils`.
"""

from collections import OrderedDict
from functools import wraps

import numba
import numpy as np
import xxhash


def undoc(func):
    """Mark a function or class as undocumented.
    This is found by inspecting the AST, so for now it must be used directly
    as @undoc, not as e.g. @decorators.undoc
    """
    return func


def memoize(func=None, maxsize=2):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    This uses an LRU cache, where, before the maxsize is exceeded, the
    least recently used item will be removed.

    Numpy array arguments will be hashed for use in the cache.

    We are not using `functools.lru_cache()` only because it requires
    hashed arguments. Here, we can create hashes for the arguments on
    our own, and still pass the unhashed arguments to the function.
    """

    def decorator(func):

        cache = OrderedDict()
        hits = 0
        misses = 0

        def cache_info():
            return {
                'hits': hits,
                'misses': misses,
                'maxsize': maxsize,
                'currsize': len(cache),
            }

        def set_cache_maxsize(x):
            nonlocal maxsize
            maxsize = x

            while len(cache) > maxsize:
                # Remove the left item (least recently used)
                cache.popitem(last=False)

        setattr(func, 'cache_info', cache_info)
        setattr(func, 'set_cache_maxsize', set_cache_maxsize)

        @wraps(func)
        def wrapped(*args, **kwargs):
            nonlocal misses
            nonlocal hits

            all_args = list(args) + sorted(kwargs.items())
            key = _make_hashable(all_args)
            if key not in cache:
                # Make sure the current size is less than the max size
                while len(cache) >= maxsize:
                    # Remove the left item (least recently used)
                    cache.popitem(last=False)

                output = func(*args, **kwargs)
                if isinstance(output, np.ndarray):
                    # Make the array readonly so that caller functions *cannot*
                    # modify the cached output array. Otherwise, we run into
                    # hard-to-track-down bugs.
                    output.flags.writeable = False

                # This inserts the item on the right (most recently used)
                cache[key] = output
                misses += 1
            else:
                # Move the item to the right (most recently used)
                cache.move_to_end(key)
                hits += 1

            return cache[key]

        return wrapped

    if func is None:
        return decorator
    else:
        return decorator(func)


def _make_hashable(items):
    """Convert a list of items into hashable forms

    Note: they may not be able to be converted back.
    """

    def convert(x):
        # Perform any conversions here to make a variable hashable
        if isinstance(x, np.ndarray):
            # Create an sha1 of the data, and throw in a string
            # and the shape.
            x = np.ascontiguousarray(x)
            return ('__type_np.ndarray', x.shape,
                    xxhash.xxh3_128_hexdigest(x))
        elif isinstance(x, (list, tuple)):
            return _make_hashable(x)
        elif isinstance(x, dict):
            return _make_hashable(sorted(x.items()))
        return x

    return tuple(map(convert, items))


# A decorator to limit the number of numba threads
def limit_numba_threads(max_threads):
    def decorator(func):
        def wrapper(*args, **kwargs):
            prev_num_threads = numba.get_num_threads()
            new_num_threads = min(prev_num_threads, max_threads)
            numba.set_num_threads(new_num_threads)
            try:
                return func(*args, **kwargs)
            finally:
                numba.set_num_threads(prev_num_threads)

        return wrapper

    return decorator
