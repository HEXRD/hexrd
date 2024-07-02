import os
import functools


class DeprecatedFunctionError(Exception):
    """Custom exception for deprecated functions."""
    pass


def deprecated(new_func: str = None):
    """
    Decorator to mark functions as deprecated. Raises an error if
    the 'ACK_DEPRECATED' environment variable is not set. Alerts the
    user to the replacement function if provided.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if new_func is not None:
                print(
                    f"Warning: {func.__name__} is deprecated and is marked for"
                    f" removal. Please use {new_func} instead."
                )
            if os.getenv('ACK_DEPRECATED') != 'true':
                raise DeprecatedFunctionError(
                    f"Function {func.__name__} is deprecated. Set environment "
                    "variable 'ACK_DEPRECATED' to 'true' to acknowledge."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
