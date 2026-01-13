from contextlib import contextmanager, redirect_stdout
from typing import Iterator
import logging
import io


@contextmanager
def redirect_stdout_to_logger(
    logger: logging.Logger, level: int = logging.DEBUG
) -> Iterator[None]:
    """
    Context manager to redirect stdout to a logger.

    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.DEBUG)

    Examples
    --------
    >>> with redirect_stdout_to_logger(logging.DEBUG):
    ...     some_function_that_prints()
    """
    captured = io.StringIO()

    with redirect_stdout(captured):
        yield

    output = captured.getvalue()
    logger.log(level, output)
