import logging
from contextlib import contextmanager, redirect_stdout
import io


@contextmanager
def redirect_stdout_to_logger(level=logging.DEBUG):
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
    logger = logging.getLogger(__name__)
    captured = io.StringIO()

    with redirect_stdout(captured):
        yield

    output = captured.getvalue()
    logger.log(level, output)
