from contextlib import contextmanager
from typing import Optional
import warnings


@contextmanager
def ignore_warnings(category: Optional[Warning] = None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=category)
        yield
