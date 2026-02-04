from contextlib import contextmanager
from typing import Optional
import warnings


@contextmanager
def ignore_warnings(category: Optional[type[Warning]] = None):
    with warnings.catch_warnings():
        if category is not None:
            warnings.simplefilter('ignore', category=category)
        else:
            warnings.simplefilter('ignore')
        yield
