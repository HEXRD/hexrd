"""package init for hexrd"""
import logging
from pkg_resources import DistributionNotFound, get_distribution

# Release data
__author__ = 'HEXRD Development Team <praxes@googlegroups.com>'
__license__ = 'BSD'

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = None

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _readenv(name, ctor, default):
    try:
        import os
        res = os.environ[name]
        del os
    except KeyError:
        del os
        return default
    else:
        try:
            return ctor(res)
        except:
            import warnings
            warnings.warn("environ %s defined but failed to parse '%s'" %
                          (name, res), RuntimeWarning)
            del warnings
            return default


# 0 = do NOT use numba
# 1 = use numba (default)
USE_NUMBA = _readenv("HEXRD_USE_NUMBA", int, 1)
if USE_NUMBA:
    try:
        import numba
    except ImportError:
        print("*** Numba not available, processing may run slower ***")
        USE_NUMBA = False

del _readenv

# path for pkgutil
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# Documentation URL (temporary)
doc_url = "https://github.com/donald-e-boyce/hexrd3/wiki"
