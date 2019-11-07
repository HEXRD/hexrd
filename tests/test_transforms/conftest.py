

# Python 2 support:
# In Python 2 there is no ModuleNotFoundError. Use ImportError instead.
try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError


def pytest_report_header(config):
    import numpy as np

    numpy_version = np.__version__
    
    try:
        import numba
        numba_version = numba.__version__
        
    except ModuleNotFoundError:
        numba_version = 'numba not installed'

    return 'numpy: {numpy_version}\nnumba: {numba_version}'.format(**locals())

