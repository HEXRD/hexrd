"""Crystal materials, imported on demand for lightweight workflow startup."""
from importlib import import_module

__all__ = ['_angstroms', '_kev', 'load_materials_hdf5', 'Material',
           'save_materials_hdf5']


def __getattr__(name):
    if name in __all__:
        value = getattr(import_module('.material', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
