"""Distortion package (python 3)"""

import abc
import pkgutil
from importlib import import_module

# import all modules to generate registry
from .registry import Registry
_builtin_modules = ('identity', 'nyi', 'ge_41rt', 'dexela_2923', 'dexela_2923_quad')


def _load_builtins():
    for name in _builtin_modules:
        import_module(f'.{name}', __name__)


def __getattr__(name):
    if name in _builtin_modules:
        return import_module(f'.{name}', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = ['maptypes', 'get_mapping']

# __path__ = []
# for loader, name, ispkg in pkgutil.iter_modules(__path__):
#     if name not in 'registry':
#         import_module(name, )

# Interface


def maptypes():
    """
    Returns list of available maptypes.

    Returns
    -------
    list
        The list of distortion functions keys in the registry.

    """
    _load_builtins()
    return list(Registry.distortion_registry.keys())


def get_mapping(maptype, params, **kwargs):
    """
    Initializes specified distortion class.

    Parameters
    ----------
    maptype : str
        The maptype (key) for the desired distortion function in the registry.
    params : array_like
        The parameters associated with the evaluation of the distortion
        function specified by maptype.
    **kwargs : dict
        Optional keyword arguments to pass the distortion function.

    Returns
    -------
    class
        The distortion functin interface associated with maptype.

    """
    _load_builtins()
    cls = Registry.distortion_registry[maptype]
    return cls(params, **kwargs)
