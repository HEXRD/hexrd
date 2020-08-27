"""Distortion package (python 3)"""
import abc
import pkgutil
from importlib import import_module

# import all modules to generate registry
from .registry import Registry
from . import identity
from . import nyi
from . import ge_41rt

__all__ = ['maptypes', 'get_mapping']

# __path__ = []
# for loader, name, ispkg in pkgutil.iter_modules(__path__):
#     if name not in 'registry':
#         import_module(name, )

# Interface


def maptypes():
    """Return list of maptypes"""
    return list(Registry.distortion_registry.keys())

def get_mapping(maptype, params, **kwargs):
    cls = Registry.distortion_registry[maptype]
    return cls(params, **kwargs)
