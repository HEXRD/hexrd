import importlib
import importlib.abc
import importlib.machinery
import sys

from .core.material import crystallography
from .core.material import jcpds
from .core.material import mksupport
from .core.material import spacegroup
from .core.material import symbols
from .core.material import symmetry
from .core.material import unitcell

# These are aliases for import paths, so we don't break old HEXRD scripts.
# We will verify the alias files *do not* exist, to avoid confusion.
module_aliases = {
    'hexrd.crystallography': crystallography,
    'hexrd.mksupport': mksupport,
    'hexrd.spacegroup': spacegroup,
    'hexrd.symbols': symbols,
    'hexrd.symmetry': symmetry,
    'hexrd.unitcell': unitcell,
}

for alias, module in module_aliases.items():
    try:
        file_exists = importlib.import_module(alias).__name__ == alias
    except ImportError:
        file_exists = False

    if file_exists:
        raise Exception(f'"{alias}" is an alias path and should not exist')

    sys.modules[alias] = module


from . import module_map


def __getattr__(name):
    # __getattr__ is only called if the attribute doesn't exist
    module = module_map.get("hexrd." + name)
    if module is not None:
        if isinstance(module, str):
            return importlib.import_module(module)
        return module
    raise AttributeError(f"Module `hexrd` has no attribute {name}")
