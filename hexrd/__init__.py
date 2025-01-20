import importlib
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
