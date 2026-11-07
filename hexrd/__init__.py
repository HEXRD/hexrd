import importlib
import types

from pathlib import Path

from . import module_map

# These are aliases for import paths, so we don't break old HEXRD scripts.
# We will verify the alias files *do not* exist, to avoid confusion.
module_aliases = {
    'hexrd.crystallography': 'hexrd.core.material.crystallography',
    'hexrd.mksupport': 'hexrd.core.material.mksupport',
    'hexrd.spacegroup': 'hexrd.core.material.spacegroup',
    'hexrd.symbols': 'hexrd.core.material.symbols',
    'hexrd.symmetry': 'hexrd.core.material.symmetry',
    'hexrd.unitcell': 'hexrd.core.material.unitcell',
}

for alias, target in module_aliases.items():
    if Path(__file__).with_name(alias.rsplit('.', 1)[1] + '.py').exists():
        raise ImportError(f'"{alias}" is an alias path and should not exist')
    # Register with the existing alias importer without importing material,
    # SciPy, and Numba merely to import hexrd (or display CLI help).
    target_path = Path(__file__).parent.parent.joinpath(*target.split('.')).with_suffix('.py')
    module_map.module_map[alias] = (target, target_path)
    module_map.flattened_module_map[alias] = target


def __getattr__(name: str) -> types.ModuleType:
    # __getattr__ is only called if the attribute doesn't exist
    module = module_map.get("hexrd." + name)
    if module is not None:
        if isinstance(module, str):
            return importlib.import_module(module)

    raise AttributeError(f"Module `hexrd` has no attribute {name}")
