# The following dynamically generates aliases for the remapped modules based
# on the file_map
import pickle
import importlib
import importlib.util
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from collections import defaultdict


def path_to_module(path: Path) -> str:
    """
    Convert a path to a module name.


    e.g.
    * "package_remapper/remapper.py" -> "package_remapper.remapper"
    * "package_remapper/__init__.py" -> "package_remapper"

    """
    if path.suffix not in (".py", ""):
        raise ValueError(f"Expected a .py file, got {path}")

    path = path.with_suffix("")
    if path.parts[-1] == "__init__":
        path = path.parent
    return path.as_posix().replace("/", ".")


HEXRD_PACKAGE_PATH = Path(__file__).parent.parent
file_map: dict[Path, list[Path]] = defaultdict(list)
with open(HEXRD_PACKAGE_PATH / "file_table.tsv", "r") as f:
    for line in f:
        if not line.strip():
            continue
        kv = line.strip().split()
        if len(kv) != 2:
            continue
        k, v = line.strip().split()
        file_map[Path(k)].append(Path(v))

module_map: dict[str, tuple[str, Path]] = {}

for old_path, new_paths in file_map.items():
    if old_path.suffix not in  ("", ".py") or not "hexrd" in old_path.parts:
        continue
    old_module_path = path_to_module(old_path)
    # TODO: This just picks one. We should probably pick the right one? We should know the right one after
    # We finish the refactor.
    module_map[old_module_path] = (
        path_to_module(new_paths[0]),
        HEXRD_PACKAGE_PATH / new_paths[0],
    )


class ModuleAlias:
    def __init__(self, current_path: list[str]):
        self.current_path = current_path

    def __getattr__(self, name):
        full_path = self.current_path + [name]
        full_name = ".".join(full_path)
        if full_name in module_map:
            module, _fp = module_map[full_name]
            if isinstance(module, ModuleAlias):
                return module
            else:
                return importlib.import_module(module)
        current_module = ".".join(self.current_path)
        raise AttributeError(
            f"Module `{current_module}` has no attribute {name}"
        )


flattened_module_map: dict[str, ModuleAlias | str] = {}

for key, (mapped_module, _mapped_fp) in module_map.items():
    parts = mapped_module.split(".")
    for i in range(len(parts) - 1):
        module = ".".join(parts[: i + 1])
        if module not in flattened_module_map:
            flattened_module_map[module] = ModuleAlias(parts[:i])
    flattened_module_map[key] = mapped_module


def get(alias: str) -> ModuleAlias | str | None:
    """
    Returns the the module or an alias to it if it exists.
    """
    if alias in flattened_module_map:
        return flattened_module_map[alias]
    return None



class ModuleSpecWithParent(importlib.machinery.ModuleSpec):
    def __init__(self, name, loader, *, origin=None, parent=None, is_package=False):
        super().__init__(name, loader, origin=origin, is_package=is_package)
        self._parent = parent

    @property
    def parent(self):
        return self._parent
class ModuleAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in module_map:
            mapped_module, mapped_fp = module_map[fullname]

            if mapped_fp.name != "__init__.py":
                parent = mapped_module.rsplit(".", 1)[0]
            else:
                parent = mapped_module

            
            # Need to set these to be the exact same module so that class comparison
            # works correctly if you are comparing classes that are imported one way with classes
            # that are imported the mapped way.
            sys.modules[fullname] = importlib.import_module(mapped_module)

            # We have to totally change the structure of the package, so we need a custom submodule for ModuleSpec
            # ModuleSpec.parent is used for relative imports.
            if mapped_fp.is_file():
                spec = ModuleSpecWithParent(
                    mapped_module,
                    importlib.machinery.SourceFileLoader(
                        mapped_module, mapped_fp.as_posix()
                    ),
                    origin=mapped_fp.as_posix(),
                    parent=parent,
                    is_package=mapped_fp.name == "__init__.py",
                )
                # Need to set this, since ModuleSpec doesn't by defualt.
                # This tells importlib to set __file__, which is used by a few things in here.
                spec.has_location = True
            else:
                spec = ModuleSpecWithParent(
                    mapped_module,
                    importlib.machinery.NamespaceLoader(
                        mapped_module, 
                        list(mapped_fp.parts), 
                        path_finder=importlib.machinery.PathFinder.find_spec, # type: ignore
                    ),
                    parent=parent,
                    is_package=True
                )
            return spec
        return None


sys.meta_path.append(ModuleAliasFinder())
