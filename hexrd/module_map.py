# The following dynamically generates aliases for the remapped modules based
# on the file_map
from collections import defaultdict
import importlib
import importlib.abc
import importlib.machinery
from pathlib import Path
import sys
from typing import Union


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


HEXRD_PACKAGE_PATH = Path(__file__).parent
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
    if old_path.suffix not in ("", ".py") or not "hexrd" in old_path.parts:
        continue
    old_module_path = path_to_module(old_path)
    # Default to pick the core module if it exists. Otherwise pick the first one.
    selected_path = new_paths[0]
    for new_path in new_paths:
        if 'core' in new_path.parts:
            selected_path = new_path
            break
    module_map[old_module_path] = (
        path_to_module(selected_path),
        HEXRD_PACKAGE_PATH.parent / selected_path,
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


flattened_module_map: dict[str, Union[ModuleAlias, str]] = {}

for key, (mapped_module, _mapped_fp) in module_map.items():
    parts = mapped_module.split(".")
    for i in range(len(parts) - 1):
        module = ".".join(parts[: i + 1])
        if module not in flattened_module_map:
            flattened_module_map[module] = ModuleAlias(parts[:i])
    flattened_module_map[key] = mapped_module

def get(alias: str) -> Union[ModuleAlias, str, None]:
    """
    Returns the the module or an alias to it if it exists.
    """
    if alias in flattened_module_map:
        return flattened_module_map[alias]
    return None


class ModuleSpecWithParent(importlib.machinery.ModuleSpec):
    def __init__(
        self, name, loader, *, origin=None, parent=None, is_package=False
    ):
        super().__init__(name, loader, origin=origin, is_package=is_package)
        self._parent = parent

    @property
    def parent(self):
        return self._parent


class ModuleAliasImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname in module_map:
            mapped_module, mapped_fp = module_map[fullname]
            # We only want to remap modules that go somewhere else.
            # If we are already trying to import something that exists, let
            # the other importers take care of it so we don't just loop forever.
            if fullname == mapped_module:
                return None

            return importlib.machinery.ModuleSpec(
                fullname,
                self,
                origin=mapped_fp.as_posix(),
                is_package=mapped_fp.name == "__init__.py",
            )
        return None

    def load_module(self, fullname):
        """
        This is a deprecated implementation path, but it is a lot easier to do override it this way
        than to override it with create and exec_module.
        """
        if fullname not in module_map:
            raise ImportError(f"Module {fullname} not found in module_map")

        mapped_module, _mapped_fp = module_map[fullname]
        base_mod = importlib.import_module(mapped_module)

        extra_candidates: list[str] = []
        for old_path, new_paths in file_map.items():
            if path_to_module(old_path) == fullname:
                for p in new_paths:
                    candidate = path_to_module(p)
                    if candidate != mapped_module:
                        extra_candidates.append(candidate)
                break

        if extra_candidates:
            for candidate in extra_candidates:
                try:
                    cand_mod = importlib.import_module(candidate)
                except Exception:
                    continue

                if hasattr(base_mod, "__path__") and hasattr(cand_mod, "__path__"):
                    try:
                        for p in list(cand_mod.__path__):
                            if p not in base_mod.__path__:
                                base_mod.__path__.append(p)
                    except Exception:
                        pass

                base_all = getattr(base_mod, "__all__", None)
                cand_all = getattr(cand_mod, "__all__", None)
                if cand_all:
                    if base_all is None:
                        base_mod.__all__ = list(cand_all)
                    else:
                        for name in cand_all:
                            if name not in base_all:
                                base_all.append(name)
                        base_mod.__all__ = base_all

                for name, val in cand_mod.__dict__.items():
                    if name in ("__name__", "__file__", "__package__", "__path__", "__loader__", "__spec__"):
                        continue
                    if name not in base_mod.__dict__:
                        base_mod.__dict__[name] = val

        sys.modules[fullname] = base_mod
        return sys.modules[fullname]


# We need to redirect __all__ attempts to import hexrd things into our own
# handler.
sys.meta_path.insert(0, ModuleAliasImporter())
