import yaml
import numpy as np
from pathlib import Path


def _dict_path_by_id(d, value, path=()):
    if id(d) == value:
        return path
    elif isinstance(d, dict):
        for k, v in d.items():
            p = _dict_path_by_id(v, value, path + (k, ))
            if p is not None:
                return p
    elif isinstance(d, list):
        for i, v in enumerate(d):
            p = _dict_path_by_id(v, value, path + (str(i),))
            if p is not None:
                return p

    return None


class NumPyIncludeDumper(yaml.Dumper):
    """
    A yaml.Dumper implementation that will dump numpy.ndarray's. The arrays are
    saved using numpy.save(...) in path generate from the values path in the
    YAML document, relative to the location of the YAML document. For example

    "foo":
        "bar": ndarray

    The ndarray would be saved in foo/bar.npy.

    """
    def __init__(self, stream, **kwargs):
        super().__init__(stream, **kwargs)

        self._basedir = Path(stream.name).parent
        self._dct = None

    def ndarray_representer(self, data):
        path = _dict_path_by_id(self._dct, id(data))
        path = Path(*path)
        if path is None:
            raise ValueError("Unable to determine array path.")

        array_path = self._basedir / path.with_suffix('.npy')
        array_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(array_path, data)
        relative_array_path = array_path.relative_to(self._basedir)

        return self.represent_scalar('!include', str(relative_array_path))

    # We need intercept the dict so we can lookup the paths to ndarray's
    def represent(self, data):
        self._dct = data
        return super().represent(data)


NumPyIncludeDumper.add_representer(np.ndarray,
                                   NumPyIncludeDumper.ndarray_representer)
