import yaml
from pathlib import Path
import numpy as np


class NumPyIncludeLoader(yaml.SafeLoader):
    """
    A yaml.Loader implemenation that allows !include <numpy_file_path>. This
    allows the loading of npy files into the YAML document.
    """

    def __init__(self, stream):
        self._basedir = Path(stream.name).parent

        super(NumPyIncludeLoader, self).__init__(stream)

    def include(self, node):
        file_path = self._basedir / self.construct_scalar(node)

        a = np.load(file_path)

        return a


NumPyIncludeLoader.add_constructor('!include', NumPyIncludeLoader.include)
