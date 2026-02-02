from importlib.metadata import version

import numpy as np
from numpy.typing import NDArray

import h5py


def h5py_read_string(dataset: h5py.Dataset) -> NDArray[np.str_]:
    """
    Always return a numpy array of unicode strings (np.str_),
    independent of h5py version or dataset dtype.
    """
    arr = dataset[()]
    arr = np.asarray(arr)

    if arr.dtype.kind == "S":
        # Bytes type
        arr = arr.astype(np.str_)

    return arr
