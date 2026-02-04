from importlib.metadata import version

import numpy as np
from numpy.typing import NDArray

import h5py


def h5py_read_string(dataset: h5py.Dataset) -> NDArray[np.str_]:
    if version('h5py') >= '3':
        # In h5py >= 3.0.0, h5py no longer converts the data type to a
        # string automatically, and we have to do it manually...
        string_dtype: h5py.string_dtype = h5py.h5t.check_string_dtype(dataset.dtype)
        if string_dtype is not None and string_dtype.encoding == 'utf-8':
            dataset = dataset.asstr()

    h5_data = dataset[()]
    if isinstance(h5_data, (bytes, np.bytes_)):
        h5_data = h5_data.decode('utf-8')

    return h5_data
