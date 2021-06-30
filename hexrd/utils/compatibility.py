from importlib.metadata import version

import h5py


def h5py_read_string(dataset):
    if version('h5py') >= '3':
        # In h5py >= 3.0.0, h5py no longer converts the data type to a
        # string automatically, and we have to do it manually...
        string_dtype = h5py.check_string_dtype(dataset.dtype)
        if string_dtype is not None and string_dtype.encoding == 'utf-8':
            dataset = dataset.asstr()

    return dataset[()]
