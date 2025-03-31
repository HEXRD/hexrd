from hexrd.core.utils.compatibility import h5py_read_string

import numpy as np


def unwrap_dict_to_h5(grp, d, asattr=False):
    """
    Unwraps a dictionary to an HDF5 file of the same structure.

    Parameters
    ----------
    grp : HDF5 group object
        The HDF5 group to recursively unwrap the dict into.
    d : dict
        Input dict (of dicts).
    asattr : bool, optional
        Flag to write end member in dictionary tree to an attribute. If False,
        if writes the object to a dataset using numpy.  The default is False.

    Returns
    -------
    None.

    """
    while len(d) > 0:
        key, item = d.popitem()
        if isinstance(item, dict):
            subgrp = grp.create_group(key)
            unwrap_dict_to_h5(subgrp, item, asattr=asattr)
        else:
            if asattr:
                try:
                    grp.attrs.create(key, item)
                except TypeError:
                    if item is None:
                        continue
                    else:
                        raise
            else:
                try:
                    grp.create_dataset(key, data=np.atleast_1d(item))
                except TypeError:
                    if item is None:
                        continue
                    else:
                        # probably a string badness

                        if isinstance(item, np.ndarray) and np.issubdtype(item.dtype, 'U'):
                            item = str(item) # hdf5 files do not support unicode arrays
                        grp.create_dataset(key, data=item)


def unwrap_h5_to_dict(f, d):
    """
    Unwraps a simple HDF5 file to a dictionary of the same structure.

    Parameters
    ----------
    f : HDF5 file (mode r)
        The input HDF5 file object.
    d : dict
        dictionary object to update.

    Returns
    -------
    None.

    Notes
    -----
    As written, ignores attributes and uses numpy to cast HDF5 datasets to
    dict entries.  Checks for 'O' type arrays and casts to strings; also
    converts single-element arrays to scalars.
    """
    for key, val in f.items():
        try:
            d[key] = {}
            unwrap_h5_to_dict(val, d[key])
        except AttributeError:
            # reached a dataset
            if np.dtype(val) == 'O':
                d[key] = h5py_read_string(val)
            else:
                tmp = np.array(val)
                if tmp.ndim == 1 and len(tmp) == 1:
                    d[key] = tmp[0]
                else:
                    d[key] = tmp
