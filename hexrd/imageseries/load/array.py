"""Adapter class for numpy array (3D)
"""
from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator

import numpy as np


class ArrayImageSeriesAdapter(ImageSeriesAdapter):
    """Collection of Images in numpy array

    Parameters
    ----------
    fname: None
       should be None
    data: array (n, m, l)
       3-dimensional data array
    metadata: dict (optional)
       the metadata dictionary
    """
    format = 'array'

    def __init__(self, fname, **kwargs):
        data_arr = np.array(kwargs['data'])
        if data_arr.ndim < 3:
            self._data = np.tile(data_arr, (1, 1, 1))
        elif data_arr.ndim == 3:
            self._data = data_arr
        else:
            raise RuntimeError(
                    'input array must be 2-d or 3-d; you provided ndim=%d'
                    % data_arr.ndim
                )

        self._meta = kwargs.pop('meta', dict())
        self._shape = self._data.shape
        self._nframes = self._shape[0]
        self._nxny = self._shape[1:3]

    @property
    def metadata(self):
        """Image sequence metadata"""
        return self._meta

    @property
    def shape(self):
        return self._nxny

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, key):
        return self._data[key].copy()

    def __iter__(self):
        return ImageSeriesIterator(self)

    def __len__(self):
        return self._nframes
