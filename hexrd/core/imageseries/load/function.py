"""Adapter class for a custom function that takes an int as an argument
and returns a 2D numpy array.
"""

import numpy as np

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator


class FunctionImageSeriesAdapter(ImageSeriesAdapter):
    """A highly customizable "generator-like" imageseries adapter

    This adapter allows a function to be provided which should
    generate and return a frame for the index requested. It allows
    any function to be used that can provide a frame.

    Note: to match behavior with other imageseries adapters, it may
    be advisable to return a new copy each time a frame is requested
    (rather than returning a locally stored/cached copy), since other
    parts of the code may modify the frame in place.

    Parameters
    ----------
    fname: None
       should be None
    func: a function that returns a numpy array for a index
    num_frames: the number of frames provided by the function
    metadata: dict (optional)
       the metadata dictionary
    """

    format = 'function'

    def __init__(self, fname, **kwargs):
        self._func = kwargs['func']
        self._nframes = kwargs['num_frames']

        self._meta = kwargs.pop('meta', {})

        first_frame = self._first_frame
        self._nxny = first_frame.shape
        self._dtype = first_frame.dtype

    @property
    def metadata(self):
        """Image sequence metadata"""
        return self._meta

    @property
    def _first_frame(self):
        return self._func(0)

    @property
    def shape(self):
        return self._nxny

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value: np.dtype):
        self._dtype = value

    def __getitem__(self, key):
        if not isinstance(key, int):
            msg = f'Only int keys are supported, but "{key}" was provided'
            raise Exception(msg)

        return self._func(key)

    def __iter__(self):
        return ImageSeriesIterator(self)

    def __len__(self):
        return self._nframes
