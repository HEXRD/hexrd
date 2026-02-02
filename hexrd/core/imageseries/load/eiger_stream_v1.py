"""HDF5 adapter class"""

import warnings

from dectris.compression import decompress
import h5py
import numpy as np

from hexrd.core.utils.hdf5 import unwrap_h5_to_dict

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator


class EigerStreamV1ImageSeriesAdapter(ImageSeriesAdapter):

    format = 'eiger-stream-v1'

    def __init__(self, fname, **kwargs):
        if isinstance(fname, h5py.File):
            self._h5name: str = fname.filename
            self._h5file: h5py.File = fname
        else:
            self._h5name = fname
            self._h5file = h5py.File(self._h5name, 'r')

        self._load_metadata()

    def close(self):
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def __del__(self):
        # !!! Note this is not ideal, as the use of __del__ is problematic.
        #     However, it is highly unlikely that the usage of a ImageSeries
        #     would pose a problem.  A warning will (hopefully) be emitted if
        #     an issue arises at some point
        try:
            self.close()
        except Exception:
            msg = "EigerStreamV1ImageSeriesAdapter could not close h5 file"
            warnings.warn(msg)

    def __getitem__(self, key):
        if isinstance(key, int):
            idx = key
            rest = None
        else:
            idx = key[0]
            rest = key[1:]

        entry = self._data_group[str(idx)]
        d = {}
        unwrap_h5_to_dict(entry, d)
        data = _decompress_frame(d)
        if rest is None:
            return data
        else:
            return data[rest]

    def __iter__(self):
        return ImageSeriesIterator(self)

    def __len__(self):
        return len(self._data_group)

    def __getstate__(self):
        # Remove any non-pickleable attributes. Prefix them with the private prefix
        to_remove = [f'_{self.__class__.__name__}_h5file']

        state = self.__dict__.copy()
        for attr in to_remove:
            state.pop(attr)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5file = h5py.File(self._h5name, 'r')
        self._load_metadata()

    def _load_metadata(self):
        self.__meta = self._get_metadata()

    def _get_metadata(self):
        d = {}
        unwrap_h5_to_dict(self._h5file['/metadata'], d)
        return d

    @property
    def metadata(self):
        """(read-only) Image sequence metadata

        note: metadata loaded on open and allowed to be modified
        """
        return self.__meta

    @property
    def _data_group(self):
        return self._h5file['data']

    @property
    def _first_data_entry(self):
        return self._data_group['0']

    @property
    def dtype(self):
        return np.dtype(self._first_data_entry['dtype'][()])

    @property
    def shape(self):
        return tuple(self._first_data_entry['shape'][()])


def _decompress_frame(d: dict) -> np.ndarray:
    compression_type = d['compression_type']
    dtype = d['dtype']
    shape = d['shape']
    data = d['data']
    elem_size = d['elem_size']

    if compression_type is None:
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    decompressed_bytes = decompress(data, compression_type, elem_size=elem_size)
    return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
