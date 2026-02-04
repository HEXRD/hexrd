"""HDF5 adapter class
"""

from typing import Any
import warnings

from dectris.compression import decompress
import h5py
import numpy as np

from hexrd.core.utils.hdf5 import unwrap_h5_to_dict

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator


class EigerStreamV2ImageSeriesAdapter(ImageSeriesAdapter):
    format = 'eiger-stream-v2'

    def __init__(self, fname, **kwargs):
        if isinstance(fname, h5py.File):
            self.__h5name = fname.filename
            self.__h5file = fname
        else:
            self.__h5name = fname
            self.__h5file = h5py.File(self.__h5name, 'r')

        self.threshold_setting = kwargs.pop('threshold_setting', 'threshold_1')
        self.multiplier = kwargs.pop('multiplier', 1)

        self.__data_group_path = '/data'
        self._load_metadata()

    @property
    def threshold_setting(self) -> str:
        """The three possible threshold settings are:

        * threshold_1
        * threshold_2
        * man_diff

        For man_diff, the multiplier is utilitzed in the expression:

            threshold_1 - multiplier * threshold_2
        """
        return self._threshold_setting

    @threshold_setting.setter
    def threshold_setting(self, v: str):
        """The three possible threshold settings are:

        * threshold_1
        * threshold_2
        * man_diff

        For man_diff, the multiplier is utilitzed in the expression:

            threshold_1 - multiplier * threshold_2
        """
        possible_values = [
            'threshold_1',
            'threshold_2',
            'man_diff',
        ]

        if v not in possible_values:
            poss_str = ', '.join(possible_values)
            msg = (
                f'"{v}" is not one of the possible values for the '
                f'threshold_setting. Possible values are: {poss_str}'
            )
            raise NotImplementedError(msg)

        self._threshold_setting = v

    @property
    def multiplier(self) -> float:
        """The multiplier is only used if the threshold_setting is man_diff

        In that case, the following equation is used:

            threshold_1 - multiplier * threshold_2

        The multiplier will be automatically converted to a float.
        """
        return self._multiplier

    @multiplier.setter
    def multiplier(self, v: float | int):
        """The multiplier is only used if the threshold_setting is man_diff

        In that case, the following equation is used:

            threshold_1 - multiplier * threshold_2

        The multiplier will be automatically converted to a float.
        """
        self._multiplier = float(v)

    def close(self):
        if self.__h5file is not None:
            self.__h5file.close()
            self.__h5file = None

    def __del__(self):
        # !!! Note this is not ideal, as the use of __del__ is problematic.
        #     However, it is highly unlikely that the usage of a ImageSeries
        #     would pose a problem.  A warning will (hopefully) be emitted if
        #     an issue arises at some point
        try:
            self.close()
        except Exception:
            msg = "EigerStreamV2ImageSeriesAdapter could not close h5 file"
            warnings.warn(msg)

    def __getitem__(self, key):
        if isinstance(key, int):
            idx = key
            rest = None
        else:
            idx = key[0]
            rest = key[1:]

        if self.threshold_setting == 'man_diff':
            threshold_1_data = self._load_frame(idx, 'threshold_1')
            threshold_2_data = self._load_frame(idx, 'threshold_2')
            data = threshold_1_data - self.multiplier * threshold_2_data
        else:
            data = self._load_frame(idx, self.threshold_setting)

        if rest is None:
            return data
        else:
            return data[rest]

    def __iter__(self):
        return ImageSeriesIterator(self)

    def __len__(self):
        return len(self._data_group)

    def __getstate__(self):
        # Remove any non-pickleable attributes
        to_remove = [
            '__h5file',
        ]

        # Prefix them with the private prefix
        prefix = f'_{self.__class__.__name__}'
        to_remove = [f'{prefix}{x}' for x in to_remove]

        # Make a copy of the dict to modify
        state = self.__dict__.copy()

        # Remove them
        for attr in to_remove:
            state.pop(attr)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__h5file = h5py.File(self.__h5name, 'r')
        self._load_metadata()

    def _load_metadata(self):
        self.__meta = self._get_metadata()

    def _get_metadata(self):
        d = {}
        unwrap_h5_to_dict(self.__h5file['/metadata'], d)
        return d

    @property
    def metadata(self):
        """(read-only) Image sequence metadata

        note: metadata loaded on open and allowed to be modified
        """
        return self.__meta

    @property
    def _data_group(self):
        return self.__h5file[self.__data_group_path]

    @property
    def _first_threshold_1_entry(self):
        return self._data_group[f'0/threshold_1']

    @property
    def dtype(self) -> np.dtype:
        if self.threshold_setting == 'man_diff':
            return np.dtype('float64')

        return np.dtype(self._first_threshold_1_entry['dtype'][()])

    @dtype.setter
    def dtype(self, value: np.dtype):
        self._dtype = value

    @property
    def shape(self):
        return tuple(self._first_threshold_1_entry['shape'][()])

    def _load_frame(self, idx: int, group_name: str) -> np.ndarray:
        entry = self._data_group[f'{idx}/{group_name}']

        d = {}
        unwrap_h5_to_dict(entry, d)
        return _decompress_frame(d)

    def set_option(self, key: str, value: Any):
        possible_options = [
            'threshold_setting',
            'multiplier',
        ]

        if key not in possible_options:
            poss_str = ', '.join(possible_options)
            msg = f'"{key}" not in possible options: {poss_str}'
            raise NotImplementedError(msg)

        setattr(self, key, value)

    def option_values(self) -> dict:
        options = [
            'threshold_setting',
            'multiplier',
        ]
        return {k: getattr(self, k) for k in options}


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
