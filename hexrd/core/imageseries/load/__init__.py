import abc
import pkgutil
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray

from ..imageseriesabc import ImageSeriesABC, RegionType
from .registry import Registry

# Metaclass for adapter registry


class _RegisterAdapterClass(abc.ABCMeta):
    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        Registry.register(cls)


class ImageSeriesAdapter(ImageSeriesABC, metaclass=_RegisterAdapterClass):
    format: str | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dtype: Optional[np.dtype] = None

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value: Optional[np.dtype]):
        self._dtype = value

    def get_region(self, frame_idx: int, region: RegionType) -> np.ndarray:
        r = region
        return self[frame_idx][r[0][0] : r[0][1], r[1][0] : r[1][1]]

    @property
    def metadata(self) -> dict[str, NDArray[np.float64]]:
        return {}

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError()

    def option_values(self) -> dict:
        raise NotImplementedError()

    def set_option(self, key: str, value: Any):
        raise NotImplementedError()

    def __getitem__(self, _):
        pass


# import all adapter modules

from . import (
    array,
    framecache,
    function,
    hdf5,
    imagefiles,
    rawimage,
    metadata,
    trivial,
)

try:
    from dectris.compression import decompress
except ImportError:
    # Dectris compression is not available. Skip the eiger_stream_v1
    pass
else:
    # Eiger stream formats are supported
    from . import eiger_stream_v1
    from . import eiger_stream_v2


# for loader, name, ispkg in pkgutil.iter_modules(__path__):
#    if name is not 'registry':
#        __import__(name, globals=globals())
#
# couldn't get the following line to work due to relative import issue:
#     loader.find_module(name).load_module(name)
