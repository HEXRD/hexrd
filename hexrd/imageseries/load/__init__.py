import abc
import pkgutil
import numpy as np

from ..imageseriesabc import ImageSeriesABC
from .registry import Registry

RegionType = tuple[tuple[int, int], tuple[int, int]]

# Metaclass for adapter registry

class _RegisterAdapterClass(abc.ABCMeta):

    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        Registry.register(cls)

class ImageSeriesAdapter(ImageSeriesABC, metaclass=_RegisterAdapterClass):

    format = None

    def get_region(self, frame_idx: int, region: RegionType) -> np.ndarray:
        r = region
        return self[frame_idx][r[0][0]:r[0][1], r[1][0]:r[1][1]]

    def __getitem__(self, _):
        pass

# import all adapter modules

from . import (
    array, framecache, function, hdf5, imagefiles, rawimage, metadata, trivial
)

try:
    from dectris.compression import decompress
except ImportError:
    # Dectris compression is not available. Skip the eiger_stream_v1
    pass
else:
    # Eiger stream v1 is supported
    from . import eiger_stream_v1


#for loader, name, ispkg in pkgutil.iter_modules(__path__):
#    if name is not 'registry':
#        __import__(name, globals=globals())
    #
    # couldn't get the following line to work due to relative import issue:
    #     loader.find_module(name).load_module(name)
