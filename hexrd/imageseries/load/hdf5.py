"""HDF5 adapter class
"""
import h5py

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator

class HDF5ImageSeriesAdapter(ImageSeriesAdapter):
    """collection of images in HDF5 format"""

    format = 'hdf5'

    def __init__(self, fname, **kwargs):
        """Constructor for H5FrameSeries

        *fname* - filename of the HDF5 file
        *kwargs* - keyword arguments, choices are:
           path - (required) path of dataset in HDF5 file
        """
        self.__h5name = fname
        self.__path = kwargs['path']
        self.__dataname = kwargs.pop('dataname', 'images')
        self.__images = '/'.join([self.__path, self.__dataname])
        self._meta = self._getmeta()

    def __getitem__(self, key):
        with self._dset as dset:
            return dset.__getitem__(key)

    def __iter__(self):
        return ImageSeriesIterator(self)

    #@memoize
    def __len__(self):
        with self._dset as dset:
            return len(dset)

    @property
    def _dgroup(self):
        # return a context manager to ensure proper file handling
        # always use like: "with self._dgroup as dgroup:"
        return H5ContextManager(self.__h5name, self.__path)

    @property
    def _dset(self):
        # return a context manager to ensure proper file handling
        # always use like: "with self._dset as dset:"
        return H5ContextManager(self.__h5name, self.__images)

    def _getmeta(self):
        mdict = {}
        with self._dgroup as dgroup:
            for k, v in list(dgroup.attrs.items()):
                mdict[k] = v

        return mdict

    @property
    #@memoize
    def metadata(self):
        """(read-only) Image sequence metadata

        note: metadata loaded on open and allowed to be modified
        """
        return self._meta

    @property
    def dtype(self):
        with self._dset as dset:
            return dset.dtype

    @property
    #@memoize so you only need to do this once
    def shape(self):
        with self._dset as dset:
            return dset.shape[1:]

    pass  # end class


class H5ContextManager:

    def __init__(self, fname, path):
        self._fname = fname
        self._path = path
        self._f = None

    def __enter__(self):
        self._f = h5py.File(self._fname, 'r')
        return self._f[self._path]

    def __exit__(self, *args):
        self._f.close()
