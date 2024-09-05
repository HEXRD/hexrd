"""Adapter class for frame caches
"""
import os
from threading import Lock

import numpy as np
from scipy.sparse import csr_matrix
import yaml

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator
from .metadata import yamlmeta

class FrameCacheImageSeriesAdapter(ImageSeriesAdapter):
    """collection of images in HDF5 format"""

    format = 'frame-cache'

    def __init__(self, fname, style='npz', **kwargs):
        """Constructor for frame cache image series

        *fname* - filename of the yml file
        *kwargs* - keyword arguments (none required)
        """
        self._fname = fname
        self._framelist = []
        self._framelist_was_loaded = False
        self._load_framelist_lock = Lock()

        if style.lower() in ('yml', 'yaml', 'test'):
            self._from_yml = True
            self._load_yml()
        else:
            self._from_yml = False
            self._load_cache()

    def _load_yml(self):
        with open(self._fname, "r") as f:
            d = yaml.load(f)
        datad = d['data']
        self._cache = datad['file']
        self._nframes = datad['nframes']
        self._shape = tuple(datad['shape'])
        self._dtype = np.dtype(datad['dtype'])
        self._meta = yamlmeta(d['meta'], path=self._cache)

    def _load_cache(self):
        arrs = np.load(self._fname)
        # HACK: while the loaded npz file has a getitem method
        # that mimicks a dict, it doesn't have a "pop" method.
        # must make an empty dict to pop after assignment of
        # class attributes so we can get to the metadata
        keysd = dict.fromkeys(list(arrs.keys()))
        self._nframes = int(arrs['nframes'])
        self._shape = tuple(arrs['shape'])
        # Check the type so we can read files written
        # using Python 2.7
        array_dtype = arrs['dtype'].dtype
        # Python 3
        if array_dtype.type == np.str_:
            dtype_str = str(arrs['dtype'])
        # Python 2.7
        else:
            dtype_str = arrs['dtype'].tobytes().decode()
        self._dtype = np.dtype(dtype_str)

        keysd.pop('nframes')
        keysd.pop('shape')
        keysd.pop('dtype')
        for i in range(self._nframes):
            keysd.pop(f"{i}_row")
            keysd.pop(f"{i}_col")
            keysd.pop(f"{i}_data")

        # all rmaining keys should be metadata
        for key in keysd:
            keysd[key] = arrs[key]
        self._meta = keysd

    def _load_framelist(self):
        """load into list of csr sparse matrices"""
        self._framelist = []
        if self._from_yml:
            bpath = os.path.dirname(self._fname)
            if os.path.isabs(self._cache):
                cachepath = self._cache
            else:
                cachepath = os.path.join(bpath, self._cache)
            arrs = np.load(cachepath)
        else:
            arrs = np.load(self._fname)

        for i in range(self._nframes):
            row = arrs[f"{i}_row"]
            col = arrs[f"{i}_col"]
            data = arrs[f"{i}_data"]
            frame = csr_matrix((data, (row, col)),
                               shape=self._shape,
                               dtype=self._dtype)
            self._framelist.append(frame)

    @property
    def metadata(self):
        """(read-only) Image sequence metadata
        """
        return self._meta

    def load_metadata(self, indict):
        """(read-only) Image sequence metadata

        Currently returns none
        """
        # TODO: Remove this. Currently not used;
        # saved temporarily for np.array trigger
        metad = {}
        for k, v in list(indict.items()):
            if v == '++np.array':
                newk = k + '-array'
                metad[k] = np.array(indict.pop(newk))
                metad.pop(newk, None)
            else:
                metad[k] = v
        return metad

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def _load_framelist_if_needed(self):
        if not self._framelist_was_loaded:
            # Only one thread should load the framelist.
            # Acquire the lock for loading the framelist.
            with self._load_framelist_lock:
                # It is possible that another thread already loaded
                # the framelist by the time this lock was acquired.
                # Check again.
                if not self._framelist_was_loaded:
                    self._load_framelist()
                    self._framelist_was_loaded = True

    def __getitem__(self, key):
        self._load_framelist_if_needed()
        return self._framelist[key].toarray()

    def __iter__(self):
        return ImageSeriesIterator(self)

    #@memoize
    def __len__(self):
        return self._nframes
