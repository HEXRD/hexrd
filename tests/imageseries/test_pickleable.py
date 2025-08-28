import os
import pickle
import tempfile
import unittest

from .common import make_array_ims
from hexrd.core.imageseries.load.hdf5 import HDF5ImageSeriesAdapter
from hexrd.core.imageseries.load.framecache import FrameCacheImageSeriesAdapter
from hexrd.core import imageseries


class ImageSeriesPickleableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.tmpdir)


class TestHDF5SeriesAdapter(ImageSeriesPickleableTest):
    def setUp(self):
        self.h5file = os.path.join(self.tmpdir, 'test_ims.h5')
        self.h5path = 'array-data'
        self.fmt = 'hdf5'
        _, self.is_a = make_array_ims()

    def tearDown(self):
        os.remove(self.h5file)

    def test_fmth5(self):
        """save/load HDF5 format"""
        imageseries.write(self.is_a, self.h5file, self.fmt, path=self.h5path)
        adapter = HDF5ImageSeriesAdapter(self.h5file, path=self.h5path)
        # will throw if adapter is not pickleable
        pickle.dumps(adapter)


class TestFormatFrameCacheNPZSeriesAdapter(ImageSeriesPickleableTest):
    def setUp(self):
        self.fcfile = os.path.join(self.tmpdir, 'frame-cache.npz')
        self.fmt = 'frame-cache'
        self.style = 'npz'
        self.thresh = 0.5
        self.cache_file = 'frame-cache.npz'
        _, self.is_a = make_array_ims()

    def tearDown(self):
        os.remove(self.fcfile)

    def test_npz(self):
        imageseries.write(
            self.is_a,
            self.fcfile,
            self.fmt,
            style=self.style,
            threshold=self.thresh,
            cache_file=self.cache_file,
        )
        adapter = FrameCacheImageSeriesAdapter(self.fcfile, style=self.style)
        # will throw if adapter is not pickleable
        pickle.dumps(adapter)


class TestFormatFrameCacheFCH5SeriesAdapter(ImageSeriesPickleableTest):
    def setUp(self):
        self.fcfile = os.path.join(self.tmpdir, 'frame-cache.fch5')
        self.fmt = 'frame-cache'
        self.style = 'fch5'
        self.thresh = 0.5
        self.cache_file = 'frame-cache.fch5'
        _, self.is_a = make_array_ims()

    def tearDown(self):
        os.remove(self.fcfile)

    def test_fmth5(self):
        imageseries.write(
            self.is_a,
            self.fcfile,
            self.fmt,
            style=self.style,
            threshold=self.thresh,
            cache_file=self.cache_file,
        )
        adapter = FrameCacheImageSeriesAdapter(self.fcfile, style=self.style)
        # will throw if adapter is not pickleable
        pickle.dumps(adapter)
