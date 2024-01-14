import os
import tempfile
import unittest

import numpy as np

from .common import ImageSeriesTest
from .common import make_array_ims, compare, compare_meta

from hexrd import imageseries


class ImageSeriesFormatTest(ImageSeriesTest):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.tmpdir)


class TestFormatH5(ImageSeriesFormatTest):

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
        is_h = imageseries.open(self.h5file, self.fmt, path=self.h5path)

        diff = compare(self.is_a, is_h)
        self.assertAlmostEqual(diff, 0., "h5 reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_h))

    def test_fmth5_nparray(self):
        """HDF5 format with numpy array metadata"""
        key = 'np-array'
        npa = np.array([0,2.0,1.3])
        self.is_a.metadata[key] = npa
        imageseries.write(self.is_a, self.h5file, self.fmt, path=self.h5path)
        is_h = imageseries.open(self.h5file, self.fmt, path=self.h5path)
        meta = is_h.metadata

        diff = np.linalg.norm(meta[key] - npa)
        self.assertAlmostEqual(diff, 0., "h5 numpy array metadata failed")

    def test_fmth5_nocompress(self):
        """HDF5 options: no compression"""
        imageseries.write(self.is_a, self.h5file, self.fmt,
                          path=self.h5path, gzip=0)
        is_h = imageseries.open(self.h5file, self.fmt, path=self.h5path)

        diff = compare(self.is_a, is_h)
        self.assertAlmostEqual(diff, 0., "h5 reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_h))

    def test_fmth5_compress_err(self):
        """HDF5 options: compression level out of range"""
        with self.assertRaises(ValueError):
            imageseries.write(self.is_a, self.h5file, self.fmt,
                              path=self.h5path, gzip=10)

    def test_fmth5_chunk(self):
        """HDF5 options: chunk size"""
        imageseries.write(self.is_a, self.h5file, self.fmt,
                          path=self.h5path, chunk_rows=0)
        is_h = imageseries.open(self.h5file, self.fmt, path=self.h5path)

        diff = compare(self.is_a, is_h)
        self.assertAlmostEqual(diff, 0., "h5 reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_h))


class TestFormatFrameCache(ImageSeriesFormatTest):

    def setUp(self):
        self.fcfile = os.path.join(self.tmpdir,  'frame-cache.npz')
        self.fmt = 'frame-cache'
        self.thresh = 0.5
        self.cache_file='frame-cache.npz'
        _, self.is_a = make_array_ims()

    def tearDown(self):
        os.remove(os.path.join(self.tmpdir, self.cache_file))

    def test_fmtfc(self):
        """save/load frame-cache format"""
        imageseries.write(self.is_a, self.fcfile, self.fmt,
            threshold=self.thresh, cache_file=self.cache_file)
        is_fc = imageseries.open(self.fcfile, self.fmt)
        diff = compare(self.is_a, is_fc)
        self.assertAlmostEqual(diff, 0., "frame-cache reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_fc))

    def test_fmtfc_nocache_file(self):
        """save/load frame-cache format with no cache_file arg"""
        imageseries.write(
            self.is_a, self.fcfile, self.fmt,
            threshold=self.thresh
        )
        is_fc = imageseries.open(self.fcfile, self.fmt)
        diff = compare(self.is_a, is_fc)
        self.assertAlmostEqual(diff, 0., "frame-cache reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_fc))

    def test_fmtfc_nparray(self):
        """frame-cache format with numpy array metadata"""
        key = 'np-array'
        npa = np.array([0,2.0,1.3])
        self.is_a.metadata[key] = npa

        imageseries.write(self.is_a, self.fcfile, self.fmt,
            threshold=self.thresh, cache_file=self.cache_file
        )
        is_fc = imageseries.open(self.fcfile, self.fmt)
        meta = is_fc.metadata
        diff = np.linalg.norm(meta[key] - npa)
        self.assertAlmostEqual(diff, 0.,
                               "frame-cache numpy array metadata failed")
