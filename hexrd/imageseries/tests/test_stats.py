import numpy as np

from hexrd import imageseries
from hexrd.imageseries import stats
from .common import ImageSeriesTest, make_array_ims


class TestImageSeriesStats(ImageSeriesTest):

    # These tests compare imageseries operations to numpy operations

    def test_stats_average(self):
        """imageseries.stats: average

        Compares with numpy average
        """
        a, is_a = make_array_ims()
        is_avg = stats.average(is_a)
        np_avg = np.average(a, axis=0).astype(np.float32)
        err = np.linalg.norm(np_avg - is_avg)
        self.assertAlmostEqual(err, 0., msg="stats.average failed")
        self.assertEqual(is_avg.dtype, np.float32)

    def test_stats_median(self):
        """imageseries.stats: median"""
        a, is_a = make_array_ims()
        ismed = stats.median(is_a)
        amed = np.median(a, axis=0)
        err = np.linalg.norm(amed - ismed)
        self.assertAlmostEqual(err, 0., msg="stats.median failed")
        self.assertEqual(ismed.dtype, np.float32)

    def test_stats_max(self):
        """imageseries.stats: max"""
        a, is_a = make_array_ims()
        ismax = stats.max(is_a)
        amax = np.max(a, axis=0)
        err = np.linalg.norm(amax - ismax)
        self.assertAlmostEqual(err, 0., msg="stats.max failed")
        self.assertEqual(ismax.dtype, is_a.dtype)


    def test_stats_min(self):
        """imageseries.stats: min"""
        a, is_a = make_array_ims()
        ismin = stats.min(is_a)
        amin = np.min(a, axis=0)
        err = np.linalg.norm(amin - ismin)
        self.assertAlmostEqual(err, 0., msg="stats.min failed")
        self.assertEqual(ismin.dtype, is_a.dtype)


    def test_stats_percentile(self):
        """imageseries.stats: percentile"""
        a, is_a = make_array_ims()
        isp90 = stats.percentile(is_a, 90)
        ap90 = np.percentile(a, 90, axis=0).astype(np.float32)
        err = np.linalg.norm(ap90 - isp90)
        self.assertAlmostEqual(err, 0., msg="stats.percentile failed")
        self.assertEqual(isp90.dtype, np.float32)

    # These tests compare chunked operations (iterators) to non-chunked ops

    def test_stats_average_chunked(self):
        """imageseries.stats: chunked average"""
        a, is_a = make_array_ims()
        a_avg = stats.average(a)

        # Run with 1 chunk
        img = np.zeros(is_a.shape)
        for ismed1 in stats.average_iter(is_a, 1):
            pass
        err = np.linalg.norm(a_avg - ismed1)
        self.assertAlmostEqual(err, 0., msg="stats.average failed (1 chunk)")

        # Run with 2 chunks
        for ismed2 in stats.average_iter(is_a, 2):
            pass
        err = np.linalg.norm(a_avg - ismed2)
        self.assertAlmostEqual(err, 0., msg="stats.average failed")

    def test_stats_median_chunked(self):
        """imageseries.stats: chunked median"""
        a, is_a = make_array_ims()
        a_med = stats.median(is_a)

        # Run with 1 chunk
        img = np.zeros(is_a.shape)
        for ismed1 in stats.median_iter(is_a, 1):
            pass
        err = np.linalg.norm(a_med - ismed1)
        self.assertAlmostEqual(err, 0., msg="stats.average failed (1 chunk)")

        # Run with 2 chunks
        for ismed2 in stats.median_iter(is_a, 2):
            pass
        err = np.linalg.norm(a_med - ismed2)
        self.assertAlmostEqual(err, 0., msg="stats.average failed (2 chunks)")

        # Run with 3 chunks, with buffer
        for ismed3 in stats.median_iter(is_a, 3, use_buffer=True):
            pass
        err = np.linalg.norm(a_med - ismed3)
        self.assertAlmostEqual(
            err, 0., msg="stats.average failed (3 chunks, buffer)"
        )

        # Run with 3 chunks, no buffer
        for ismed3 in stats.median_iter(is_a, 3, use_buffer=False):
            pass
        err = np.linalg.norm(a_med - ismed3)
        self.assertAlmostEqual(
            err, 0., msg="stats.average failed (3 chunks, no buffer)"
        )
