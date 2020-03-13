import numpy as np

from hexrd import imageseries
from hexrd.imageseries import stats

from .common import ImageSeriesTest, make_array, make_array_ims


class TestImageSeriesStats(ImageSeriesTest):


    def test_stats_median(self):
        """Processed imageseries: median"""
        a = make_array()
        is_a = imageseries.open(None, 'array', data=a)
        ismed = stats.median(is_a)
        amed = np.median(a, axis=0)
        err = np.linalg.norm(amed - ismed)
        self.assertAlmostEqual(err, 0., msg="stats.median failed")


    def test_stats_max(self):
        """Processed imageseries: median"""
        a = make_array()
        is_a = imageseries.open(None, 'array', data=a)
        ismax = stats.max(is_a)
        amax = np.max(a, axis=0)
        err = np.linalg.norm(amax - ismax)
        self.assertAlmostEqual(err, 0., msg="stats.max failed")


    def test_stats_min(self):
        """Processed imageseries: median"""
        a = make_array()
        is_a = imageseries.open(None, 'array', data=a)
        ismin = stats.min(is_a)
        amin = np.min(a, axis=0)
        err = np.linalg.norm(amin - ismin)
        self.assertAlmostEqual(err, 0., msg="stats.min failed")


    def test_stats_percentile(self):
        """Processed imageseries: median"""
        a = make_array()
        is_a = imageseries.open(None, 'array', data=a)
        isp90 = stats.percentile(is_a, 90)
        ap90 = np.percentile(a, 90, axis=0)
        err = np.linalg.norm(ap90 - isp90)
        self.assertAlmostEqual(err, 0., msg="stats.min failed")


    def test_stats_chunk(self):
        """Processed imageseries: median"""
        a = make_array()
        is_a = imageseries.open(None, 'array', data=a)
        amed = np.median(a, axis=0)

        # Run with 1 chunk
        img = np.zeros(is_a.shape)
        ismed1 = stats.median(is_a, chunk=(0, 1, img))
        err = np.linalg.norm(amed - ismed1)
        self.assertAlmostEqual(err, 0., msg="stats.median failed")

        # Run with 2 chunks
        img = np.zeros(is_a.shape)
        img = stats.median(is_a, chunk=(0, 2, img))
        ismed2 = stats.median(is_a, chunk=(1, 2, img))
        err = np.linalg.norm(amed - ismed2)
        self.assertAlmostEqual(err, 0., msg="stats.median failed")
