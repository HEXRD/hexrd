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
        self.assertAlmostEqual(err, 0., msg="median image failed")

    def test_stats_max(self):
        """Processed imageseries: median"""
        a = make_array()
        is_a = imageseries.open(None, 'array', data=a)
        ismax = stats.max(is_a)
        amax = np.max(a, axis=0)
        err = np.linalg.norm(amax - ismax)
        self.assertAlmostEqual(err, 0., msg="max image failed")
