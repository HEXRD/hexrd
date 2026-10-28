import numpy as np
import pytest
from scipy import ndimage
from skimage.measure import regionprops

from hexrd.core.instrument._spotutils import _weighted_centroids


@pytest.mark.parametrize('dtype', ['uint16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(1, 9, 13), (5, 13, 19), (8, 20, 20)])
def test_weighted_centroids_are_identical_to_regionprops(dtype, shape):
    rng = np.random.default_rng(0)
    for _ in range(10):
        data = (rng.random(shape) * 1000).astype(dtype)
        labels, count = ndimage.label(data > 850, np.ones((3, 3, 3)))
        expected = np.array(
            [region.centroid_weighted for region in regionprops(labels, data)]
        )
        actual = _weighted_centroids(labels, data, count)
        # Exact comparisons: a changed centroid can affect peak selection or
        # the finite differences used by the grain optimizer.
        np.testing.assert_array_equal(actual, expected)


def test_weighted_centroids_keep_label_order_and_bounding_box_offsets():
    data = np.zeros((7, 9, 13), dtype=float)
    data[1:3, 2:4, 3:6] = np.arange(12).reshape(2, 2, 3) + 1
    data[5, 7, 11] = 100
    labels, count = ndimage.label(data > 0, np.ones((3, 3, 3)))
    expected = np.array(
        [region.centroid_weighted for region in regionprops(labels, data)]
    )
    np.testing.assert_array_equal(_weighted_centroids(labels, data, count), expected)
