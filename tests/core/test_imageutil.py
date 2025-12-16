import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from hexrd.core import imageutil

# --- Helper Functions ---


def create_peak_data(
    size=100, peak_pos=50, width=2, height=100, background=10
):
    """
    Generate 1D array with a Gaussian peak.
    Width (sigma) set to 2 so w=4 effectively clips it.
    """
    x = np.arange(size)
    peak = height * np.exp(-0.5 * ((x - peak_pos) / width) ** 2)
    return peak + background


def create_peak_data_2d(
    shape=(50, 50), center=(25, 25), width=1.5, height=100, background=10
):
    y, x = np.indices(shape)
    r2 = (x - center[1]) ** 2 + (y - center[0]) ** 2
    peak = height * np.exp(-0.5 * r2 / width**2)
    return peak + background


# --- SNIP 1D Tests ---


def test_scale_image_snip():
    data = np.array([10.0, 100.0])
    offset = 0.0

    scaled = imageutil._scale_image_snip(data, offset, invert=False)
    assert scaled[0] > 0
    assert scaled[1] > scaled[0]

    inverted = imageutil._scale_image_snip(scaled, offset, invert=True)
    np.testing.assert_allclose(inverted, data)


def test_fast_snip1d():
    y = create_peak_data(height=50, background=10)
    y[0] = 5

    bkg = imageutil.fast_snip1d(y[None, :], w=4, numiter=2)

    assert bkg[0, 50] < 40
    assert 8 < bkg[0, 10] < 12


def test_snip1d_basic():
    y = create_peak_data()
    y_2d = np.vstack([y, y])
    bkg = imageutil.snip1d(y_2d, w=4, numiter=2, max_workers=1)

    assert bkg.shape == (2, 100)
    assert bkg[0, 50] < y[50]
    assert np.isclose(bkg[0, 10], 10, atol=2)


def test_snip1d_threshold_masking():
    y = create_peak_data(background=10)
    y[0:5] = 1.0

    y_2d = y[None, :]
    bkg = imageutil.snip1d(y_2d, w=4, numiter=2, threshold=5.0, max_workers=1)

    assert np.all(bkg[0, 0:5] == 5.0)


def test_snip1d_parallel():
    y = create_peak_data()
    y_2d = np.vstack([y, y, y])

    bkg = imageutil.snip1d(y_2d, w=4, numiter=1, max_workers=2)
    assert bkg.shape == (3, 100)


def test_snip1d_quad():
    y = create_peak_data(width=2)
    bkg = imageutil.snip1d_quad(y, w=6, numiter=4)

    assert bkg[50] < 100


# --- SNIP 2D Tests ---


def test_snip2d_linear():
    img = create_peak_data_2d()
    bkg = imageutil.snip2d(img, w=4, numiter=2, order=1)

    assert bkg.shape == img.shape
    assert bkg[25, 25] < 80
    assert bkg[0, 0] > 5


def test_snip2d_quadratic():
    img = create_peak_data_2d(width=1.5)
    bkg = imageutil.snip2d(img, w=6, numiter=4, order=2)

    assert bkg[25, 25] < 80


# --- Feature Detection Tests ---


def test_find_peaks_label():
    img = np.zeros((20, 20))
    img[5, 5] = 100
    img[15, 15] = 100

    kwargs = {'threshold': 50, 'filter_radius': 0}

    nspots, coms = imageutil.find_peaks_2d(img, 'label', kwargs)

    assert nspots == 2
    assert np.any(np.isclose(coms, [5, 5]).all(axis=1))
    assert np.any(np.isclose(coms, [15, 15]).all(axis=1))


def test_find_peaks_label_filtered():
    img = np.zeros((20, 20))
    img[10, 10] = 100

    with pytest.warns(DeprecationWarning):
        kwargs = {'threshold': 0.1, 'filter_radius': 1}
        nspots, _ = imageutil.find_peaks_2d(img, 'label', kwargs)
        assert nspots >= 1


@patch('hexrd.core.imageutil.blob_log')
def test_find_peaks_blob_log(mock_blob):
    img = np.zeros((10, 10))
    mock_blob.return_value = np.array([[5, 5, 1]])

    kwargs = {'min_sigma': 1, 'max_sigma': 2}
    nspots, coms = imageutil.find_peaks_2d(img, 'blob_log', kwargs)

    assert nspots == 1
    assert np.allclose(coms[0], [5, 5])
    mock_blob.assert_called()


@patch('hexrd.core.imageutil.blob_dog')
def test_find_peaks_blob_dog(mock_blob):
    img = np.zeros((10, 10))
    mock_blob.return_value = np.array([[2, 2, 1]])

    kwargs = {'min_sigma': 1}
    nspots, coms = imageutil.find_peaks_2d(img, 'blob_dog', kwargs)

    assert nspots == 1
    assert np.allclose(coms[0], [2, 2])
    mock_blob.assert_called()


# --- Edge Case Tests ---


def test_run_snip1d_row_all_masked():
    task = (0, (np.zeros(10), np.ones(10, dtype=bool)))
    idx, res = imageutil._run_snip1d_row(task, numiter=1, w=1, min_val=0)
    assert idx == 0
    assert np.isnan(res)
