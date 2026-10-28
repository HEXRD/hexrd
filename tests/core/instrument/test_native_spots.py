import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse import csr_array

from hexrd.core.extensions import transforms as native
from hexrd.core.instrument._spotutils import _weighted_centroids


@pytest.mark.parametrize('dtype', ['uint16', 'int32', 'float32', 'float64'])
def test_native_spot_statistics_are_exact(dtype):
    rng = np.random.default_rng(42)
    shapes = np.array([(1, 9, 13), (5, 13, 19), (8, 20, 20), (3, 7, 5)], dtype=np.int64)
    patches = [(rng.random(tuple(shape)) * 1000).astype(dtype) for shape in shapes]
    patches[0][:] = 0
    offsets = np.r_[0, np.cumsum([p.size for p in patches])]
    centers, sums, maxima, valid = native._spotStats(
        np.concatenate([p.ravel() for p in patches]), offsets, shapes, 850.0
    )
    for i, patch in enumerate(patches):
        labels, count = ndimage.label(patch > 850, np.ones((3, 3, 3)))
        assert valid[i] == bool(count)
        if not count:
            assert np.all(np.isnan(centers[i]))
            continue
        coms = _weighted_centroids(labels, patch, count)
        closest = np.argmin(np.sum((coms - np.array(patch.shape) / 2) ** 2, axis=1))
        selected = patch[labels == closest + 1]
        np.testing.assert_array_equal(centers[i], coms[closest])
        np.testing.assert_equal(sums[i], selected.sum())
        np.testing.assert_equal(maxima[i], selected.max())
        assert sums.dtype == selected.sum().dtype
        assert maxima.dtype == selected.dtype


@pytest.mark.parametrize('dtype', ['uint16', 'int32', 'float32', 'float64'])
def test_native_sparse_gather_and_roi_views(dtype):
    rng = np.random.default_rng(42)
    dense = rng.integers(0, 100, (4, 20, 30)).astype(dtype)
    dense[dense < 80] = 0
    frames = native._SparseFrames([csr_array(frame) for frame in dense])
    ids = np.array([[0, 2], [1, 3]])
    offsets = np.array([0, 5, 12])
    rows = rng.integers(-3, 10, 12)
    cols = rng.integers(-3, 12, 12)
    actual = frames.view(4, 5, 7, 9).gather(ids, offsets, rows, cols)
    expected = []
    for i in range(2):
        slc = slice(offsets[i], offsets[i + 1])
        rr, cc = rows[slc].clip(0, 6) + 4, cols[slc].clip(0, 8) + 5
        expected.extend(dense[frame, rr, cc] for frame in ids[i])
    np.testing.assert_array_equal(actual, np.concatenate(expected))
    assert actual.dtype == dense.dtype


def test_native_statistics_keep_ties_and_nan_behavior():
    data = np.zeros((3, 7, 7), dtype=float)
    data[1, 2, 2] = 10
    data[1, 5, 5] = 10
    # Disconnected components at equal distance must keep the first label.
    centers, sums, maxima, valid = native._spotStats(
        data.ravel(), np.array([0, data.size]), np.array([data.shape]), 1.0
    )
    np.testing.assert_array_equal(centers[0], [1.0, 2.0, 2.0])


@pytest.mark.parametrize('beam', [[0.0, 0.0, -1.0], [0.1, -0.2, -1.0]])
def test_measured_xy_keeps_scalar_transform_rounding(beam):
    from hexrd.core.transforms import xfcapi

    rng = np.random.default_rng(6)
    angles = rng.uniform([0.02, -np.pi, -np.pi], [0.6, np.pi, np.pi], (40, 3))
    rd = xfcapi.make_rmat_of_expmap(np.array([0.05, 0.02, -0.03]))
    rc = xfcapi.make_rmat_of_expmap(np.array([0.2, -0.4, 0.3]))
    td, ts, tc = (
        np.array([0.2, 1.0, -1000.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, -0.5, 0.6]),
    )
    beam = np.array(beam)
    expected = []
    for angle in angles:
        vector = xfcapi.angles_to_gvec(angle, beam_vec=beam, chi=0.1, rmat_c=rc)
        sample = xfcapi.make_sample_rmat(0.1, angle[2])
        expected.append(
            xfcapi.gvec_to_xy(vector, rd, sample, rc, td, ts, tc, beam_vec=beam)
        )
    actual = native._spotMeasuredXY(angles, rd, rc, 0.1, td, ts, tc, beam)
    np.testing.assert_array_equal(actual, np.array(expected))


def test_large_integer_moments_and_nan_masks_keep_numpy_rounding():
    rng = np.random.default_rng(9)
    for patch, threshold in [
        (rng.integers(2**31, 2**32 - 1, size=(3, 128, 256), dtype=np.uint32), 0.0),
        (np.array([[[1.0, np.nan], [0.0, 3.0]], [[0.0, 0.0], [0.0, 0.0]]]), 0.0),
        (np.array([[[-2, 2], [0, 3]], [[0, 0], [0, 0]]], dtype=np.int32), -3.0),
    ]:
        centers, sums, maxima, valid = native._spotStats(
            patch.ravel(), np.array([0, patch.size]), np.array([patch.shape]), threshold
        )
        labels, count = ndimage.label(patch > threshold, np.ones((3, 3, 3)))
        coms = _weighted_centroids(labels, patch, count)
        closest = np.argmin(np.sum((coms - np.array(patch.shape) / 2) ** 2, axis=1))
        np.testing.assert_array_equal(centers[0], coms[closest])
        selected = patch[labels == closest + 1]
        np.testing.assert_equal(sums[0], selected.sum())
        np.testing.assert_equal(maxima[0], selected.max())


@pytest.mark.parametrize('dtype', ['uint16', 'float32', 'float64'])
def test_native_components_match_scipy_on_sparse_and_dense_patches(dtype):
    rng = np.random.default_rng(12)
    for threshold in (100.0, 700.0, 950.0, 1100.0):
        shapes = np.array([(2, 7, 9), (5, 10, 12), (3, 14, 8)])
        patches = [(rng.random(tuple(s)) * 1000).astype(dtype) for s in shapes]
        offsets = np.r_[0, np.cumsum([p.size for p in patches])]
        centers, sums, maxima, valid = native._spotStats(
            np.concatenate([p.ravel() for p in patches]), offsets, shapes, threshold
        )
        for i, patch in enumerate(patches):
            labels, count = ndimage.label(patch > threshold, np.ones((3, 3, 3)))
            assert valid[i] == bool(count)
            if not count:
                continue
            coms = _weighted_centroids(labels, patch, count)
            closest = np.argmin(np.sum((coms - np.array(patch.shape) / 2) ** 2, axis=1))
            np.testing.assert_array_equal(centers[i], coms[closest])
            np.testing.assert_equal(sums[i], patch[labels == closest + 1].sum())


@pytest.mark.parametrize('threshold', [74.999999, np.float64(74.999999)])
def test_float32_threshold_promotion_matches_numpy(threshold):
    from hexrd.core.instrument._spotutils import _comparison_threshold

    patch = np.full((2, 2, 2), 75.0, dtype=np.float32)
    effective = _comparison_threshold(patch.dtype, threshold)
    _, _, _, valid = native._spotStats(
        patch.ravel(), np.array([0, patch.size]), np.array([patch.shape]), effective
    )
    assert valid[0] == np.any(patch > threshold)


def test_native_views_own_buffers_and_preserve_duplicate_sums():
    import gc

    data = np.array([2, 3, 4], dtype=np.int32)
    frame = csr_array((data, np.array([1, 1, 2]), np.array([0, 2, 2, 3])), shape=(3, 4))
    parent = native._SparseFrames([frame])
    view = parent.view(0, 1, 3, 2)
    del parent, frame, data
    gc.collect()
    actual = view.gather(
        np.array([[0]]),
        np.array([0, 4]),
        np.array([0, 0, 1, 2]),
        np.array([0, 0, 1, 1]),
    )
    np.testing.assert_array_equal(actual, [5, 5, 0, 4])


@pytest.mark.parametrize(
    'offsets,shapes',
    [([0, 8], [[-1, -2, 4]]), ([0, 8], [[2**62, 2, 2]]), ([0, 8], [[2, 2, 3]])],
)
def test_native_rejects_invalid_patch_sizes(offsets, shapes):
    with pytest.raises(ValueError):
        native._spotStats(np.zeros(8), np.array(offsets), np.array(shapes), 1.0)


@pytest.mark.parametrize('dtype', ['uint8', 'int16', 'uint16', 'int32', 'uint32'])
def test_integer_run_components_merge_across_rows_and_frames(dtype):
    patch = np.zeros((3, 5, 13), dtype=dtype)
    patch[0, 0, 1] = 11
    patch[0, 4, 9] = 13
    patch[1, 1, 2] = 17
    patch[1, 3, 8] = 19
    patch[2, 2, 3:8] = 23
    patch[0, 2, 12] = 29
    labels, count = ndimage.label(patch > 1, np.ones((3, 3, 3)))
    assert count == 2
    centers = _weighted_centroids(labels, patch, count)
    chosen = np.argmin(np.sum((centers - np.array(patch.shape) / 2) ** 2, axis=1))
    actual, sums, maxima, valid = native._spotStats(
        patch.ravel(),
        np.array([0, patch.size]),
        np.array([patch.shape]),
        1.0,
    )
    np.testing.assert_array_equal(actual[0], centers[chosen])
    selected = patch[labels == chosen + 1]
    np.testing.assert_equal(sums[0], selected.sum())
    np.testing.assert_equal(maxima[0], selected.max())
    assert valid[0]


@pytest.mark.parametrize(
    'dtype', ['uint8', 'int16', 'uint16', 'int32', 'uint32', 'float64']
)
def test_sparse_extraction_keeps_dense_sample_connectivity_and_moments(dtype):
    rng = np.random.default_rng(607)
    dense = rng.integers(0, 100, (5, 17, 21)).astype(dtype)
    dense[dense < 70] = 0
    series = native._SparseFrames([csr_array(frame) for frame in dense]).view(
        2, 3, 12, 14
    )
    shape = np.array([[3, 5, 7], [3, 9, 11]])
    sizes = np.prod(shape[:, 1:], axis=1)
    offsets = np.r_[0, np.cumsum(sizes)]
    ids = np.array([[0, 2, 4], [1, 1, 3]])
    rows = rng.integers(-3, 16, offsets[-1])
    cols = rng.integers(-3, 18, offsets[-1])
    for threshold in (0.0, 75.0, 110.0, -1.0):
        expected = native._spotStats(
            series.gather(ids, offsets, rows, cols), offsets * 3, shape, threshold
        )
        actual = series.extract(ids, offsets, rows, cols, shape, threshold)
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(after, before)
            assert after.dtype == before.dtype


def test_sparse_extraction_coalesces_duplicates_before_thresholding():
    frame = csr_array(
        (
            np.array([40, 50, 80], dtype=np.int32),
            np.array([1, 1, 2]),
            np.array([0, 0, 3, 3]),
        ),
        shape=(3, 4),
    )
    series = native._SparseFrames([frame])
    shape = np.array([[1, 2, 3]])
    ids, offsets = np.array([[0]]), np.array([0, 6])
    rows, cols = np.array([1, 1, 1, 2, 2, 2]), np.array([0, 1, 2, 0, 1, 2])
    actual = series.extract(ids, offsets, rows, cols, shape, 75.0)
    expected = native._spotStats(
        series.gather(ids, offsets, rows, cols), offsets, shape, 75.0
    )
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(after, before)


def test_packed_spot_writing_preserves_scalar_field_order_and_missing_values():
    from io import StringIO
    from hexrd.core.instrument.hedm_instrument import PatchDataWriter
    from hexrd.core.instrument._spotutils import _SpotData

    records = _SpotData(
        np.array([0, -100, 2]),
        np.array([4, 5, 6]),
        np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0], [7.0, 8.0, 9.0]]),
        np.array([2**31, 0, 100], dtype=np.uint32),
        np.array([2**30, 0, 75], dtype=np.uint32),
        np.array([[0.1, -0.2, 0.3], [0.0, -0.0, 1.0], [np.inf, -np.inf, np.nan]]),
        np.array([[0.2, -0.3, 0.4], [np.nan] * 3, [0.0, -0.0, np.nan]]),
        np.array([[2.0, -3.0], [np.nan, np.nan], [-0.0, 0.0]]),
        np.array([[4.0, -5.0], [6.0, -7.0], [np.inf, np.nan]]),
    )
    scalar, bulk = StringIO(), StringIO()
    left, right = PatchDataWriter(scalar), PatchDataWriter(bulk)
    for row in records:
        left.dump_patch(*row[:7], row[8], row[7])
    right.dump_packed(records)
    assert bulk.getvalue() == scalar.getvalue()
