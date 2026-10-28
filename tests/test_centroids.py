import numpy as np
import pytest
from scipy import ndimage, sparse

from hexrd.hedm.centroids import iter_centroids


@pytest.mark.parametrize('as_sparse', [False, True])
def test_streaming_components_match_volume_labeling(as_sparse):
    volume = np.random.default_rng(107).integers(0, 20, (8, 13, 17))
    threshold = 16
    frames = [sparse.csr_array(f) if as_sparse else f for f in volume]
    intervals = np.column_stack((np.arange(8), np.arange(1, 9)))
    actual = list(iter_centroids(iter(frames), intervals, threshold))
    labels, count = ndimage.label(volume > threshold, np.ones((3, 3, 3)))
    expected = []
    for index in range(1, count + 1):
        z, r, c = np.nonzero(labels == index)
        weights = volume[z, r, c].astype(float)
        mass = weights.sum()
        expected.append(
            [
                np.dot(r, weights) / mass,
                np.dot(c, weights) / mass,
                np.dot(z + 0.5, weights) / mass,
                mass,
                weights.max(),
                len(z),
            ]
        )
    key = lambda row: tuple(row[:3])
    got = [[p.row, p.col, p.omega, p.intensity, p.maximum, p.pixels] for p in actual]
    np.testing.assert_allclose(
        sorted(got, key=key), sorted(expected, key=key), rtol=1e-15, atol=1e-15
    )


def test_streaming_branches_merge_and_gaps_split():
    a = np.array([[10, 0, 20], [0, 0, 0]])
    b = np.array([[0, 30, 0], [0, 0, 0]])
    peaks = list(iter_centroids([a, b], [[0, 1], [1, 2]], 1))
    assert len(peaks) == 1
    p = peaks[0]
    assert p.pixels == 3 and p.intensity == 60
    assert p.col == pytest.approx(70 / 60)
    assert p.omega == 1.0
    assert p.bounds == (0, 2, 0, 1, 0, 3)
    assert len(list(iter_centroids([a, b], [[0, 1], [2, 3]], 1))) == 3
    assert len(list(iter_centroids([a, b], [[0, 1], [1, 0]], 1))) == 3


def test_masks_boundaries_empty_and_descending_omega():
    frames = [np.array([[0, 10], [20, 0]]), np.zeros((2, 2))]
    mask = np.array([[True, False], [True, True]])
    peaks = list(iter_centroids(frames, [[2, 1], [1, 0]], 1, mask))
    assert len(peaks) == 1
    assert (peaks[0].row, peaks[0].col, peaks[0].omega) == (1, 0, 1.5)
    # These flat indices are adjacent but the pixels are not neighbors.
    wide = np.zeros((2, 8))
    wide[0, 7] = wide[1, 0] = 10
    assert len(list(iter_centroids([wide], [[0, 1]], 1))) == 2
    assert list(iter_centroids([np.zeros((2, 2))], [[0, 1]], 1)) == []


def test_sparse_duplicate_pixels_are_coalesced():
    frame = sparse.coo_array(([5, 7], ([1, 1], [1, 1])), shape=(3, 3))
    peaks = list(iter_centroids([frame], [[0, 1]], 10))
    assert len(peaks) == 1 and peaks[0].intensity == 12


@pytest.mark.parametrize('threshold', [-1, np.nan, np.inf])
def test_bad_thresholds(threshold):
    with pytest.raises(ValueError, match='threshold'):
        list(iter_centroids([], np.empty((0, 2)), threshold))


def test_omega_length_must_match_frames():
    with pytest.raises(ValueError, match='Fewer'):
        list(iter_centroids([], [[0, 1]], 1))
    with pytest.raises(ValueError, match='More'):
        list(iter_centroids([np.zeros((2, 2))], np.empty((0, 2)), 1))


def test_catalog_audit_does_not_accept_tiny_measurement_changes():
    from scripts.explore_fit_grains_centroids import audit_measurements

    catalog = np.array([[1.0, 2.0, 3.0]])
    measured = catalog.copy()
    measured[0, 0] = np.nextafter(measured[0, 0], np.inf)
    report = audit_measurements(catalog, measured, [1.0, 1.0, 1.0])
    assert report['matched'] == 1
    assert report['exactly_equal'] == 0
    assert audit_measurements(catalog, catalog, [1.0, 1.0, 1.0])['exactly_equal'] == 1


def test_request_replay_rejects_changed_grain_or_options():
    from scripts.explore_fit_grains_centroid_replay import RequestReplay

    plane, images = object(), object()
    trace = RequestReplay(lambda *args, **kw: [np.array([1.0, 2.0])])
    parameters = np.zeros(12)
    trace(plane, parameters, images, threshold=75, filename=None)
    trace.replaying = True
    changed = parameters.copy()
    changed[0] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match='request changed'):
        trace(plane, changed, images, threshold=75, filename=None)
    with pytest.raises(ValueError, match='request changed'):
        trace(plane, parameters, images, threshold=76, filename=None)
    with pytest.raises(ValueError, match='scope changed'):
        trace(object(), parameters, images, threshold=75, filename=None)
    actual = trace(plane, parameters, images, threshold=75, filename=None)
    np.testing.assert_array_equal(actual[0], [1.0, 2.0])
    with pytest.raises(ValueError, match='Unrecorded'):
        trace(plane, parameters, images, threshold=75, filename=None)
