"""Tests for the staged find-orientations pipeline
(:mod:`hexrd.hedm.find_orientations`) and its typed inputs.

Two layers:

- Unit tests for the pieces with subtle logic (valid-span normalization and
  the numba scoring kernel).  These need no example data.
- A golden end-to-end test on the NIST ruby example (gated on
  ``HEXRD_EXAMPLE_REPO_PATH``, like the legacy find-orientations tests).
  The completeness scores are integer-count ratios, so they are compared
  bit for bit; the stored references were cross-validated against an
  independent implementation of the whole workflow.
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from hexrd.core.material.material_data import PlaneData
from hexrd.hedm.experiment import (
    Clustering,
    ClusteringAlgorithm,
    FindOrientations,
    HedmExperiment,
    SeedSearch,
    SeedSearchMethod,
)
from hexrd.hedm.find_orientations import (
    EtaOmegaMaps,
    FindOrientationsResult,
    _apply_distortion,
    _clean_map,
    _count_hits,
    _count_hits_all,
    _dilated_hit,
    _eta_bin_edges,
    _fiber,
    _filter_maps,
    _find_in_range,
    _find_peaks,
    _load_eta_omega_maps,
    _merge_duplicates,
    _resolve_active_rings,
    _ring_pixels,
    _save_eta_omega_maps,
    cluster_grains,
    find_orientations,
    generate_orientation_fibers,
    normalize_ranges,
    write_results,
)

DATA_DIR = Path(__file__).resolve().parent / 'data' / 'find_orientations_pipeline'


def _plain(fn):
    """The pure-Python function under a numba dispatcher, for line coverage."""
    return fn.py_func


def _fake_plane_data(**overrides):
    """A minimal PlaneData for unit tests: 4 reflections, second one excluded."""
    identity_qsym = np.array([[1.0], [0.0], [0.0], [0.0]])
    fields = dict(
        hkls=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]]),
        exclusions=np.array([False, True, False, False]),
        unexcluded_hkls=np.array([[1, 0, 0], [1, 1, 1], [2, 0, 0]]),
        two_thetas=np.array([0.10, 0.14, 0.17, 0.20]),
        symm_hkls=[np.array([[1, -1], [0, 0], [0, 0]]),
                   np.array([[1], [1], [1]]),
                   np.array([[2], [0], [0]])],
        laue_group='oh',
        q_sym=identity_qsym,
        B=np.eye(3),
        wavelength=0.15,
    )
    fields.update(overrides)
    return PlaneData(**fields)


# ---------------------------------------------------------------------------
# normalize_ranges + _find_in_range: the valid-span machinery
# ---------------------------------------------------------------------------
def test_normalize_ranges_interleaves_pairs():
    # spans must come out interleaved [start, stop, start, stop] and sorted;
    # the binary search in _find_in_range depends on it
    starts = np.array([-1.0, 1.0])
    stops = np.array([-0.5, 1.5])
    spans = normalize_ranges(starts, stops, -np.pi)
    assert np.array_equal(spans, [-1.0, -0.5, 1.0, 1.5])
    assert np.all(np.diff(spans) >= 0)


def test_normalize_ranges_eta_mask():
    # the standard find-orientations eta mask: +/-5 deg around the rotation
    # axis at eta = +/-90 deg; the second range wraps past pi
    mask = np.radians(5)
    ranges = np.array([[-np.pi / 2 + mask, np.pi / 2 - mask],
                       [np.pi / 2 + mask, 3 * np.pi / 2 - mask]])
    spans = normalize_ranges(ranges[:, 0], ranges[:, 1], -np.pi)
    assert np.all(np.diff(spans) >= 0)

    # odd index from the span search means "inside a valid range"
    def valid(angle_deg):
        return _find_in_range(np.radians(angle_deg), spans) & 1 == 1

    assert valid(0) and valid(-45) and valid(120) and valid(-120) and valid(179)
    assert not valid(90) and not valid(-90)        # on the rotation axis
    assert not valid(87) and not valid(-93)        # inside the mask band
    assert valid(84.9) and valid(95.1)             # just outside it


def test_normalize_ranges_full_circle():
    spans = normalize_ranges(np.array([0.0]), np.array([2 * np.pi]), -np.pi)
    assert np.array_equal(spans, [-np.pi, np.pi])


def test_normalize_ranges_wraps_past_offset():
    # a range crossing the window end splits into a leading and trailing span
    spans = normalize_ranges(np.array([3.0]), np.array([3.5]), -np.pi)
    assert np.isclose(spans[0], -np.pi)
    assert np.isclose(spans[1], 3.5 - 2 * np.pi)
    assert np.isclose(spans[2], 3.0)
    assert np.isclose(spans[3], np.pi)


def test_normalize_ranges_rejects_bad_input():
    with pytest.raises(ValueError, match='Invalid'):
        normalize_ranges(np.array([1.0]), np.array([0.5]), -np.pi)
    # overlapping ranges corrupt the interleaved-span invariant
    with pytest.raises(ValueError, match='overlap'):
        normalize_ranges(np.array([0.0, 0.5]), np.array([1.0, 1.5]), -np.pi)


# the numba kernels run both compiled and as their underlying Python
# functions: the latter is what line coverage can see
@pytest.fixture(params=['compiled', 'python'])
def impl(request):
    if request.param == 'compiled':
        return lambda fn: fn
    return _plain


def test_find_in_range_bounds(impl):
    find_in_range = impl(_find_in_range)
    spans = np.array([0.0, 1.0, 2.0, 3.0])
    assert find_in_range(-0.1, spans) == -2      # below
    assert find_in_range(3.0, spans) == -2       # at/after the end
    assert find_in_range(0.5, spans) == 1        # in first span (odd)
    assert find_in_range(1.5, spans) == 2        # in the gap (even)
    assert find_in_range(2.5, spans) == 3        # in second span (odd)


# ---------------------------------------------------------------------------
# the scoring kernel
# ---------------------------------------------------------------------------
def test_dilated_hit(impl):
    dilated_hit = impl(_dilated_hit)
    m = np.zeros((5, 5))
    m[2, 2] = 10.0
    assert dilated_hit(2, 2, 0, 0, m, 1.0) == 1     # direct hit
    assert dilated_hit(0, 0, 0, 0, m, 1.0) == 0     # miss
    assert dilated_hit(0, 0, 2, 2, m, 1.0) == 1     # hit via dilation window
    assert dilated_hit(4, 4, 1, 1, m, 1.0) == 0     # window clipped at edge

    m[1, 1] = np.nan
    assert dilated_hit(1, 1, 0, 0, m, 1.0) == -1    # NaN: off-detector bin


def test_count_hits_simple(impl):
    count_hits = impl(_count_hits)
    # one ring, one reflection; map covers omega x eta = 4 x 8 bins
    ring = np.zeros((4, 8))
    eta_edges = np.linspace(-np.pi, np.pi, 9)
    ome_edges = np.linspace(-0.5, 0.5, 5)
    full_eta = np.array([-np.pi, np.pi])
    full_ome = np.array([-0.5, 0.5])
    ring_for_hkl = np.zeros(1, dtype=np.int64)
    maps3d = ring.reshape(1, 4, 8)

    def score(angs_0, angs_1):
        return count_hits(angs_0, angs_1, ring_for_hkl, eta_edges, ome_edges,
                          full_eta, full_ome, -0.5, maps3d, 0, 0, 1.0)

    nan3 = np.full((1, 3), np.nan)
    on_map = np.array([[0.05, 0.1, 0.0]])          # (tth, eta, ome) in-range
    off_ome = np.array([[0.05, 0.1, 0.7]])         # omega outside the scan

    assert score(nan3, nan3) == 0.0                # nothing predicted
    assert score(on_map, nan3) == 0.0              # predicted, map empty: 0/1

    ring[2, 4] = 5.0                               # bin holding (eta=0.1, ome=0)
    assert score(on_map, nan3) == 1.0              # 1/1
    assert score(on_map, on_map) == 1.0            # both solutions hit: 2/2
    assert score(on_map, off_ome) == 1.0           # invalid solution not counted
    assert score(np.array([[0.05, 2.0, 0.0]]), nan3) == 0.0  # eta elsewhere: 0/1

    # a reflection whose eta falls outside the valid spans is not counted
    masked_eta = np.array([0.0, 1.0])
    assert count_hits(np.array([[0.05, 2.0, 0.0]]), nan3, ring_for_hkl,
                      eta_edges, ome_edges, masked_eta, full_ome, -0.5,
                      maps3d, 0, 0, 1.0) == 0.0
    # omega valid by the spans but off the omega grid: not counted either
    wide_ome = np.array([-0.7, 0.7])
    assert count_hits(np.array([[0.05, 0.1, 0.6]]), nan3, ring_for_hkl,
                      eta_edges, ome_edges, full_eta, wide_ome, -0.7,
                      maps3d, 0, 0, 1.0) == 0.0


def test_count_hits_all_matches_single(impl):
    count_hits_all = impl(_count_hits_all)
    rng = np.random.default_rng(3)
    n_trials, n_hkls = 5, 4
    ring_maps = (rng.random((2, 6, 10)) * 4).round()
    eta_edges = np.linspace(-np.pi, np.pi, 11)
    ome_edges = np.linspace(-0.5, 0.5, 7)
    full_eta = np.array([-np.pi, np.pi])
    full_ome = np.array([-0.5, 0.5])
    ring_for_hkl = np.array([0, 0, 1, 1])
    angs_0 = rng.uniform(-1, 1, size=(n_trials, n_hkls, 3))
    angs_1 = rng.uniform(-1, 1, size=(n_trials, n_hkls, 3))

    scores = count_hits_all(angs_0, angs_1, ring_for_hkl, eta_edges,
                            ome_edges, full_eta, full_ome, -0.5, ring_maps,
                            1, 1, 1.0)
    expected = [
        _count_hits(angs_0[q], angs_1[q], ring_for_hkl, eta_edges, ome_edges,
                    full_eta, full_ome, -0.5, ring_maps, 1, 1, 1.0)
        for q in range(n_trials)
    ]
    assert np.array_equal(scores, expected)


def test_eta_bin_edges():
    edges = _eta_bin_edges(np.radians(0.25))
    assert len(edges) == 1441
    assert edges[0] == -np.pi and edges[-1] == np.pi
    assert np.all(np.diff(edges) > 0)


# ---------------------------------------------------------------------------
# eta-omega map helpers
# ---------------------------------------------------------------------------
def test_resolve_active_rings():
    pd = _fake_plane_data()
    assert np.array_equal(_resolve_active_rings(None, pd), [0, 1, 2])
    assert np.array_equal(
        _resolve_active_rings(np.array([[2, 0, 0], [1, 0, 0]]), pd), [2, 0])
    with pytest.raises(ValueError, match='not non-excluded'):
        _resolve_active_rings(np.array([[1, 1, 0]]), pd)   # the excluded one
    with pytest.raises(ValueError, match='duplicate'):
        _resolve_active_rings(np.array([[1, 0, 0], [1, 0, 0]]), pd)

    # the legacy spelling: master hkl IDs (hkl 1 is excluded -> ring 1 is ID 2)
    assert np.array_equal(_resolve_active_rings(np.array([0, 3]), pd), [0, 2])
    with pytest.raises(ValueError, match='not non-excluded'):
        _resolve_active_rings(np.array([1]), pd)           # excluded ID
    with pytest.raises(ValueError, match='not non-excluded'):
        _resolve_active_rings(np.array([99]), pd)          # out of range
    with pytest.raises(ValueError, match='duplicate'):
        _resolve_active_rings(np.array([0, 0]), pd)


def test_eta_omega_maps_roundtrip(tmp_path):
    maps = EtaOmegaMaps(
        ring_maps=np.arange(24.0).reshape(2, 3, 4),
        eta_edges=np.linspace(-np.pi, np.pi, 5),
        omega_edges=np.linspace(-0.5, 0.5, 4),
        omegas=np.array([-1 / 3, 0.0, 1 / 3]),
        omega_period=np.array([-0.5, -0.5 + 2 * np.pi]),
        two_theta_ranges=np.array([[0.09, 0.11], [0.19, 0.21]]),
        ring_ids=np.array([0, 2]),
        eta_step=np.pi / 2,
    )
    path = tmp_path / 'maps.npz'
    _save_eta_omega_maps(maps, str(path))
    loaded = _load_eta_omega_maps(str(path), _fake_plane_data(), 0.2)
    for field in maps.__dataclass_fields__:
        assert np.array_equal(getattr(loaded, field), getattr(maps, field)), field


def test_load_legacy_eta_omega_maps(tmp_path):
    """The legacy hexrd archive format must load into the same structure."""
    pd = _fake_plane_data()
    ring_maps = np.arange(24.0).reshape(2, 3, 4)
    eta_edges = np.linspace(-np.pi, np.pi, 5)
    ome_edges = np.linspace(-0.5, 0.5, 4)
    path = tmp_path / 'legacy_maps.npz'
    np.savez(
        path,
        dataStore=ring_maps,
        etaEdges=eta_edges,
        omeEdges=ome_edges,
        omegas=0.5 * (ome_edges[:-1] + ome_edges[1:]),
        etas=0.5 * (eta_edges[:-1] + eta_edges[1:]),
        iHKLList=np.array([0, 3]),     # master hkl IDs (hkl 1 is excluded)
        planeData_args=np.array([]),
    )

    maps = _load_eta_omega_maps(str(path), pd, two_theta_width=0.02)
    assert np.array_equal(maps.ring_maps, ring_maps)
    # master hkl IDs 0 and 3 are rings 0 and 2 of the non-excluded list
    assert np.array_equal(maps.ring_ids, [0, 2])
    half = np.radians(0.02) / 2   # two_theta_width is degrees
    assert np.allclose(maps.two_theta_ranges,
                       [[0.10 - half, 0.10 + half], [0.20 - half, 0.20 + half]])
    assert np.isclose(maps.eta_step, eta_edges[1] - eta_edges[0])
    assert np.allclose(maps.omega_period, [ome_edges[0], ome_edges[0] + 2 * np.pi])


def _tiny_maps(ring_maps):
    n_rings, n_ome, n_eta = ring_maps.shape
    return EtaOmegaMaps(
        ring_maps=ring_maps,
        eta_edges=np.linspace(-np.pi, np.pi, n_eta + 1),
        omega_edges=np.linspace(-0.5, 0.5, n_ome + 1),
        omegas=np.linspace(-0.4, 0.4, n_ome),
        omega_period=np.array([-0.5, -0.5 + 2 * np.pi]),
        two_theta_ranges=np.tile([0.09, 0.11], (n_rings, 1)),
        ring_ids=np.arange(n_rings),
        eta_step=2 * np.pi / n_eta,
    )


def test_filter_maps():
    base = np.outer(np.ones(4), [1.0, 2.0, 3.0, 4.0])   # constant eta columns

    untouched = _tiny_maps(base.copy().reshape(1, 4, 4))
    _filter_maps(untouched, median=False, log_fwhm=None)
    assert np.array_equal(untouched.ring_maps[0], base)

    median_only = _tiny_maps(base.copy().reshape(1, 4, 4))
    _filter_maps(median_only, median=True, log_fwhm=None)
    assert np.allclose(median_only.ring_maps[0], 0.0)   # column medians removed

    log_too = _tiny_maps(base.copy().reshape(1, 4, 4))
    log_too.ring_maps[0][2, 2] += 10.0                  # an actual peak
    _filter_maps(log_too, median=True, log_fwhm=1.0)
    peak_map = log_too.ring_maps[0]
    assert np.unravel_index(np.argmax(peak_map), peak_map.shape) == (2, 2)


def test_clean_map():
    m = np.array([[np.nan, 1.0, 5.0],
                  [0.0, np.nan, 2.0]])
    _clean_map(m)
    # NaNs and everything at or below the median (1.5) collapse to the
    # floor, which is then shifted to zero
    assert not np.any(np.isnan(m))
    assert np.array_equal(m, [[0.0, 0.0, 3.5],
                              [0.0, 0.0, 0.5]])


def test_apply_distortion():
    xy = np.array([[10.0, 20.0], [-5.0, 3.0]])

    plain = SimpleNamespace(distortion=SimpleNamespace(
        function_name='', parameters=np.array([])))
    assert _apply_distortion(plain, xy) is xy       # no distortion: passthrough

    from hexrd.core import distortion as distortion_pkg
    params = [6.8e-5, -3.1e-4, -1.2e-4, 2.0, 2.0, 2.0]
    ge = SimpleNamespace(distortion=SimpleNamespace(
        function_name='GE_41RT', parameters=params))
    expected = distortion_pkg.get_mapping('GE_41RT', params).apply(xy)
    assert np.allclose(_apply_distortion(ge, xy), expected)


def test_ring_pixels():
    panel_tth = np.array([0.05, 0.10, 0.11, 0.20])
    panel_eta = np.array([0.0, 1.0, -2.0, 3.0])
    eta_edges = np.linspace(-np.pi, np.pi, 9)

    rp = _ring_pixels(np.array([0.09, 0.12]), panel_tth, panel_eta, eta_edges)
    assert np.array_equal(rp.in_ring, [False, True, True, False])
    # eta bins covered: one around eta=1.0, one around eta=-2.0
    assert len(rp.bins) == 2

    assert _ring_pixels(np.array([0.5, 0.6]), panel_tth, panel_eta,
                        eta_edges) is None


# ---------------------------------------------------------------------------
# seed peaks and fibers
# ---------------------------------------------------------------------------
def _two_spot_map():
    """A 40x60 map with gaussian spots at (10, 15) and (30, 45)."""
    om, eta = np.mgrid[0:40, 0:60].astype(float)
    spots = (np.exp(-((om - 10) ** 2 + (eta - 15) ** 2) / 4)
             + np.exp(-((om - 30) ** 2 + (eta - 45) ** 2) / 4))
    return 100 * spots


@pytest.mark.parametrize('method,kwargs', [
    ('label', {'filter_radius': 1, 'threshold': 1}),
    ('label', {'threshold': 1}),
    ('blob_log', {'min_sigma': 0.5, 'max_sigma': 5, 'num_sigma': 10,
                  'threshold': 0.01, 'overlap': 0.1}),
    ('blob_dog', {'min_sigma': 0.5, 'max_sigma': 5, 'sigma_ratio': 1.6,
                  'threshold': 0.01, 'overlap': 0.1}),
])
def test_find_peaks_methods(method, kwargs):
    seed_search = SeedSearch(
        hkl_seeds=np.array([0]), fiber_step=np.radians(0.5),
        method=SeedSearchMethod(method), method_kwargs=kwargs)
    num_spots, coms = _find_peaks(_two_spot_map(), seed_search)
    assert num_spots == 2
    coms = coms[np.argsort(coms[:, 0])]
    assert np.allclose(coms[0], [10, 15], atol=1.0)
    assert np.allclose(coms[1], [30, 45], atol=1.0)


def test_fiber_returns_unit_quaternions():
    quats = _fiber(np.array([1, 0, 0]), tth=0.2, eta_c=0.1, ome_c=0.3,
                   chi=0.0, b_matrix=np.eye(3), fiber_ndiv=120, csym='oh')
    assert quats.shape[0] == 4
    assert 0 < quats.shape[1] <= 120
    assert np.allclose(np.linalg.norm(quats, axis=0), 1.0)
    # unique orientations only
    assert np.unique(quats.round(10), axis=1).shape[1] == quats.shape[1]


def test_prefetch_imports():
    from hexrd.hedm.find_orientations import (
        _prefetch_heavy_imports,
        _prefetch_skimage,
    )

    _prefetch_heavy_imports()
    _prefetch_skimage()


def test_build_maps_accumulates_in_float64():
    """The map builder must sum uint16 frames without wrapping, skip image
    series with no matching panel, and skip rings that miss every panel."""
    from hexrd.core.experiment import Detector
    from hexrd.hedm.find_orientations import build_eta_omega_maps
    from scipy.sparse import csr_array

    # a small panel 100mm from the sample: pixel two-thetas are a few mrad,
    # so a [0, 0.01] rad ring window covers the whole panel
    detector = Detector.from_dict('p0', {
        'pixels': {'rows': 4, 'columns': 6, 'size': [0.2, 0.2]},
        'transform': {'translation': [0.0, 0.0, -100.0]},
    })
    pd = _fake_plane_data(
        two_thetas=np.array([0.005, 0.14, 1.0, 1.1]),
        # ring 0 covers the panel; rings 1-2 (tth ~ 1 rad) miss it entirely
    )

    hot = np.zeros((4, 6), dtype=np.uint16)
    hot[1, 2] = 40000                     # 2 frames x 40000 > uint16 max
    faint = np.zeros((4, 6), dtype=np.uint16)
    faint[3, 5] = 7

    class _Ims:
        def __init__(self, panel, frames):
            self.panel = panel
            self.images = [csr_array(f) for f in frames]
            self.omega = np.radians(
                [[0.0, 120.0], [120.0, 240.0], [240.0, 360.0]])

        def __len__(self):
            return len(self.images)

    fo = FindOrientations.from_dict({
        'orientation_maps': {'threshold': 10, 'eta_step': 5.0},
        'seed_search': {'hkl_seeds': [0]},
        'clustering': {'radius': 1.0, 'completeness': 0.5},
    })
    experiment = SimpleNamespace(
        find_orientations=fo,
        active_material=SimpleNamespace(two_theta_width=0.6),  # degrees
        max_workers=1,
        detectors={'p0': detector},
        beam=SimpleNamespace(vector=np.array([0.0, 0.0, -1.0])),
        image_series_list=[
            _Ims('p0', [hot, hot, faint]),
            _Ims('unknown_panel', [hot, hot, faint]),   # no matching detector
        ],
    )

    maps = build_eta_omega_maps(experiment, pd)
    assert maps.ring_maps.shape[0] == 3
    assert maps.ring_maps.shape[1] == 3               # one row per frame

    # the hot pixel's intensity sums past the uint16 max, in its eta bin
    ring0 = maps.ring_maps[0]
    assert np.nansum(ring0[0]) == 40000.0
    assert np.nansum(ring0[1]) == 40000.0
    assert np.nanmax(ring0) <= 40000.0
    # the faint pixel sits below the threshold: zeroed, not dropped
    assert np.nansum(ring0[2]) == 0.0
    # rings that miss the panel stay NaN everywhere
    assert np.all(np.isnan(maps.ring_maps[1:]))
    # omega axis comes from the image series
    assert np.allclose(maps.omega_edges,
                       np.radians([0.0, 120.0, 240.0, 360.0]))


def test_omega_grid():
    from hexrd.hedm.find_orientations import _omega_grid

    # contiguous scan: one row per frame, edges from the scan itself
    contiguous = np.radians(np.column_stack([np.arange(4) * 1.0,
                                             np.arange(1, 5) * 1.0]))
    rows, omegas, edges, period = _omega_grid(contiguous)
    assert np.array_equal(rows, [0, 1, 2, 3])
    assert np.allclose(edges, np.radians([0, 1, 2, 3, 4]))
    assert np.allclose(omegas, np.radians([0.5, 1.5, 2.5, 3.5]))
    assert np.allclose(period, np.radians([0, 360]))

    # two wedges with a gap: frames land on the uniform grid, gap rows exist
    wedged = np.radians(np.array([[-10.0, -9.0], [-9.0, -8.0],
                                  [5.0, 6.0], [6.0, 7.0]]))
    rows, omegas, edges, period = _omega_grid(wedged)
    assert np.array_equal(rows, [0, 1, 15, 16])
    assert len(omegas) == 17
    assert np.allclose(edges, np.radians(-10.0 + np.arange(18)))

    with pytest.raises(ValueError, match='step size'):
        _omega_grid(np.radians(np.array([[0.0, 1.0], [1.0, 3.0],
                                         [7.0, 8.0]])))
    with pytest.raises(ValueError, match='align'):
        _omega_grid(np.radians(np.array([[0.0, 1.0], [1.5, 2.5]])))


def test_build_maps_multi_wedge_leaves_nan_gap_rows():
    """A wedged scan's gap rows stay NaN, so scoring excludes them."""
    from hexrd.core.experiment import Detector
    from hexrd.hedm.find_orientations import build_eta_omega_maps
    from scipy.sparse import csr_array

    detector = Detector.from_dict('p0', {
        'pixels': {'rows': 4, 'columns': 6, 'size': [0.2, 0.2]},
        'transform': {'translation': [0.0, 0.0, -100.0]},
    })
    pd = _fake_plane_data(two_thetas=np.array([0.005, 0.14, 1.0, 1.1]))

    frame = np.zeros((4, 6), dtype=np.uint16)
    frame[1, 2] = 100

    class _Ims:
        panel = 'p0'
        # two frames, then a 3-frame gap, then one more
        omega = np.radians([[0.0, 1.0], [1.0, 2.0], [5.0, 6.0]])
        images = [csr_array(frame)] * 3

        def __len__(self):
            return 3

    fo = FindOrientations.from_dict({
        'orientation_maps': {'threshold': 10, 'eta_step': 5.0},
        'seed_search': {'hkl_seeds': [0]},
        'clustering': {'radius': 1.0, 'completeness': 0.5},
    })
    experiment = SimpleNamespace(
        find_orientations=fo,
        active_material=SimpleNamespace(two_theta_width=0.6),
        max_workers=1,
        detectors={'p0': detector},
        beam=SimpleNamespace(vector=np.array([0.0, 0.0, -1.0])),
        image_series_list=[_Ims()],
    )

    maps = build_eta_omega_maps(experiment, pd)
    ring0 = maps.ring_maps[0]
    assert ring0.shape[0] == 6                     # uniform 1-degree grid
    covered = ~np.all(np.isnan(ring0), axis=1)
    assert np.array_equal(covered, [True, True, False, False, False, True])
    assert np.allclose(maps.omega_edges, np.radians(np.arange(7.0)))


def test_generate_orientation_fibers_skips_nan_spots(monkeypatch):
    import hexrd.hedm.find_orientations as pipeline

    monkeypatch.setattr(
        pipeline, '_find_peaks',
        lambda ring_map, seed_search: (2, np.array([[np.nan, np.nan],
                                                    [1.0, 2.0]])))
    fo = FindOrientations.from_dict({
        'seed_search': {'hkl_seeds': [0],
                        'method': {'label': {'threshold': 1}}},
        'clustering': {'radius': 1.0, 'completeness': 0.5},
    })
    experiment = SimpleNamespace(find_orientations=fo,
                                 oscillation_stage=SimpleNamespace(chi=0.0),
                                 max_workers=1)
    maps = _tiny_maps(np.zeros((1, 4, 8)))

    fibers = pipeline.generate_orientation_fibers(
        experiment, _fake_plane_data(), maps)
    # only the one valid spot contributes a fiber
    assert fibers.shape[0] == 4 and fibers.shape[1] > 0


def test_generate_orientation_fibers_requires_seeds():
    fo = FindOrientations.from_dict({
        'seed_search': {'method': {'label': {'threshold': 1}}},
        'clustering': {'radius': 1.0, 'completeness': 0.5},
    })
    experiment = SimpleNamespace(find_orientations=fo,
                                 oscillation_stage=SimpleNamespace(chi=0.0),
                                 max_workers=1)
    maps = _tiny_maps(np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match='hkl_seeds'):
        generate_orientation_fibers(experiment, _fake_plane_data(), maps)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def _rot_z_quats(angles_deg):
    half = np.radians(np.asarray(angles_deg, dtype=float)) / 2
    return np.vstack([np.cos(half), np.zeros_like(half),
                      np.zeros_like(half), np.sin(half)])


def _clustering_experiment(algorithm, radius=1.0, completeness=0.5):
    fo = FindOrientations.from_dict({
        'seed_search': {'hkl_seeds': [0]},
        'clustering': {'radius': radius, 'completeness': completeness,
                       'algorithm': str(algorithm)},
    })
    return SimpleNamespace(find_orientations=fo)


@pytest.mark.parametrize('algorithm', list(ClusteringAlgorithm))
def test_cluster_grains_two_clusters(algorithm):
    # two tight bundles of rotations about z, 10 degrees apart
    rng = np.random.default_rng(11)
    jitter = lambda: rng.normal(scale=0.01, size=8)
    quats = _rot_z_quats(np.r_[jitter(), 10.0 + jitter()])
    completeness = np.full(quats.shape[1], 0.9)

    qbar = cluster_grains(_clustering_experiment(algorithm),
                          _fake_plane_data(), quats, completeness,
                          min_samples=2)
    assert qbar.shape == (4, 2)
    angles = np.degrees(2 * np.arctan2(qbar[3], qbar[0]))
    assert np.allclose(sorted(np.abs(angles)), [0.0, 10.0], atol=0.1)


def test_cluster_grains_thresholds_and_small_inputs():
    experiment = _clustering_experiment('dbscan', completeness=0.5)
    quats = _rot_z_quats([0.0, 10.0])

    # no candidate above the completeness threshold
    none = cluster_grains(experiment, _fake_plane_data(), quats,
                          np.array([0.1, 0.2]))
    assert none.shape == (4, 0)

    # exactly one candidate: returned as-is
    one = cluster_grains(experiment, _fake_plane_data(), quats,
                         np.array([0.9, 0.2]))
    assert np.allclose(one, quats[:, :1])


def test_cluster_grains_large_input_falls_back_to_ort_dbscan():
    # the O(n^2) quaternion-metric algorithms are swapped out above 25000
    # candidates; this must stay fast and still find both clusters
    rng = np.random.default_rng(5)
    n = 25001
    angles = np.r_[rng.normal(0.0, 0.01, n // 2),
                   rng.normal(10.0, 0.01, n - n // 2)]
    quats = _rot_z_quats(angles)
    completeness = np.full(n, 0.9)

    qbar = cluster_grains(_clustering_experiment('sph-dbscan'),
                          _fake_plane_data(), quats, completeness,
                          min_samples=5)
    assert qbar.shape == (4, 2)


def test_merge_duplicates():
    from hexrd.core.transforms import xfcapi

    qsym = np.ascontiguousarray(_fake_plane_data().q_sym)

    def quat_distance(x, y):
        return xfcapi.quat_distance(np.ascontiguousarray(x),
                                    np.ascontiguousarray(y), qsym)

    # centroids at 0, 0.02 and 10 degrees: the first two merge within 1 degree
    qbar = _rot_z_quats([0.0, 0.02, 10.0])
    merged = _merge_duplicates(qbar, qsym, quat_distance, radius=1.0)
    assert merged.shape == (4, 2)

    # nothing within the radius: unchanged
    apart = _rot_z_quats([0.0, 10.0])
    assert np.array_equal(
        _merge_duplicates(apart, qsym, quat_distance, radius=1.0), apart)


# ---------------------------------------------------------------------------
# results files
# ---------------------------------------------------------------------------
def test_write_results_formats(tmp_path):
    # one grain: a 60 degree rotation about z, so exp_map = [0, 0, pi/3]
    qbar = _rot_z_quats([60.0])
    trials = _rot_z_quats([0.0, 30.0, 60.0])
    scores = np.array([0.1, 0.2, 0.9])
    results = FindOrientationsResult(
        grain_orientations=qbar, test_orientations=trials, completeness=scores)
    experiment = SimpleNamespace(
        analysis_dir=str(tmp_path / 'analysis'),
        active_material=SimpleNamespace(active='ruby'))

    out_dir = write_results(results, experiment)
    assert out_dir == experiment.analysis_dir

    accepted = np.loadtxt(Path(out_dir) / 'accepted-orientations-ruby.dat',
                          ndmin=2)
    assert np.allclose(accepted, qbar.T)

    with np.load(Path(out_dir) / 'scored-orientations-ruby.npz') as scored:
        assert np.array_equal(scored['test_quaternions'], trials)
        assert np.array_equal(scored['score'], scores)

    grains_out = Path(out_dir) / 'grains.out'
    header = grains_out.read_text().splitlines()[0]
    # the header must keep the legacy GrainDataWriter column set
    for column in ('grain ID', 'completeness', 'chi^2', 'exp_map_c[0]',
                   't_vec_c[2]', 'inv(V_s)[0,1]*sqrt(2)', 'ln(V_s)[0,1]'):
        assert column in header

    table = np.loadtxt(grains_out, ndmin=2)
    assert table.shape == (1, 21)
    grain_id, completeness, chi2 = table[0, :3]
    assert (grain_id, completeness, chi2) == (0.0, 1.0, 0.0)
    assert np.allclose(table[0, 3:6], [0.0, 0.0, np.pi / 3])   # exp_map
    assert np.allclose(table[0, 6:9], 0.0)                     # centroid
    assert np.allclose(table[0, 9:15], [1, 1, 1, 0, 0, 0])     # inv(V_s)
    assert np.allclose(table[0, 15:21], 0.0)                   # ln(V_s)


# ---------------------------------------------------------------------------
# the new pipeline must not touch the legacy config machinery
# ---------------------------------------------------------------------------
def test_pipeline_imports_are_config_free():
    code = (
        'import sys\n'
        'import hexrd.hedm.find_orientations\n'
        'import hexrd.hedm.experiment\n'
        'import hexrd.hedm.cli.find_orientations\n'
        'bad = [m for m in sys.modules'
        " if m.startswith('hexrd') and '.config' in m]\n"
        'assert not bad, f"legacy config modules loaded: {bad}"\n'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


# ---------------------------------------------------------------------------
# golden end-to-end on the NIST ruby example
# ---------------------------------------------------------------------------
def _ruby_config_path(example_repo_path, tmp_path, **find_orientations_updates):
    """Write the single-GE ruby config with outputs redirected to tmp_path."""
    include = example_repo_path / 'NIST_ruby' / 'single_GE' / 'include'
    with open(include / 'cstudy.yml') as f:
        cfg = next(yaml.safe_load_all(f))

    # absolute inputs from the example repo; outputs stay under tmp_path
    cfg['material']['definitions'] = str(include / 'materials.h5')
    cfg['instrument'] = str(include / 'ge_detector.yml')
    for entry in cfg['image_series']['data']:
        entry['file'] = str((include / entry['file']).resolve())
    cfg['analysis_name'] = 'analysis'
    cfg['find_orientations'].update(find_orientations_updates)

    config_path = tmp_path / 'ruby.yml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config_path


@pytest.fixture
def ruby_experiment(example_repo_path, tmp_path):
    """The single-GE ruby analysis, with outputs redirected to tmp_path."""
    return HedmExperiment(str(_ruby_config_path(example_repo_path, tmp_path)))


def test_ruby_material_bridge(ruby_experiment):
    materials = ruby_experiment.get_materials()
    assert any(m.name == 'ruby' for m in materials)

    material = ruby_experiment.get_active_material()
    pd = material.plane_data
    assert material.name == 'ruby'
    assert pd.laue_group == 'd3d'
    assert len(pd.unexcluded_hkls) == 6            # dmin = 1.60 angstrom
    assert len(pd.symm_hkls) == 6
    assert np.count_nonzero(~pd.exclusions) == 6
    assert pd.two_thetas.shape == (len(pd.hkls),)
    assert 0.15 < pd.wavelength < 0.16             # 80.725 keV beam, angstrom
    assert pd.B.shape == (3, 3)


def test_ruby_pipeline_golden(ruby_experiment):
    results = find_orientations(ruby_experiment)

    # completeness scores are hits/total counts: bit-exact is expected
    expected_scores = np.load(DATA_DIR / 'ruby_completeness.npy')
    assert results.test_orientations.shape == (4, len(expected_scores))
    assert np.array_equal(results.completeness, expected_scores)

    expected_qbar = np.load(DATA_DIR / 'ruby_qbar.npy')
    assert results.num_grains == 1
    assert np.allclose(results.grain_orientations, expected_qbar,
                       rtol=0.0, atol=1.0e-12)


def test_ruby_pipeline_maps_cache(ruby_experiment):
    """A second run must load the cached maps and reproduce the results;
    a clean run must rebuild them and still agree."""
    first = find_orientations(ruby_experiment)
    assert os.path.exists(ruby_experiment.eta_ome_maps_file)
    second = find_orientations(ruby_experiment)
    assert np.array_equal(first.completeness, second.completeness)
    assert np.array_equal(first.grain_orientations, second.grain_orientations)

    os.remove(ruby_experiment.eta_ome_maps_file)
    cleaned = find_orientations(ruby_experiment, clean=True)
    assert os.path.exists(ruby_experiment.eta_ome_maps_file)
    assert np.array_equal(first.completeness, cleaned.completeness)


def test_ruby_quaternion_grid(example_repo_path, tmp_path):
    """use_quaternion_grid scores a fixed set of trials instead of seed search."""
    grid = np.hstack([
        np.load(DATA_DIR / 'ruby_qbar.npy'),         # the known grain
        _rot_z_quats([20.0, 40.0, 60.0]),             # decoys
    ])
    grid_path = tmp_path / 'grid.npy'
    np.save(grid_path, grid)

    # a non-label seed method exercises the skimage prefetch, and is
    # otherwise ignored: the grid replaces the seed search entirely
    config_path = _ruby_config_path(
        example_repo_path, tmp_path,
        use_quaternion_grid=str(grid_path),
        seed_search={'hkl_seeds': [0], 'method': {'blob_log': {}}})
    experiment = HedmExperiment(str(config_path))
    results = find_orientations(experiment)

    assert results.test_orientations.shape == grid.shape
    assert results.num_grains == 1
    # the surviving orientation is the known grain, not a decoy
    expected = np.load(DATA_DIR / 'ruby_qbar.npy')
    assert np.allclose(np.abs(results.grain_orientations.T @ expected), 1.0,
                       atol=1e-6)


# ---------------------------------------------------------------------------
# the CLI driver
# ---------------------------------------------------------------------------
def _cli_args(config_path, **overrides):
    import argparse

    defaults = dict(yml=str(config_path), quiet=True, force=False,
                    clean=False, study=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cli_parser():
    import argparse

    from hexrd.hedm.cli.find_orientations import configure_parser, execute

    parser = argparse.ArgumentParser()
    configure_parser(parser.add_subparsers())
    args = parser.parse_args(
        ['find-orientations', 'config.yml', '-q', '-c', '--study', '2'])
    assert args.yml == 'config.yml'
    assert args.quiet and args.clean and not args.force
    assert args.study == 2
    assert args.func is execute


def test_cli_refuses_to_clobber(example_repo_path, tmp_path):
    from hexrd.hedm.cli.find_orientations import execute

    config_path = _ruby_config_path(example_repo_path, tmp_path)
    analysis_dir = tmp_path / 'analysis'
    analysis_dir.mkdir()
    (analysis_dir / 'accepted-orientations-ruby.dat').write_text('')

    with pytest.raises(SystemExit):
        execute(_cli_args(config_path), None)


def test_cli_end_to_end(example_repo_path, tmp_path):
    from hexrd.hedm.cli.find_orientations import execute

    config_path = _ruby_config_path(example_repo_path, tmp_path)
    execute(_cli_args(config_path), None)

    analysis_dir = tmp_path / 'analysis'
    accepted = np.loadtxt(analysis_dir / 'accepted-orientations-ruby.dat',
                          ndmin=2)
    expected = np.load(DATA_DIR / 'ruby_qbar.npy')
    assert np.allclose(np.abs(accepted @ expected), 1.0, atol=1e-6)
    assert (analysis_dir / 'grains.out').exists()
    assert (analysis_dir / 'scored-orientations-ruby.npz').exists()
    assert (analysis_dir / 'find-orientations-ruby.log').exists()

    # a second run without --force must refuse to clobber the analysis
    with pytest.raises(SystemExit):
        execute(_cli_args(config_path), None)

    # --force runs again, reusing the cached maps
    execute(_cli_args(config_path, force=True), None)


# ---------------------------------------------------------------------------
# golden end-to-end on the multiruby Dexela composite (8 panels)
# ---------------------------------------------------------------------------
def _multiruby_experiment(example_repo_path, tmp_path, maps_file=None):
    """The multiruby composite analysis, with outputs redirected to tmp_path."""
    include = example_repo_path / 'NIST_ruby' / 'multiruby_dexelas' / 'include'
    with open(include / 'mruby_config_composite.yml') as f:
        cfg = next(yaml.safe_load_all(f))

    cfg['material']['definitions'] = str(include / 'materials.h5')
    cfg['instrument'] = str(include / cfg['instrument'])
    for entry in cfg['image_series']['data']:
        entry['file'] = str((include / entry['file']).resolve())
    cfg['analysis_name'] = 'analysis'
    if maps_file is not None:
        cfg['find_orientations']['orientation_maps']['file'] = str(maps_file)

    config_path = tmp_path / 'multiruby.yml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    from hexrd.hedm.experiment import HedmExperiment

    return HedmExperiment(str(config_path))


@pytest.fixture
def multiruby_reference_maps(example_repo_path):
    results = example_repo_path / 'NIST_ruby' / 'multiruby_dexelas' / 'results'
    return results / 'results_mruby_composite_hexrd06_py27_ruby_eta-ome_maps.npz'


@pytest.fixture
def multiruby_reference_orientations(example_repo_path):
    results = example_repo_path / 'NIST_ruby' / 'multiruby_dexelas' / 'results'
    path = results / 'accepted_orientations_results_mruby_composite_hexrd06_py27_ruby.dat'
    return np.loadtxt(path, ndmin=2)


def test_multiruby_full_pipeline_golden(example_repo_path, tmp_path,
                                        multiruby_reference_maps,
                                        multiruby_reference_orientations):
    """The full workflow from raw frames on the 8-panel Dexela composite.

    The shipped reference maps predate the float64 fix (their intensities
    wrapped at the uint16 max) and were produced by hexrd 0.6 on python 2.7,
    whose ring windows put a handful of edge pixels in different eta bins.
    So the map comparison asserts what that reference can support: identical
    angular grids, coverage equal up to a few whole eta columns, and the
    overwhelming majority of shared bins **bit-identical**.  The physics is
    then pinned by the found grains matching the reference orientations.
    """
    from hexrd.hedm.find_orientations import _load_eta_omega_maps
    from hexrd.core.rotations import misorientation

    experiment = _multiruby_experiment(example_repo_path, tmp_path)
    material = experiment.get_active_material()
    results = find_orientations(experiment, material)

    # the maps this run built and cached, vs the shipped reference
    maps = _load_eta_omega_maps(
        experiment.eta_ome_maps_file, material.plane_data,
        experiment.active_material.two_theta_width,
    )
    reference = _load_eta_omega_maps(
        str(multiruby_reference_maps), material.plane_data,
        experiment.active_material.two_theta_width,
    )

    assert maps.ring_maps.shape == reference.ring_maps.shape
    assert np.linalg.norm(maps.eta_edges - reference.eta_edges) < 1.0e-6
    assert np.linalg.norm(maps.omega_edges - reference.omega_edges) < 1.0e-6
    assert np.array_equal(maps.ring_ids, reference.ring_ids)
    for ring in range(len(maps.ring_maps)):
        mine, ref = maps.ring_maps[ring], reference.ring_maps[ring]
        # everything we cover, the reference covered
        assert not np.any(~np.isnan(mine) & np.isnan(ref))
        # what the reference covers and we don't is a few whole eta columns
        missing = np.isnan(mine) & ~np.isnan(ref)
        missing_cols = np.unique(np.nonzero(missing)[1])
        assert len(missing_cols) <= 5, f'ring {ring}: {missing_cols}'
        assert all(missing[:, c].all() for c in missing_cols)
        # shared bins: nearly all values match the reference bit for bit
        both = ~np.isnan(mine) & ~np.isnan(ref)
        exact = np.mean(mine[both] == ref[both])
        assert exact > 0.97, f'ring {ring}: only {exact:.2%} bins bit-equal'

    # and the grains those maps produce match the reference orientations
    reference_quats = multiruby_reference_orientations
    assert results.num_grains == len(reference_quats)
    qsym = material.plane_data.q_sym
    for i, q in enumerate(results.grain_orientations.T):
        angles, _ = misorientation(q.reshape(4, 1), reference_quats.T, (qsym,))
        assert np.degrees(np.min(angles)) < 0.05, f'grain {i} misorientation'


def test_multiruby_find_orientations_golden(example_repo_path, tmp_path,
                                            multiruby_reference_maps,
                                            multiruby_reference_orientations):
    """Full workflow from the reference maps: must find the reference grains."""
    from hexrd.core.rotations import misorientation

    experiment = _multiruby_experiment(
        example_repo_path, tmp_path, maps_file=multiruby_reference_maps
    )
    material = experiment.get_active_material()
    results = find_orientations(experiment, material)

    reference = multiruby_reference_orientations
    assert results.num_grains == len(reference)
    qsym = material.plane_data.q_sym
    for i, q in enumerate(results.grain_orientations.T):
        angles, _ = misorientation(q.reshape(4, 1), reference.T, (qsym,))
        assert np.degrees(np.min(angles)) < 0.05, f'grain {i} misorientation'
