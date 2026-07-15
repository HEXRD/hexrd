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
from pathlib import Path

import numpy as np
import pytest
import yaml

from hexrd.hedm.find_orientations import (
    _count_hits,
    _dilated_hit,
    _eta_bin_edges,
    _find_in_range,
    find_orientations,
    normalize_ranges,
)

DATA_DIR = Path(__file__).resolve().parent / 'data' / 'find_orientations_pipeline'


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


def test_find_in_range_bounds():
    spans = np.array([0.0, 1.0, 2.0, 3.0])
    assert _find_in_range(-0.1, spans) == -2      # below
    assert _find_in_range(3.0, spans) == -2       # at/after the end
    assert _find_in_range(0.5, spans) == 1        # in first span (odd)
    assert _find_in_range(1.5, spans) == 2        # in the gap (even)
    assert _find_in_range(2.5, spans) == 3        # in second span (odd)


# ---------------------------------------------------------------------------
# the scoring kernel
# ---------------------------------------------------------------------------
def test_dilated_hit():
    m = np.zeros((5, 5))
    m[2, 2] = 10.0
    assert _dilated_hit(2, 2, 0, 0, m, 1.0) == 1     # direct hit
    assert _dilated_hit(0, 0, 0, 0, m, 1.0) == 0     # miss
    assert _dilated_hit(0, 0, 2, 2, m, 1.0) == 1     # hit via dilation window
    assert _dilated_hit(4, 4, 1, 1, m, 1.0) == 0     # window clipped at edge

    m[1, 1] = np.nan
    assert _dilated_hit(1, 1, 0, 0, m, 1.0) == -1    # NaN: off-detector bin


def test_count_hits_simple():
    # one ring, one reflection; map covers omega x eta = 4 x 8 bins
    ring = np.zeros((4, 8))
    eta_edges = np.linspace(-np.pi, np.pi, 9)
    ome_edges = np.linspace(-0.5, 0.5, 5)
    full_eta = np.array([-np.pi, np.pi])
    full_ome = np.array([-0.5, 0.5])
    ring_for_hkl = np.zeros(1, dtype=np.int64)
    maps3d = ring.reshape(1, 4, 8)

    def score(angs_0, angs_1):
        return _count_hits(angs_0, angs_1, ring_for_hkl, eta_edges, ome_edges,
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


def test_eta_bin_edges():
    edges = _eta_bin_edges(np.radians(0.25))
    assert len(edges) == 1441
    assert edges[0] == -np.pi and edges[-1] == np.pi
    assert np.all(np.diff(edges) > 0)


# ---------------------------------------------------------------------------
# golden end-to-end on the NIST ruby example
# ---------------------------------------------------------------------------
@pytest.fixture
def ruby_experiment(example_repo_path, tmp_path):
    """The single-GE ruby analysis, with outputs redirected to tmp_path."""
    include = example_repo_path / 'NIST_ruby' / 'single_GE' / 'include'
    with open(include / 'cstudy.yml') as f:
        cfg = next(yaml.safe_load_all(f))

    # absolute inputs from the example repo; outputs stay under tmp_path
    cfg['material']['definitions'] = str(include / 'materials.h5')
    cfg['instrument'] = str(include / 'ge_detector.yml')
    for entry in cfg['image_series']['data']:
        entry['file'] = str((include / entry['file']).resolve())
    cfg['analysis_name'] = 'analysis'

    config_path = tmp_path / 'ruby.yml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    from hexrd.core.config.experiment import Experiment

    return Experiment(str(config_path))


def test_ruby_material_bridge(ruby_experiment):
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
    """A second run must load the cached maps and reproduce the results."""
    first = find_orientations(ruby_experiment)
    assert os.path.exists(ruby_experiment.eta_ome_maps_file)
    second = find_orientations(ruby_experiment)
    assert np.array_equal(first.completeness, second.completeness)
    assert np.array_equal(first.grain_orientations, second.grain_orientations)


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

    from hexrd.core.config.experiment import Experiment

    return Experiment(str(config_path))


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
