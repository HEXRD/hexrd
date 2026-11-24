"""Exact native mapping and bounded incremental-map contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

from hexrd.core import imageseries
from hexrd.core.extensions import transforms as native
from hexrd.core.instrument import HEDMInstrument, PlanarDetector
from hexrd.core.material import Material
from hexrd.hedm.experiment import HedmExperiment
from hexrd.hedm.find_orientations import (
    EtaOmegaMapAccumulator, EtaOmegaMapPlan, _direct_sum_safe,
    build_eta_omega_maps, load_plane_data,
)


def exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def prefix_reference(frame, blocks, threshold, bins):
    weights = frame.astype('float64').ravel()
    weights = np.where(weights < threshold, 0., weights)
    result = []
    with np.errstate(invalid='ignore'):
        for ring in blocks:
            cumulative = np.zeros(bins + 1)
            for pixels, cuts in ring:
                prefix = np.r_[0., np.cumsum(weights[pixels])]
                cumulative += prefix[cuts]
            result.append(np.diff(cumulative))
    return np.asarray(result)


@pytest.mark.parametrize('dtype', ['bool', 'u1', 'i1', 'u2', 'i2', 'u4', 'i4',
                                 'u8', 'i8', 'f4', 'f8', '>f8'])
@pytest.mark.parametrize('threshold', [-1., 0., 2.5, np.inf, np.nan])
def test_native_histogram_preserves_block_reductions(dtype, threshold):
    frame = np.arange(20).reshape(4, 5).astype(dtype)
    blocks = [[(np.array([9, 1, 5, 7]), np.array([0, 2, 4])),
               (np.array([10, 11, 13, 15]), np.array([0, 1, 4]))],
              [(np.array([0, 4, 8]), np.array([0, 1, 3]))], []]
    plan = native._HistogramPlan(blocks, frame.shape, 2)
    exact(plan.accumulate(frame, threshold), prefix_reference(frame, blocks, threshold, 2))


@pytest.mark.parametrize('view', ['strided', 'unaligned', 'readonly'])
def test_native_histogram_borrows_safe_array_views(view):
    raw = np.arange(40., dtype='f8').reshape(4, 10)
    frame = raw[:, ::2]
    if view == 'unaligned':
        frame = np.ndarray((4, 5), dtype='f8', buffer=bytearray(161), offset=1)
        frame[:] = raw[:, ::2]
    if view == 'readonly':
        frame = frame.copy()
        frame.flags.writeable = False
    pixels = np.arange(20, dtype='i8')
    cuts = np.array([0, 10, 20])
    blocks = [[(pixels, cuts)]]
    plan = native._HistogramPlan(blocks, frame.shape, 2)
    expected = prefix_reference(frame, blocks, 2.5, 2)
    pixels[:] = 0  # The plan owns its geometry rather than borrowing mutable arrays.
    exact(plan.accumulate(frame, 2.5), expected)


def test_native_histogram_special_values_and_validation():
    frame = np.array([[1e16, 1., -1e16, -0.], [np.inf, 2., np.nan, 3.]])
    blocks = [[(np.arange(4), np.array([0, 2, 4])),
               (np.arange(4, 8), np.array([0, 2, 4]))]]
    plan = native._HistogramPlan(blocks, frame.shape, 2)
    exact(plan.accumulate(frame, -np.inf), prefix_reference(frame, blocks, -np.inf, 2))
    for pixels, cuts in [(np.array([-1]), np.array([0, 0, 1])),
                         (np.array([8]), np.array([0, 0, 1])),
                         (np.array([0, 1]), np.array([0, 2, 1]))]:
        with pytest.raises(ValueError):
            native._HistogramPlan([[(pixels, cuts)]], frame.shape, 2)
    with pytest.raises(ValueError):
        plan.accumulate(frame.T, 0.)


@pytest.mark.parametrize('threshold', [0., .5, 1.25, 1000.5, np.inf, np.nan])
def test_native_histogram_lossless_half_count_scale(threshold):
    frame = np.array([[0, 1, 2, 2001], [2**24+1, 2**30, 2**32-1, 3]], dtype='u4')
    blocks = [[(np.arange(8), np.array([0, 4, 8]))]]
    plan = native._HistogramPlan(blocks, frame.shape, 2)
    exact(plan.accumulate(frame, threshold, .5),
          prefix_reference(frame.astype('f8') * .5, blocks, threshold, 2))
    for scale in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match='scale'):
            plan.accumulate(frame, threshold, scale)


@pytest.mark.parametrize('scale', [1., .5])
@pytest.mark.parametrize('threshold', [0., .5, 1000., 2**32])
def test_sparse_wire_histogram_matches_reference_prefixes(scale, threshold):
    frame = np.resize(np.array([0, 1, 2000, 2001, 2**24+1, 2**32-1], dtype='u4'), (8, 8))
    blocks = [[(np.arange(32), np.array([0, 16, 32])),
               (np.arange(32, 64), np.array([0, 0, 32]))],
              [(np.arange(32), np.array([0, 16, 32]))]]
    plan = native._HistogramPlan(blocks, frame.shape, 2)
    pixels = np.flatnonzero(frame.ravel()).astype('u4')
    values = frame.ravel()[pixels]
    exact(plan.accumulate_sparse(pixels, values, threshold, scale),
          prefix_reference(frame.astype('f8')*scale, blocks, threshold, 2))
    with pytest.raises(IndexError):
        plan.accumulate_sparse(pixels[::-1].copy(), values[::-1].copy(), 0.)


def test_sparse_histogram_falls_back_for_noncanonical_prefix_geometry():
    values = np.array([2**32-1, 1], dtype='u4')
    pixels = np.arange(2, dtype='u4')
    for geometry in ([[(np.array([0, 0, 1]), np.array([0, 1, 3]))]],
                     [[(np.array([0, 0, 1]), np.array([0, 2, 3]))]],
                     [[(np.array([0, 1]), np.array([1, 1, 2]))]]):
        plan = native._HistogramPlan(geometry, (1, 2), 2)
        assert plan.accumulate_sparse(pixels, values, 0.) is None


def test_sparse_histogram_declines_inexact_mass():
    count = 2**21 + 1
    plan = native._HistogramPlan([[(np.arange(count), np.array([0, 0, count]))]], (1, count), 2)
    pixels = np.arange(count, dtype='u4')
    values = np.full(count, 2**32-1, dtype='u4')
    assert plan.accumulate_sparse(pixels, values, 0.) is None


def test_map_plan_owns_geometry_after_pixel_cache_release(tmp_path):
    from hexrd.core.instrument.planar_detector import _pixel_angles

    experiment, frames, _ = make_experiment(tmp_path)
    plan = EtaOmegaMapPlan(experiment, load_plane_data(experiment))
    expected = plan.histogram('panel0', frames['panel0'][0])
    _pixel_angles.cache_clear()
    assert _pixel_angles.cache_info()['currsize'] == 0
    exact(plan.histogram('panel0', frames['panel0'][0]), expected)


def test_ring_bound_is_safe_without_reordering_fractional_inputs():
    source = SimpleNamespace(dtype=np.dtype('u4'), shape=(3500, 3500))
    assert not _direct_sum_safe(source, 1)
    assert _direct_sum_safe(source, 1, pixel_count=335144)
    assert not _direct_sum_safe(source, 1, pixel_count=335144, frames_per_row=10)
    source.dtype = np.dtype('f8')
    assert not _direct_sum_safe(source, 1, pixel_count=1)


def make_experiment(tmp_path, dtype='f8', panels=1, gap=False):
    instrument = HEDMInstrument(max_workers=1)
    instrument.detectors.clear()
    series, frames = {}, {}
    edges = np.arange(5, dtype=float)
    omega = np.column_stack([edges[:-1], edges[1:]])
    if gap:
        omega[2:] += 1
    rng = np.random.default_rng(4)
    for i in range(panels):
        name = f'panel{i}'
        panel = PlanarDetector(rows=32, cols=40, pixel_size=(1., 1.),
                               tvec=np.array([8+i, 0., -100.]), name=name,
                               saturation_level=2**32 - 1)
        mask = np.ones(panel.shape, dtype=bool)
        mask[:, 18:21] = False
        panel.panel_buffer = mask
        instrument.detectors[name] = panel
        data = rng.integers(0, 100, size=(4, *panel.shape)).astype(dtype)
        if data.dtype.kind == 'f':
            data /= 2
        data[:, :, 18:21] = 65000
        frames[name] = data
        series[name] = imageseries.open(None, 'array', data=data, meta={'omega': omega})
    plane_data = Material().planeData
    plane_data.tThWidth = np.radians(1.)
    config = dict(analysis_name='test', working_dir=str(tmp_path), multiprocessing=1,
                  material=dict(active='test', tth_width=1.),
                  find_orientations=dict(orientation_maps=dict(threshold=2.5, eta_step=5.),
                                         clustering=dict(radius=0.5, completeness=0.9)))
    experiment = HedmExperiment.from_objects(instrument, series, plane_data, config)
    return experiment, frames, omega


@pytest.mark.parametrize('dtype', ['u2', 'u4', 'f4', 'f8'])
@pytest.mark.parametrize('panels,gap', [(1, False), (2, False), (2, True)])
def test_incremental_maps_equal_batch_and_preserve_snapshots(tmp_path, dtype, panels, gap):
    experiment, frames, omega = make_experiment(tmp_path, dtype, panels, gap)
    plane = load_plane_data(experiment)
    plan = EtaOmegaMapPlan(experiment, plane)
    accumulator = EtaOmegaMapAccumulator(plan, 4, filename=tmp_path / 'maps.npy')
    old = None
    for index in range(4):
        accumulator.append({name: data[index] for name, data in reversed(list(frames.items()))},
                           omega[index])
        if index == 1:
            snapshot = accumulator.snapshot()
            old = snapshot.ring_maps.tobytes()
    assert snapshot.ring_maps.tobytes() == old
    assert not snapshot.ring_maps.flags.writeable
    batch = build_eta_omega_maps(experiment, plane)
    actual = accumulator.snapshot()
    for name in ['ring_maps', 'eta_edges', 'omega_edges', 'omegas', 'omega_period',
                 'two_theta_ranges', 'ring_ids']:
        exact(getattr(actual, name), getattr(batch, name))
    assert actual.eta_step == batch.eta_step
    # The reference instrument independently bins the original input images.
    expected, _ = experiment.instrument.extract_polar_maps(
        experiment.get_active_material()._material.planeData, experiment.image_series,
        threshold=2.5, tth_tol=1., eta_tol=5.)
    values = np.array(list(expected.values()))
    reference = np.nansum(values, axis=0)
    reference[np.isnan(values).all(axis=0)] = np.nan
    if gap:
        expanded = np.full(batch.ring_maps.shape, np.nan)
        expanded[:, [0, 1, 3, 4]] = reference
        reference = expanded
    exact(batch.ring_maps, reference)


def test_incremental_rejects_excess_memory_and_failed_append_is_atomic(tmp_path):
    experiment, frames, omega = make_experiment(tmp_path)
    plan = EtaOmegaMapPlan(experiment, load_plane_data(experiment))
    with pytest.raises(MemoryError):
        EtaOmegaMapAccumulator(plan, 10000, max_size_mb=0.01)
    accumulator = EtaOmegaMapAccumulator(plan, 1)
    with pytest.raises(ValueError):
        accumulator.snapshot()
    with pytest.raises(ValueError):
        accumulator.append({}, omega[0])
    with pytest.raises(ValueError):
        accumulator.append({'panel0': frames['panel0'][0]}, [0., 361.])
    assert len(accumulator) == 0
    accumulator.append({'panel0': frames['panel0'][0]}, omega[0])
    with pytest.raises(ValueError):
        accumulator.append({'panel0': frames['panel0'][1]}, omega[1])
    assert len(accumulator) == 1
    with pytest.raises(MemoryError):
        EtaOmegaMapAccumulator(plan, 1, max_size_mb=0)


def test_supplied_maps_filter_a_copy_and_reject_wrong_axes(tmp_path):
    from dataclasses import replace
    from hexrd.hedm.find_orientations import _filter_maps, load_or_build_eta_omega_maps

    experiment, frames, omega = make_experiment(tmp_path)
    plane = load_plane_data(experiment)
    accumulator = EtaOmegaMapAccumulator(EtaOmegaMapPlan(experiment, plane), 4)
    for index in range(4):
        accumulator.append({'panel0': frames['panel0'][index]}, omega[index])
    raw = accumulator.snapshot()
    before = raw.ring_maps.tobytes()
    experiment.config['find_orientations']['orientation_maps']['filter_maps'] = True
    expected = replace(raw, ring_maps=raw.ring_maps.copy())
    _filter_maps(expected, True, None)
    filtered = load_or_build_eta_omega_maps(experiment, plane, maps=raw)
    exact(filtered.ring_maps, expected.ring_maps)
    assert raw.ring_maps.tobytes() == before
    with pytest.raises(ValueError, match='supplied maps'):
        load_or_build_eta_omega_maps(
            experiment, plane, maps=replace(raw, omega_edges=raw.omega_edges + 1))


def test_sparse_integer_ring_bound_matches_reference(tmp_path, monkeypatch):
    from hexrd.hedm.experiment import _AdapterImageSeries

    instrument = HEDMInstrument(max_workers=1)
    instrument.detectors.clear()
    panel = PlanarDetector(rows=1450, cols=1450, pixel_size=(1., 1.),
                           tvec=np.array([8., 0., -100.]), name='panel',
                           saturation_level=2**32-1)
    instrument.detectors['panel'] = panel
    frames = np.full((1, *panel.shape), 2**32-1, dtype='u4')
    series = imageseries.open(None, 'array', data=frames, meta={'omega': np.array([[0., 1.]])})
    experiment = HedmExperiment.from_objects(
        instrument, {'panel': series}, Material().planeData,
        dict(analysis_name='test', working_dir=str(tmp_path), multiprocessing=1,
             material=dict(active='test', tth_width=1.),
             find_orientations=dict(orientation_maps=dict(threshold=0, eta_step=5.),
                                    clustering=dict(radius=.5, completeness=.9))))
    monkeypatch.setattr(_AdapterImageSeries, 'dense_frames', property(lambda self: None))
    assert not _direct_sum_safe(experiment.image_series_list[0], 1)
    plane = load_plane_data(experiment)
    result = build_eta_omega_maps(experiment, plane)
    expected, _ = instrument.extract_polar_maps(
        experiment.get_active_material()._material.planeData, experiment.image_series,
        threshold=0, tth_tol=1., eta_tol=5.)
    exact(result.ring_maps, expected['panel'])


def test_gapped_complete_turn_keeps_the_final_omega_endpoint(tmp_path):
    experiment, frames, _ = make_experiment(tmp_path)
    omega = np.array([[0., 1.], [1., 2.], [358., 359.], [359., 360.]])
    experiment.image_series_list[0]._series.metadata['omega'][:] = omega
    plane = load_plane_data(experiment)
    accumulator = EtaOmegaMapAccumulator(EtaOmegaMapPlan(experiment, plane), 4)
    for i in range(4):
        accumulator.append({'panel0': frames['panel0'][i]}, omega[i])
    actual = accumulator.snapshot()
    expected = build_eta_omega_maps(experiment, plane)
    exact(actual.ring_maps, expected.ring_maps)
    assert actual.ring_maps.shape[1] == 360
    assert np.isnan(actual.ring_maps[:, 2:358]).all()
    assert actual.omega_edges[-1] == pytest.approx(2*np.pi)


def test_prepared_histograms_keep_source_threshold_and_panel_order(tmp_path):
    experiment, frames, omega = make_experiment(tmp_path, dtype='f4', panels=2)
    experiment.config['find_orientations']['orientation_maps']['threshold'] = 2.0000001
    plan = EtaOmegaMapPlan(experiment, load_plane_data(experiment))
    direct, prepared = [EtaOmegaMapAccumulator(plan, 4) for _ in range(2)]
    for i in range(4):
        images = {panel: data[i]*2 for panel, data in frames.items()}
        direct.append(images, omega[i])
        histograms = {panel: plan.histogram(panel, image.astype('u4'), source_dtype='f4')
                      for panel, image in reversed(list(images.items()))}
        prepared.append_histograms(histograms, omega[i])
    exact(prepared.snapshot().ring_maps, direct.snapshot().ring_maps)
