from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from hexrd.core.instrument import HEDMInstrument
from hexrd.core.projections.polar import PolarView

from hexrd.core.cake import _builder as writer
from hexrd.core.cake import CakeOperator


@pytest.fixture
def small_instrument_config():
    return {
        'beam': {
            'energy': 65.0,
            'vector': {'azimuth': 90.0, 'polar_angle': 90.0},
        },
        'detectors': {
            'panel': {
                'pixels': {
                    'columns': 64,
                    'rows': 64,
                    'size': [0.2, 0.2],
                    'roi': [3, 5],
                },
                'saturation_level': 65535.0,
                'transform': {
                    'translation': [0.0, 0.0, -100.0],
                    'tilt': [0.0, 0.0, 0.0],
                },
            }
        },
        'oscillation_stage': {
            'chi': 0.0,
            'translation': [0.0, 0.0, 0.0],
        },
    }


def _apply_saved_operator(path, images):
    with h5py.File(path, 'r') as stream:
        root = stream['cake_operator']
        panel_names = root['input/panel_names'].asstr()[:]
        panel_shapes = root['input/panel_shapes'][:]
        active_indptr = root['input/active_pixel_indptr'][:]
        active_local = root['input/active_local_indices'][:]
        active = np.empty(len(active_local), dtype=np.float64)
        for i, (name, shape) in enumerate(zip(panel_names, panel_shapes)):
            start, stop = active_indptr[i : i + 2]
            image = np.asarray(images[name])
            assert image.shape == tuple(shape)
            active[start:stop] = image.ravel()[active_local[start:stop]]

        indptr = root['operator/output_indptr'][:]
        sources = root['operator/source_indices'][:]
        weights = root['operator/weights'][:]
        output = np.zeros(len(indptr) - 1)
        for row in range(len(output)):
            for group in range(indptr[row], indptr[row + 1]):
                source = sources[group]
                weight = weights[group]
                output[row] += (
                    weight[0] * active[source[0]]
                    + weight[1] * active[source[1]]
                    + weight[2] * active[source[2]]
                    + weight[3] * active[source[3]]
                )

        shape = tuple(root['grid/shape'][:])
        mask = (indptr[1:] == indptr[:-1]).reshape(shape)
        return np.ma.array(output.reshape(shape), mask=mask)


def test_export_is_compact_compressed_and_matches_polar_view(
    tmp_path, small_instrument_config
):
    instrument = HEDMInstrument(deepcopy(small_instrument_config))
    spec = writer.CakeOperatorSpec(
        tth_range=(0.0, 4.0),
        eta_range=(-180.0, 180.0),
        pixel_size=(0.2, 1.0),
    )
    output = tmp_path / 'small.hexrdcake'
    progress = []
    summary = writer.export_cake_operator(
        output,
        instrument,
        spec,
        eta_chunk_rows=7,
        metadata={'test_metadata': {'value': 1}},
        update_progress=progress.append,
    )
    writer.verify_cake_operator(output)

    assert summary.path == output
    assert summary.grid_shape == spec.shape
    assert summary.active_pixel_count < summary.full_panel_pixel_count
    assert summary.file_size == output.stat().st_size
    assert progress[-1] == 100
    assert progress == sorted(progress)

    with h5py.File(output, 'r') as stream:
        root = stream['cake_operator']
        assert root.attrs['format'] == writer.FORMAT_NAME
        assert root.attrs['format_version_major'] == 2
        assert 'output_mask' not in root['grid']
        assert root.attrs['representation'] == writer.REPRESENTATION
        assert root.attrs['compression'] == 'gzip-1'
        assert root.attrs['weight_precision'] == 'float64'
        assert root.attrs['mask_policy'] == (
            'drop-four-corner-group-if-any-source-invalid'
        )
        assert root['instrument/detectors/panel/pixels/rows'][0] == 64
        assert root['input_adapters/composite/frame_shape'][:].tolist() == [67, 69]

        for name in (
            'operator/output_indptr',
            'operator/source_indices',
            'operator/weights',
        ):
            dataset = root[name]
            assert dataset.compression == 'gzip'
            assert dataset.compression_opts == 1
            assert dataset.shuffle
            assert dataset.fletcher32
            assert dataset.chunks is not None

        assert root['operator/weights'].dtype == np.dtype(np.float64)
        assert root['operator/source_indices'].dtype == np.dtype(np.uint32)
        assert root['operator/output_indptr'].dtype == np.dtype(np.uint32)
        assert root['input/active_pixel_indptr'].dtype == np.dtype(np.uint64)
        assert root['input/active_local_indices'].dtype == np.dtype(np.uint32)

    rng = np.random.default_rng(0)
    images = {'panel': rng.random((64, 64))}
    actual = _apply_saved_operator(output, images)
    polar = PolarView(
        spec.tth_range,
        instrument,
        eta_min=spec.eta_range[0],
        eta_max=spec.eta_range[1],
        pixel_size=spec.pixel_size,
        cache_coordinate_map=True,
    )
    expected = polar.warp_image(images, pad_with_nans=False)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(
        actual.data[~actual.mask], expected.data[~expected.mask]
    )
    operator = CakeOperator.load(output, verify_digest=True)
    applied = operator.apply(images)
    np.testing.assert_array_equal(applied, actual)


def test_static_invalid_pixels_are_not_stored(tmp_path, small_instrument_config):
    panel_buffer = np.ones((64, 64), dtype=bool)
    panel_buffer[:, 32:] = False
    small_instrument_config['detectors']['panel']['buffer'] = panel_buffer
    instrument = HEDMInstrument(small_instrument_config)
    spec = writer.CakeOperatorSpec(
        tth_range=(0.0, 4.0),
        eta_range=(-180.0, 180.0),
        pixel_size=(0.2, 5.0),
    )
    output = tmp_path / 'masked.hexrdcake'
    writer.export_cake_operator(output, instrument, spec)

    with h5py.File(output, 'r') as stream:
        root = stream['cake_operator']
        active = root['input/active_local_indices'][:]
        _, columns = np.divmod(active, 64)
        assert np.all(columns < 32)


def test_known_panel_buffer_is_written_symbolically(monkeypatch):
    buffer = np.array([[True, False], [True, True]])
    panel = SimpleNamespace(panel_buffer=buffer)
    instrument = SimpleNamespace(
        detectors={'panel': panel},
        write_config=lambda style: {'detectors': {'panel': {'buffer': buffer.copy()}}},
    )
    monkeypatch.setattr(writer, 'valid_panel_buffer_names', lambda: ['known-eiger'])
    monkeypatch.setattr(
        writer, 'panel_buffer_from_str', lambda name, candidate_panel: buffer
    )

    config, replacements = writer._normalized_instrument_config(instrument)

    assert config['detectors']['panel']['buffer'] == 'known-eiger'
    assert replacements == {'panel': 'known-eiger'}


def test_export_is_atomic_on_failure(tmp_path, small_instrument_config, monkeypatch):
    instrument = HEDMInstrument(small_instrument_config)
    spec = writer.CakeOperatorSpec(
        tth_range=(0.0, 2.0),
        eta_range=(-10.0, 10.0),
        pixel_size=(0.2, 1.0),
    )
    output = tmp_path / 'existing.hexrdcake'
    output.write_bytes(b'original')

    def fail(*args, **kwargs):
        raise RuntimeError('expected failure')

    monkeypatch.setattr(writer, '_write_operator', fail)
    with pytest.raises(RuntimeError, match='expected failure'):
        writer.export_cake_operator(output, instrument, spec, overwrite=True)

    assert output.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [output]


@pytest.fixture
def built_operator(tmp_path, small_instrument_config):
    instrument = HEDMInstrument(deepcopy(small_instrument_config))
    spec = writer.CakeOperatorSpec((0, 4), (-180, 180), (0.2, 5))
    path = tmp_path / 'operator.hexrdcake'
    return CakeOperator.build(path, instrument, spec, eta_chunk_rows=7), path


def test_roundtrip_batch_and_regional_composite(built_operator, tmp_path):
    operator, path = built_operator
    frame = np.arange(67 * 69, dtype=np.float64).reshape(67, 69)
    images = {'panel': frame[3:67, 5:69]}  # Noncontiguous ROI; no full-panel copy.
    expected = operator.apply(images)
    np.testing.assert_array_equal(operator.apply_composite(frame), expected)
    active = operator.gather_active_pixels(images)
    batch = operator.apply_active_pixels(np.column_stack([active, 2 * active]))
    np.testing.assert_array_equal(batch[..., 0], expected)
    np.testing.assert_array_equal(batch[..., 1], expected * 2)
    assert operator._instrument_cache is None

    regional = CakeOperator.load(
        path, eta_slice=slice(2, 7), compact_input=True, verify_digest=True
    )
    assert regional.active_pixel_count < operator.active_pixel_count
    np.testing.assert_array_equal(regional.apply_composite(frame), expected[2:7])
    saved = tmp_path / 'regional.hexrdcake'
    regional.save(saved)
    reloaded = CakeOperator.load(saved, verify_digest=True)
    np.testing.assert_array_equal(reloaded.apply_composite(frame), expected[2:7])
    assert reloaded.instrument.detectors['panel'].shape == (64, 64)
    with pytest.raises(FileExistsError):
        regional.save(saved)
    regional.save(saved, overwrite=True)
    CakeOperator.load(saved, verify_digest=True)


@pytest.mark.parametrize(
    ('dataset', 'value', 'message'),
    [
        (
            'operator/output_indptr',
            lambda a: np.concatenate(([1], a[1:])),
            'start at zero',
        ),
        (
            'operator/output_indptr',
            lambda a: np.concatenate((a[:5], [a[-1]], a[6:])),
            'monotonic',
        ),
        ('operator/source_indices', lambda a: np.full_like(a, 2**32 - 1), 'exceed'),
        ('operator/source_indices', lambda a: a.astype(float) + 0.5, 'integers'),
        (
            'input/active_pixel_indptr',
            lambda a: np.array([1, a[-1]], dtype=a.dtype),
            'start at zero',
        ),
        (
            'input/active_local_indices',
            lambda a: np.full_like(a, 64 * 64),
            'panel shape',
        ),
        ('operator/weights', lambda a: np.full_like(a, np.nan), 'finite'),
        ('grid/tth_edges', lambda a: np.full_like(a, np.nan), 'finite'),
    ],
)
def test_reject_invalid_stored_arrays(built_operator, dataset, value, message):
    _, path = built_operator
    with h5py.File(path, 'r+') as stream:
        root = stream['cake_operator']
        changed = value(root[dataset][:])
        del root[dataset]
        root.create_dataset(dataset, data=changed)
    with pytest.raises(ValueError, match=message):
        CakeOperator.load(path)


def test_feature_and_digest_checks(built_operator):
    _, path = built_operator
    with h5py.File(path, 'r+') as stream:
        root = stream['cake_operator']
        root.attrs['required_features_json'] = '["future-required"]'
    with pytest.raises(ValueError, match='unknown required'):
        CakeOperator.load(path)
    with h5py.File(path, 'r+') as stream:
        root = stream['cake_operator']
        root.attrs['required_features_json'] = '["core-v1", "bilinear-output-block-v1"]'
        root['instrument/beam/energy'][0] = 66
    with pytest.raises(ValueError, match='instrument digest'):
        CakeOperator.load(path, verify_digest=True)


def test_save_failure_preserves_destination(built_operator, monkeypatch):
    operator, path = built_operator
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError('serialization failure')

    monkeypatch.setattr(operator, '_save_file', fail)
    with pytest.raises(RuntimeError, match='serialization'):
        operator.save(path, overwrite=True)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_input_validation_and_nan_propagation(built_operator):
    operator, path = built_operator
    with pytest.raises(ValueError, match='panel mismatch'):
        operator.apply({})
    with pytest.raises(ValueError, match='shape'):
        operator.apply({'panel': np.ones((32, 32))})
    with pytest.raises(ValueError, match='masked inputs'):
        operator.apply({'panel': np.ma.ones((64, 64))})
    with pytest.raises(ValueError, match='real numeric'):
        operator.apply_active_pixels(
            np.ones(operator.active_pixel_count, dtype=complex)
        )
    with pytest.raises(ValueError, match='at least one row'):
        CakeOperator.load(path, eta_slice=slice(2, 2))
    active = np.ones(operator.active_pixel_count)
    source = int(operator.source_indices[0, 0])
    active[source] = np.nan
    result = operator.apply_active_pixels(active)
    affected = np.zeros(operator.grid.size, dtype=bool)
    for row in range(operator.grid.size):
        start, stop = operator.output_indptr[row : row + 2]
        affected[row] = np.any(operator.source_indices[start:stop] == source)
    np.testing.assert_array_equal(np.isnan(result.data).ravel(), affected)


def test_overlapping_panels_preserve_accumulation_order(
    tmp_path, small_instrument_config
):
    config = deepcopy(small_instrument_config)
    config['detectors']['overlap'] = deepcopy(config['detectors']['panel'])
    instrument = HEDMInstrument(config)
    spec = writer.CakeOperatorSpec((0, 4), (-180, 180), (0.2, 5))
    operator = CakeOperator.build(tmp_path / 'overlap.hexrdcake', instrument, spec)
    rng = np.random.default_rng(32)
    images = {name: rng.random((64, 64)) for name in instrument.detectors}
    expected = PolarView(
        spec.tth_range,
        instrument,
        eta_min=-180,
        eta_max=180,
        pixel_size=spec.pixel_size,
    ).warp_image(images)
    actual = operator.apply(images)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(actual.compressed(), expected.compressed())


def test_empty_operator_and_grid_validation(tmp_path, small_instrument_config):
    config = deepcopy(small_instrument_config)
    config['detectors']['panel']['buffer'] = np.zeros((64, 64), dtype=bool)
    spec = writer.CakeOperatorSpec((0, 4), (-180, 180), (0.2, 5))
    operator = CakeOperator.build(
        tmp_path / 'empty.hexrdcake', HEDMInstrument(config), spec
    )
    assert operator.active_pixel_count == 0
    assert operator.apply({'panel': np.ones((64, 64))}).count() == 0
    for bad in [np.nan, np.inf, 0, -1]:
        with pytest.raises(ValueError):
            writer.CakeOperatorSpec((0, 4), (-180, 180), (bad, 5))


def test_legacy_mask_migrates_without_changing_valid_outputs(built_operator, tmp_path):
    operator, path = built_operator
    mask = operator.grid.output_mask.copy()
    mask.ravel()[np.flatnonzero(~mask.ravel())[0]] = True
    with h5py.File(path, 'r+') as stream:
        root = stream['cake_operator']
        root.attrs['format_version_major'] = 1
        root.attrs['required_features_json'] = '["core-v1", "bilinear-output-block-v1"]'
        root['grid'].create_dataset('output_mask', data=mask)
        root.attrs['payload_sha256'] = writer._payload_digest(root)
    legacy = CakeOperator.load(path, verify_digest=True)
    images = {'panel': np.arange(64 * 64).reshape(64, 64)}
    expected = legacy.apply(images)
    destination = tmp_path / 'migrated.hexrdcake'
    original_groups = legacy.group_count
    legacy.save(destination)
    assert legacy.group_count == original_groups
    with h5py.File(destination) as stream:
        assert 'output_mask' not in stream['cake_operator/grid']
    migrated = CakeOperator.load(destination, verify_digest=True)
    actual = migrated.apply(images)
    assert migrated.group_count < original_groups
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(actual.compressed(), expected.compressed())
