from copy import deepcopy

import h5py
import numpy as np
import pytest

from hexrd.core.cake import CakeMatrix, CakeOperatorSpec, export_cake_matrix
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.projections.polar import PolarView


@pytest.fixture
def instrument():
    panel = {
        'pixels': {'columns': 32, 'rows': 32, 'size': [0.2, 0.2], 'roi': [2, 3]},
        'transform': {'translation': [0.0, 0.0, -100.0], 'tilt': [0.0, 0.0, 0.0]},
    }
    second = deepcopy(panel)
    second['pixels']['roi'] = [2, 35]
    return HEDMInstrument(
        {
            'beam': {'energy': 65.0, 'vector': {'azimuth': 90.0, 'polar_angle': 90.0}},
            'detectors': {'z': panel, 'a': second},
            'oscillation_stage': {'chi': 0.0, 'translation': [0.0, 0.0, 0.0]},
        }
    )


@pytest.fixture
def archive(tmp_path, instrument):
    path = tmp_path / 'test.hexrdcake'
    spec = CakeOperatorSpec((0, 3), (-180, 180), (0.2, 10))
    summary = export_cake_matrix(path, instrument, spec, eta_chunk_rows=7)
    return path, spec, summary


def test_simple_csr_export_and_execution(archive, instrument):
    path, spec, summary = archive
    with h5py.File(path) as f:
        assert set(f) == {'instrument', 'grid', 'matrix'}
        assert set(f['grid']) == {'tth', 'eta'}
        assert f['matrix'].attrs['input_layout'] == 'composite'
        assert f.attrs['version'] == 'draft-csr-2'
        assert set(f['matrix']) == {'data', 'indices', 'indptr'}
        assert f['matrix'].attrs['format'] == 'csr'
    operator = CakeMatrix.load(path)
    assert operator.panel_names == ('a', 'z')  # Different from build order.
    assert operator._instrument is None
    assert summary.file_size == path.stat().st_size
    frame = np.random.default_rng(123).random((34, 67))
    images = {
        name: frame[p.roi[0][0] : p.roi[0][1], p.roi[1][0] : p.roi[1][1]]
        for name, p in instrument.detectors.items()
    }
    actual = operator.apply(images)
    expected = PolarView(
        spec.tth_range,
        instrument,
        eta_min=-180,
        eta_max=180,
        pixel_size=spec.pixel_size,
    ).warp_image(images)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_allclose(actual.compressed(), expected.compressed(), atol=1e-14)
    np.testing.assert_array_equal(operator.apply_composite(frame), actual)
    assert operator.composite_shape == frame.shape
    assert operator.matrix_for_composite(frame.shape) is operator.matrix
    np.testing.assert_array_equal(
        (operator.matrix @ frame.ravel()).reshape(operator.shape), actual.data
    )
    with pytest.raises(ValueError, match='frame shape'):
        operator.apply_composite(np.pad(frame, ((0, 5), (0, 9))))
    noncontiguous = np.asfortranarray(frame)
    np.testing.assert_array_equal(operator.apply_composite(noncontiguous), actual)
    active = operator.gather_active_pixels(images)
    batch = operator.apply_active_pixels(np.column_stack([active, 2 * active]))
    np.testing.assert_allclose(batch[..., 1].compressed(), actual.compressed() * 2)
    assert operator.instrument.detectors['a'].shape == (32, 32)


@pytest.mark.parametrize(
    ('dataset', 'transform', 'match'),
    [
        ('grid/tth', lambda a: np.array([0.0, 0.2, 3.5]), 'integer count'),
        ('grid/eta', lambda a: np.array([0.0, np.nan, 36.0]), 'finite'),
        ('matrix/indices', lambda a: a.astype(float), 'integers'),
        ('matrix/indices', lambda a: np.full_like(a, 2**30), 'valid inputs'),
        ('matrix/indptr', lambda a: a[::-1], 'monotonically'),
        ('matrix/data', lambda a: np.full_like(a, np.nan), 'finite'),
    ],
)
def test_invalid_archives_rejected(archive, dataset, transform, match):
    path, _, _ = archive
    with h5py.File(path, 'r+') as f:
        value = transform(f[dataset][:])
        del f[dataset]
        f[dataset] = value
    with pytest.raises(ValueError, match=match):
        CakeMatrix.load(path)


def test_cancelled_conversion_is_atomic(archive, instrument):
    path, spec, _ = archive
    original = path.read_bytes()

    def cancel(value):
        if value >= 90:
            raise RuntimeError('cancelled')

    with pytest.raises(RuntimeError, match='cancelled'):
        export_cake_matrix(
            path, instrument, spec, overwrite=True, update_progress=cancel
        )
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    with pytest.raises(FileExistsError):
        export_cake_matrix(path, instrument, spec)


def test_empty_matrix_and_static_buffers(tmp_path, instrument):
    for panel in instrument.detectors.values():
        panel.panel_buffer = np.zeros(panel.shape, dtype=bool)
    path = tmp_path / 'empty.hexrdcake'
    spec = CakeOperatorSpec((0, 3), (-180, 180), (0.2, 10))
    export_cake_matrix(path, instrument, spec)
    operator = CakeMatrix.load(path)
    assert operator.matrix.shape[1] == 34 * 67
    assert operator.matrix.nnz == 0
    assert operator.apply_composite(np.ones((34, 67))).count() == 0


def test_invalid_inputs_and_nan_propagation(archive):
    operator = CakeMatrix.load(archive[0])
    with pytest.raises(ValueError, match='panel names'):
        operator.apply({})
    with pytest.raises(ValueError, match='frame shape'):
        operator.apply_composite(np.ones((30, 60)))
    with pytest.raises(ValueError, match='masked inputs'):
        operator.apply_composite(np.ma.ones((34, 67)))
    with pytest.raises(ValueError, match='shape'):
        operator.apply_active_pixels(np.ones(2))
    x = np.ones(operator.matrix.shape[1])
    x[operator.matrix.indices[0]] = np.nan
    actual = operator.apply_active_pixels(x)
    expected = (operator.matrix @ x).reshape(operator.shape)
    np.testing.assert_array_equal(actual.data, expected)


def test_separate_panels_without_rois(tmp_path, instrument):
    for panel in instrument.detectors.values():
        panel.roi = None
    spec = CakeOperatorSpec((0, 3), (-180, 180), (0.2, 10))
    path = tmp_path / 'panels.hexrdcake'
    export_cake_matrix(path, instrument, spec)
    operator = CakeMatrix.load(path)
    assert operator.input_layout == 'panels'
    images = {
        name: np.random.default_rng(i).random(p.shape)
        for i, (name, p) in enumerate(instrument.detectors.items())
    }
    x = np.concatenate([images[name].ravel() for name in sorted(images)])
    np.testing.assert_array_equal(
        operator.apply(images).data.ravel(), operator.matrix @ x
    )
    expected = PolarView(
        spec.tth_range,
        instrument,
        eta_min=-180,
        eta_max=180,
        pixel_size=spec.pixel_size,
    ).warp_image(images)
    actual = operator.apply(images)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_allclose(actual.compressed(), expected.compressed(), atol=1e-14)
    with pytest.raises(ValueError, match='separate panel'):
        operator.apply_composite(np.ones((32, 64)))


def test_single_panel_without_roi(tmp_path, instrument):
    del instrument.detectors['z']
    instrument.detectors['a'].roi = None
    path = tmp_path / 'single.hexrdcake'
    export_cake_matrix(
        path, instrument, CakeOperatorSpec((0, 3), (-180, 180), (0.2, 10))
    )
    operator = CakeMatrix.load(path)
    assert operator.input_layout == 'composite'
    assert operator.composite_shape == (32, 32)
    frame = np.arange(1024).reshape(32, 32)
    np.testing.assert_array_equal(
        operator.apply_composite(frame), operator.apply({'a': frame})
    )


def test_input_layout_validation(archive):
    path = archive[0]
    with h5py.File(path, 'r+') as f:
        f['matrix'].attrs['input_layout'] = 'unknown'
    with pytest.raises(ValueError, match='input_layout'):
        CakeMatrix.load(path)


def test_legacy_compact_file_still_loads(archive):
    path = archive[0]
    # Re-express this file as a minimal legacy compact archive.
    with h5py.File(path, 'r+') as f:
        indices = f['matrix/indices'][:]
        # Legacy input segments follow sorted panels a then z.
        all_indices = []
        offsets = [0]
        raw_order = []
        for name in sorted(f['instrument/detectors']):
            p = f['instrument/detectors'][name]['pixels']
            rows, cols = (
                int(np.asarray(p['rows'][()]).item()),
                int(np.asarray(p['columns'][()]).item()),
            )
            r0, c0 = p['roi'][:]
            local = np.arange(rows * cols)
            rr, cc = np.divmod(local, cols)
            raw_order.extend((r0 + rr) * 67 + c0 + cc)
            all_indices.extend(local)
            offsets.append(len(all_indices))
        inverse = np.zeros(34 * 67, dtype=int)
        inverse[raw_order] = np.arange(len(raw_order))
        f['matrix/indices'][:] = inverse[indices]
        f['matrix'].attrs['shape'] = (f['matrix'].attrs['shape'][0], len(all_indices))
        f.attrs['version'] = 'draft-csr-1'
        f['input/panel_offsets'] = offsets
        f['input/pixel_indices'] = all_indices
    operator = CakeMatrix.load(path)
    frame = np.random.default_rng(7).random((34, 67))
    active = operator.gather_active_composite_pixels(frame)
    expected = operator.apply_active_pixels(active)
    actual = operator.apply_composite(frame)
    np.testing.assert_array_equal(actual, expected)
    C = operator.matrix_for_composite(frame.shape)
    assert C is operator.matrix_for_composite(frame.shape)
    # Legacy adapters can also support padded raw frames without changing files.
    padded = np.pad(frame, ((0, 5), (0, 9)))
    np.testing.assert_array_equal(operator.apply_composite(padded), expected)


def test_multiple_detector_groups_use_separate_inputs(tmp_path, instrument):
    instrument.detectors['a'].group = 'one'
    instrument.detectors['z'].group = 'two'
    path = tmp_path / 'groups.hexrdcake'
    export_cake_matrix(
        path, instrument, CakeOperatorSpec((0, 3), (-180, 180), (0.2, 10))
    )
    assert CakeMatrix.load(path).input_layout == 'panels'


def test_overlapping_rois_default_to_separate_panels(tmp_path, instrument):
    instrument.detectors['a'].roi = (2, 3)
    path = tmp_path / 'overlap.hexrdcake'
    export_cake_matrix(
        path, instrument, CakeOperatorSpec((0, 3), (-180, 180), (0.2, 10))
    )
    operator = CakeMatrix.load(path)
    assert operator.input_layout == 'panels'
    assert operator.matrix.shape[1] == 2 * 32 * 32


def test_full_image_shape_and_nan_checks(archive):
    operator = CakeMatrix.load(archive[0])
    with pytest.raises(ValueError, match='frame shape'):
        operator.apply_composite(np.ones((67, 34)))  # Same size, wrong stride.
    frame = np.ones(operator.composite_shape)
    frame.ravel()[operator.matrix.indices[0]] = np.nan
    result = operator.apply_composite(frame)
    np.testing.assert_array_equal(result.data.ravel(), operator.matrix @ frame.ravel())
    with h5py.File(archive[0], 'r+') as f:
        rows, cols = f['matrix'].attrs['shape']
        f['matrix'].attrs['shape'] = (rows, cols + 1)
    with pytest.raises(ValueError, match='instrument input layout'):
        CakeMatrix.load(archive[0])
