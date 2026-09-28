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
        assert set(f) == {'instrument', 'grid', 'input', 'matrix'}
        assert set(f['grid']) == {'tth', 'eta'}
        assert set(f['input']) == {'panel_offsets', 'pixel_indices'}
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
    # Actual image dimensions supply the stride: ROI extents are only a minimum.
    padded = np.pad(frame, ((0, 5), (0, 9)))
    np.testing.assert_array_equal(operator.apply_composite(padded), actual)
    np.testing.assert_array_equal(
        operator.apply_composite(padded[:, ::-1][:, ::-1]), actual
    )
    active = operator.gather_active_pixels(images)
    batch = operator.apply_active_pixels(np.column_stack([active, 2 * active]))
    np.testing.assert_allclose(batch[..., 1].compressed(), actual.compressed() * 2)
    assert operator.instrument.detectors['a'].shape == (32, 32)


@pytest.mark.parametrize(
    ('dataset', 'transform', 'match'),
    [
        ('grid/tth', lambda a: np.array([0.0, 0.2, 3.5]), 'integer count'),
        ('grid/eta', lambda a: np.array([0.0, np.nan, 36.0]), 'finite'),
        ('input/panel_offsets', lambda a: a + 1, 'panel_offsets'),
        ('input/pixel_indices', lambda a: np.full_like(a, 2000), 'inside each panel'),
        ('matrix/indices', lambda a: a.astype(float), 'integers'),
        ('matrix/indices', lambda a: np.full_like(a, 2**30), 'compact inputs'),
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
    assert operator.matrix.shape[1] == 0
    assert operator.apply_composite(np.ones((34, 67))).count() == 0


def test_invalid_inputs_and_nan_propagation(archive):
    operator = CakeMatrix.load(archive[0])
    with pytest.raises(ValueError, match='panel names'):
        operator.apply({})
    with pytest.raises(ValueError, match='too small'):
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
