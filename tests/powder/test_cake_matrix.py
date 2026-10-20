from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from hexrd.core.instrument import HEDMInstrument, PlanarDetector
from hexrd.core.projections.polar import PolarView
from hexrd.powder.cake import CakeMatrix, export_cake_matrix

GRID = dict(tth_range=(0, 3), eta_range=(-180, 180), pixel_size=(0.2, 10))


def assert_same_cake(actual: np.ma.MaskedArray, expected: np.ma.MaskedArray) -> None:
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_allclose(actual.compressed(), expected.compressed(), atol=1e-14)


@pytest.fixture
def instrument() -> HEDMInstrument:
    instrument = HEDMInstrument()
    instrument.detectors.clear()
    # Overlapping geometry, disjoint raw ROIs; insertion order differs from sort order.
    for name, column in [('z', 3), ('a', 35)]:
        instrument.detectors[name] = PlanarDetector(
            rows=32,
            cols=32,
            pixel_size=(0.2, 0.2),
            tvec=np.array([0.0, 0.0, -100.0]),
            roi=(2, column),
            saturation_level=65535,
        )
    return instrument


@pytest.mark.parametrize('composite', [True, False])
def test_matrix_matches_polar_view(
    tmp_path: Path, instrument: HEDMInstrument, composite: bool
) -> None:
    if not composite:
        for panel in instrument.detectors.values():
            panel.roi = None
    instrument.beam_dict['secondary'] = deepcopy(instrument.active_beam)
    instrument.beam_dict['secondary']['energy'] = 80.0
    instrument.active_beam_name = 'secondary'
    path = tmp_path / 'matrix.hexrdcake'
    export_cake_matrix(path, instrument, **GRID, eta_chunk_rows=7)
    with h5py.File(path) as stream:
        assert stream.attrs['version'] == 'hexrd-cake-matrix-1'
    matrix = CakeMatrix.load(path)
    assert matrix.instrument.active_beam_name == 'secondary'
    assert matrix.instrument.beam_energy == 80.0
    frame = np.random.default_rng(123).random((34, 67))
    images = {'z': frame[2:34, 3:35], 'a': frame[2:34, 35:67]}
    expected = PolarView(
        (0, 3), instrument, eta_min=-180, eta_max=180, pixel_size=(0.2, 10)
    ).warp_image(images)
    actual = matrix.apply(frame if composite else images)
    assert_same_cake(actual, expected)
    assert_same_cake(matrix.apply(images), expected)
    # The stored CSR also works directly on raw pixels in the documented order.
    pixels = (
        frame.ravel()
        if composite
        else np.concatenate([images[name].ravel() for name in sorted(images)])
    )
    np.testing.assert_array_equal(matrix.matrix @ pixels, actual.data.ravel())


def test_eta_views_share_storage(tmp_path: Path, instrument: HEDMInstrument) -> None:
    path = tmp_path / 'matrix.hexrdcake'
    export_cake_matrix(path, instrument, **GRID)
    matrix = CakeMatrix(path)
    frame = np.random.default_rng(123).random(matrix.input_shape)
    expected = matrix.apply(frame)
    for rows in (slice(0, 0), slice(0, 7), slice(7, None)):
        wedge = matrix[rows]
        assert_same_cake(wedge.apply(frame), expected[rows])
        np.testing.assert_allclose(wedge.eta_centers, matrix.eta_centers[rows])
        if wedge.matrix.nnz:
            assert np.shares_memory(wedge.matrix.data, matrix.matrix.data)
            assert np.shares_memory(wedge.matrix.indices, matrix.matrix.indices)


def test_composite_layout_rejects_incompatible_rois(
    tmp_path: Path, instrument: HEDMInstrument
) -> None:
    path = tmp_path / 'matrix.hexrdcake'
    panel = instrument.detectors['a']
    for roi, group in (((2, 3), None), ((2, 35), 'other')):
        panel.roi = roi
        panel.group = group
        with pytest.raises(ValueError, match='nonoverlapping ROIs in one detector group'):
            export_cake_matrix(path, instrument, **GRID, input_layout='composite')
        assert not path.exists()

        # Automatic layout keeps both images independent and sums their signal.
        export_cake_matrix(path, instrument, **GRID)
        matrix = CakeMatrix(path)
        assert matrix.input_layout == 'panels'
        cake = matrix.apply({'z': np.ones((32, 32)), 'a': np.full((32, 32), 3)})
        np.testing.assert_allclose(cake.compressed(), 4)
        # The reader must reject an incompatible saved layout too.
        with h5py.File(path, 'r+') as stream:
            stream['matrix'].attrs['input_layout'] = 'composite'
        with pytest.raises(ValueError, match='nonoverlapping ROIs in one detector group'):
            CakeMatrix(path)
        path.unlink()


def test_buffered_export_is_independent_of_chunk_size(
    tmp_path: Path, instrument: HEDMInstrument
) -> None:
    panel = instrument.detectors['z']
    buffer = np.ones(panel.shape, dtype=bool)
    buffer[10:20, 10:20] = False
    panel.panel_buffer = buffer.copy()
    frame = np.ones((34, 67))
    frame[12:22, 13:23] = np.nan  # Buffered pixels in full-image coordinates.
    small, whole = tmp_path / 'small.hexrdcake', tmp_path / 'whole.hexrdcake'
    export_cake_matrix(small, instrument, **GRID, eta_chunk_rows=7)
    export_cake_matrix(whole, instrument, **GRID, eta_chunk_rows=36)
    matrix, whole_matrix = CakeMatrix(small), CakeMatrix(whole)
    for name in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(
            getattr(matrix.matrix, name), getattr(whole_matrix.matrix, name)
        )
    actual = matrix.apply(frame)
    # The other panel still contributes where z is buffered; NaNs stay excluded.
    assert np.isfinite(actual.compressed()).all()
    reference = PolarView(
        (0, 3), instrument, eta_min=-180, eta_max=180, pixel_size=(0.2, 10)
    ).warp_image({'z': frame[2:34, 3:35], 'a': frame[2:34, 35:67]})
    # Where interpolation touches a NaN, only the second panel contributes 1.
    reference.data[np.isnan(reference.data)] = 1
    assert_same_cake(actual, reference)
    np.testing.assert_array_equal(matrix.instrument.detectors['z'].panel_buffer, buffer)
    np.testing.assert_array_equal(panel.panel_buffer, buffer)


def test_cancelled_export_preserves_existing_file(
    tmp_path: Path, instrument: HEDMInstrument
) -> None:
    path = tmp_path / 'matrix.hexrdcake'
    path.write_bytes(b'previous file')

    def cancel(progress: int) -> None:
        raise RuntimeError('cancelled')

    with pytest.raises(RuntimeError, match='cancelled'):
        export_cake_matrix(
            path, instrument, **GRID, overwrite=True, update_progress=cancel
        )
    with pytest.raises(ValueError, match='eta_chunk_rows'):
        export_cake_matrix(path, instrument, **GRID, overwrite=True, eta_chunk_rows=-1)
    assert path.read_bytes() == b'previous file'
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_grid_is_rejected(tmp_path: Path, instrument: HEDMInstrument) -> None:
    path = tmp_path / 'matrix.hexrdcake'
    for invalid in (
        dict(tth_range=(0, 0.01)),  # Rounds to zero two-theta bins.
        dict(eta_range=(0, 0.01)),  # Rounds to zero eta bins.
        dict(tth_range=(0, np.nan)),
        dict(eta_range=(-np.inf, 180)),
        dict(pixel_size=(np.nan, 10)),
        dict(pixel_size=(0.2, np.inf)),
    ):
        with pytest.raises(ValueError, match='grid requires'):
            export_cake_matrix(path, instrument, **(GRID | invalid))
        assert not path.exists()
    export_cake_matrix(path, instrument, **GRID)
    with h5py.File(path, 'r+') as f:
        f['grid/tth'][2] += 0.5  # Must not silently truncate the saved bin count.
    with pytest.raises(ValueError, match='grid requires'):
        CakeMatrix(path)


@pytest.mark.parametrize(
    'name, corruption',
    [
        ('data', 'nan'),
        ('indices', 'fraction'),
        ('indptr', 'fraction'),
        ('indptr', 'short-tail'),
    ],
)
def test_corrupt_csr_is_rejected(tmp_path, instrument, name, corruption):
    path = tmp_path / 'matrix.hexrdcake'
    export_cake_matrix(path, instrument, **GRID)
    with h5py.File(path, 'r+') as stream:
        matrix = stream['matrix']
        values = matrix[name][:]
        if corruption == 'nan':
            values[0] = np.nan
        elif corruption == 'fraction':
            values = values.astype(float)
            values[0] += 0.5
        else:
            # SciPy silently trims unreferenced coefficients without this check.
            values = np.minimum(values, values[-1] - 4)
        del matrix[name]
        matrix[name] = values
    with pytest.raises(ValueError, match='CSR arrays'):
        CakeMatrix(path)
