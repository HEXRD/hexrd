from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from hexrd.core.instrument import HEDMInstrument, PlanarDetector
from hexrd.core.projections.polar import PolarView
from hexrd.powder.cake import CakeMatrix, export_cake_matrix

GRID = dict(tth_range=(0, 3), eta_range=(-180, 180), pixel_size=(0.2, 10))


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
    matrix = CakeMatrix.load(path)
    assert matrix.instrument.active_beam_name == 'secondary'
    assert matrix.instrument.beam_energy == 80.0
    frame = np.random.default_rng(123).random((34, 67))
    images = {'z': frame[2:34, 3:35], 'a': frame[2:34, 35:67]}
    expected = PolarView(
        (0, 3), instrument, eta_min=-180, eta_max=180, pixel_size=(0.2, 10)
    ).warp_image(images)
    actual = matrix.apply(frame if composite else images)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_allclose(actual.compressed(), expected.compressed(), atol=1e-14)
    np.testing.assert_allclose(
        matrix.apply(images).filled(np.nan), actual.filled(np.nan), equal_nan=True
    )
    chunks = np.ma.concatenate(
        [
            matrix.apply(images, rows)
            for rows in (slice(0, 0), slice(0, 7), slice(7, None))
        ]
    )
    np.testing.assert_array_equal(chunks.mask, actual.mask)
    np.testing.assert_allclose(chunks.data, actual.data)
    # The stored CSR also works directly on raw pixels in the documented order.
    pixels = (
        frame.ravel()
        if composite
        else np.concatenate([images[name].ravel() for name in sorted(images)])
    )
    np.testing.assert_array_equal(matrix.matrix @ pixels, actual.data.ravel())


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
    a, b = CakeMatrix.load(small), CakeMatrix.load(whole)
    actual, expected = a.apply(frame), b.apply(frame)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(actual.data, expected.data)
    # The other panel still contributes where z is buffered; NaNs stay excluded.
    assert np.isfinite(actual.compressed()).all()
    reference = PolarView(
        (0, 3), instrument, eta_min=-180, eta_max=180, pixel_size=(0.2, 10)
    ).warp_image({'z': frame[2:34, 3:35], 'a': frame[2:34, 35:67]})
    # Where interpolation touches a NaN, only the second panel contributes 1.
    reference.data[np.isnan(reference.data)] = 1
    np.testing.assert_array_equal(actual.mask, reference.mask)
    np.testing.assert_allclose(actual.compressed(), reference.compressed())
    np.testing.assert_array_equal(a.instrument.detectors['z'].panel_buffer, buffer)
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
    assert path.read_bytes() == b'previous file'
    assert list(tmp_path.iterdir()) == [path]
