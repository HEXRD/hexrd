from types import SimpleNamespace

import numpy as np

from hexrd.core import imageseries
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.core.instrument import Detector, HEDMInstrument
from hexrd.core.material import Material
from hexrd.core.material.crystallography import PlaneData
from hexrd.core.utils.panel_buffer import panel_buffer_as_2d_array

GAP_VALUE = np.iinfo(np.uint32).max
SIGNAL_VALUE = 1000


def make_instrument() -> tuple[HEDMInstrument, str, Detector, PlaneData]:
    instr = HEDMInstrument(max_workers=1)
    det_key = next(iter(instr.detectors))
    panel = instr.detectors[det_key]
    panel.saturation_level = 2**16
    plane_data = Material().planeData
    plane_data.tThWidth = np.radians(0.25)
    return instr, det_key, panel, plane_data


def make_omega_imageseries(
    frames: np.ndarray, omega_start: float = 0, delta_omega: float = 1
) -> OmegaImageSeries:
    edges = omega_start + delta_omega * np.arange(len(frames) + 1)
    omegas = np.column_stack([edges[:-1], edges[1:]])
    ims = imageseries.open(None, 'array', data=frames, meta={'omega': omegas})
    return OmegaImageSeries(ims)


def test_numeric_panel_buffers() -> None:
    expected = np.array([[True, False], [False, True]])
    for dtype in (bool, np.uint8, float):
        panel = SimpleNamespace(panel_buffer=expected.astype(dtype))
        actual = panel_buffer_as_2d_array(panel)
        assert actual.dtype == bool
        np.testing.assert_array_equal(actual, expected)


def test_extract_polar_maps_respects_panel_buffer() -> None:
    instr, det_key, panel, plane_data = make_instrument()
    frames = np.ones((1, *panel.shape), dtype=np.uint32)
    frames[:, :, 1000:1010] = GAP_VALUE
    buffer = np.ones(panel.shape, dtype=bool)
    buffer[:, 1000:1010] = False

    def maps(threshold: float | None = None) -> np.ndarray:
        result, _ = instr.extract_polar_maps(
            plane_data, {det_key: make_omega_imageseries(frames)}, threshold=threshold
        )
        return result[det_key]

    unbuffered = maps()
    assert np.nanmax(unbuffered) >= GAP_VALUE
    panel.panel_buffer = buffer
    buffered = maps()
    missing = np.isnan(buffered)
    assert missing.sum() > np.isnan(unbuffered).sum()
    assert 0 < np.nanmax(buffered) < GAP_VALUE

    # Thresholds remove signal without changing which bins were measured.
    thresholded = maps(threshold=2)
    np.testing.assert_array_equal(np.isnan(thresholded), missing)
    assert np.nanmax(thresholded) == 0

    # The measured intensities must match independently zeroed gap pixels.
    frames[:, :, 1000:1010] = 0
    panel.panel_buffer = None
    np.testing.assert_array_equal(buffered[~missing], maps()[~missing])
    panel.panel_buffer = np.zeros(panel.shape, dtype=bool)
    assert np.isnan(maps()).all()


def test_pull_spots_respects_panel_buffer() -> None:
    instr, det_key, panel, plane_data = make_instrument()
    grain = np.r_[np.zeros(6), [1, 1, 1, 0, 0, 0]]
    sim = instr.simulate_rotation_series(plane_data, [grain])
    _, _, angs, xys, _ = [item[0] for item in sim[det_key]]
    i, j = panel.cartToPixel(xys[:1], pixels=True)[0]
    omega_start = np.degrees(angs[0, 2]) - 8

    # A real peak with a saturated gap pixel inside the same patch.
    frames = np.zeros((4, *panel.shape), dtype=np.uint32)
    frames[:, i, j] = SIGNAL_VALUE
    frames[:, i + 2, j + 1] = GAP_VALUE
    buffer = np.ones(panel.shape, dtype=bool)
    buffer[i + 2, j + 1] = False

    def pull(threshold: float, check_only: bool = False) -> tuple[list[bool], dict]:
        return instr.pull_spots(
            plane_data,
            grain,
            {det_key: make_omega_imageseries(frames, omega_start, 4)},
            threshold=threshold,
            ome_tol=8,
            check_only=check_only,
            ome_period=np.radians([omega_start, omega_start + 360]),
        )

    for check_only in (False, True):
        panel.panel_buffer = None
        assert any(pull(2 * SIGNAL_VALUE, check_only)[0])
        panel.panel_buffer = buffer
        detected, output = pull(10, check_only)
        assert any(detected)
        if not check_only:
            assert {row[4] for row in output[det_key] if row[0] >= 0} == {SIGNAL_VALUE}
        assert not any(pull(2 * SIGNAL_VALUE, check_only)[0])

    # Zeroing a gap must not turn it into signal above a negative threshold.
    frames = np.full(frames.shape, -10, dtype=np.int16)
    frames[:, i + 2, j + 1] = SIGNAL_VALUE
    for check_only in (False, True):
        panel.panel_buffer = None
        assert any(pull(-1, check_only)[0])
        panel.panel_buffer = buffer
        assert not any(pull(-1, check_only)[0])
