"""Panel buffers must be respected when histogramming eta-omega maps and
when pulling spots, so that e.g. saturated Eiger subpanel gap pixels do
not register as diffraction signal. See HEXRD/hexrd#959.
"""

import numpy as np
from numpy.typing import NDArray
import pytest

from hexrd.core import imageseries
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.core.instrument import Detector, HEDMInstrument
from hexrd.core.material import Material
from hexrd.core.material.crystallography import PlaneData

GAP_VALUE = np.iinfo(np.uint32).max
SIGNAL_VALUE = 1000


@pytest.fixture
def instrument() -> HEDMInstrument:
    return HEDMInstrument(max_workers=1)


@pytest.fixture
def det_key(instrument: HEDMInstrument) -> str:
    return next(iter(instrument.detectors))


@pytest.fixture
def panel(instrument: HEDMInstrument, det_key: str) -> Detector:
    panel = instrument.detectors[det_key]
    panel.saturation_level = 2**16
    return panel


@pytest.fixture
def plane_data() -> PlaneData:
    pd = Material().planeData
    pd.tThWidth = np.radians(0.25)
    return pd


def make_omega_imageseries(frames: np.ndarray, omega_start: float,
                           delta_omega: float) -> OmegaImageSeries:
    nf = len(frames)
    edges = omega_start + delta_omega * np.arange(nf + 1)
    omegas = np.column_stack([edges[:-1], edges[1:]])
    ims = imageseries.open(None, 'array', data=frames, meta={'omega': omegas})
    return OmegaImageSeries(ims)


def buffer_excluding(shape: tuple[int, int], index: tuple) -> NDArray[np.bool_]:
    # A panel buffer that is valid everywhere except at index
    buffer = np.ones(shape, dtype=bool)
    buffer[index] = False
    return buffer


def test_extract_polar_maps_respects_panel_buffer(
    instrument: HEDMInstrument,
    det_key: str,
    panel: Detector,
    plane_data: PlaneData,
) -> None:
    # A vertical stripe of saturated pixels (as in Eiger subpanel gaps),
    # plus one pixel of genuine signal on the first ring
    gap_cols = slice(1000, 1010)
    ptth, _ = panel.pixel_angles()
    tth0 = plane_data.getTTh()[0]
    signal_ij = np.unravel_index(np.argmin(np.abs(ptth - tth0)), ptth.shape)
    assert not gap_cols.start <= signal_ij[1] < gap_cols.stop

    frames = np.zeros((4, *panel.shape), dtype=np.uint32)
    frames[:, :, gap_cols] = GAP_VALUE
    frames[:, signal_ij[0], signal_ij[1]] = SIGNAL_VALUE
    oms = make_omega_imageseries(frames, 0, 1)

    # Without a buffer, the gap pixels dominate the maps
    panel.panel_buffer = None
    maps, _ = instrument.extract_polar_maps(plane_data, {det_key: oms})
    unbuffered = maps[det_key]
    assert np.nanmax(unbuffered) > SIGNAL_VALUE

    panel.panel_buffer = buffer_excluding(panel.shape, (slice(None), gap_cols))
    maps, _ = instrument.extract_polar_maps(plane_data, {det_key: oms})
    buffered = maps[det_key]

    # Gap pixels no longer contribute, but real signal still does
    assert np.nanmax(buffered) == SIGNAL_VALUE

    # Eta bins lying entirely within the gap become NaN (off-detector)
    # rather than reading as measured zeros
    assert np.isnan(buffered).sum() > np.isnan(unbuffered).sum()


def test_pull_spots_respects_panel_buffer(
    instrument: HEDMInstrument,
    det_key: str,
    panel: Detector,
    plane_data: PlaneData,
) -> None:
    delta_omega = 4.0
    ome_tol = 8.0
    gparams = np.hstack([np.zeros(6), [1, 1, 1, 0, 0, 0]])
    sim = instrument.simulate_rotation_series(
        plane_data,
        [gparams],
        ome_ranges=[np.radians([0, 360])],
        ome_period=np.radians([0, 360]),
    )
    _, _, valid_angs, valid_xys, _ = [x[0] for x in sim[det_key]]
    pix = panel.cartToPixel(valid_xys, pixels=True)
    omes = np.degrees(valid_angs[:, 2])

    # Saturated "gap" pixel inside the first spot's patch, offset from
    # the patch center and corners so patch selection is unaffected
    i, j = pix[0]
    gap_ij = (i + 2, j + 1)

    # A second predicted spot at a nearby omega but a distant pixel
    # receives genuine signal
    candidates = np.where(
        (np.abs(omes - omes[0]) < 12)
        & (np.hypot(*(pix - pix[0]).T) > 100)
    )[0]
    assert candidates.size > 0
    signal_ij = tuple(pix[candidates[0]])

    # Frame range covering both spots' omega windows
    ome_start = omes[0] - 16
    frames = np.zeros((8, *panel.shape), dtype=np.uint32)
    frames[:, gap_ij[0], gap_ij[1]] = GAP_VALUE
    frames[:, signal_ij[0], signal_ij[1]] = SIGNAL_VALUE
    oms = make_omega_imageseries(frames, ome_start, delta_omega)

    kwargs = {
        'threshold': 10,
        'ome_tol': ome_tol,
        'ome_period': np.radians([ome_start, ome_start + 360]),
        'filename': None,
    }

    def peak_maxes(output: dict[str, list]) -> list[float]:
        # rows are [peak_id, hkl_id, hkl, sum_int, max_int, ...], with
        # negative peak_ids marking patches where no peak was found
        return [row[4] for row in output[det_key] if row[0] >= 0]

    panel.panel_buffer = None
    compl, output = instrument.pull_spots(
        plane_data, gparams, {det_key: oms}, **kwargs
    )
    assert GAP_VALUE in peak_maxes(output)

    panel.panel_buffer = buffer_excluding(panel.shape, gap_ij)
    compl2, output2 = instrument.pull_spots(
        plane_data, gparams, {det_key: oms}, **kwargs
    )

    # Same patches are extracted, but the gap pixel no longer registers
    # as a peak, while the genuine signal still does
    assert len(compl2) == len(compl)
    assert set(peak_maxes(output2)) == {SIGNAL_VALUE}
