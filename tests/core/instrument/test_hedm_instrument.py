from pathlib import Path

import h5py
import numpy as np

from hexrd.core.instrument import HEDMInstrument, PlanarDetector
from hexrd.core.instrument.hedm_instrument import (
    _generate_ring_params,
    _run_histograms,
)


def test_run_histograms_does_not_overflow_uint16_weights():
    ptth = np.full((1, 2), 0.1)
    peta = np.zeros_like(ptth)
    eta_edges = np.array([-0.5, 0.5])
    tth_ranges = [(0.05, 0.15)]
    ring_params = [
        _generate_ring_params(tth_ranges[0], ptth, peta, eta_edges, 1.0)
    ]

    images = [np.full((1, 2), 40_000, dtype=np.uint16)]
    ring_maps = np.full((1, 1, 1), np.nan)
    _run_histograms((0, 1), images, tth_ranges, ring_maps, ring_params, None)

    assert ring_maps[0, 0, 0] == 80_000


def test_hdf5_instrument_round_trip(tmp_path: Path) -> None:
    instrument = HEDMInstrument()
    instrument.detectors.clear()
    instrument.detectors['panel'] = PlanarDetector(
        rows=2, cols=3, saturation_level=65535
    )
    buffer = np.array([[True, False, True], [False, True, True]])
    instrument.detectors['panel'].panel_buffer = buffer
    instrument.beam_dict['secondary'] = dict(instrument.active_beam, energy=80)
    instrument.active_beam_name = 'secondary'
    path = tmp_path / 'instrument.h5'
    instrument.write_config(str(path), style='hdf5', compression='gzip', fletcher32=True)
    with h5py.File(path, 'r+') as stream:
        assert stream['instrument/detectors/panel/buffer'].compression == 'gzip'
        # Loading must only read the instrument, not unrelated archive contents.
        stream['unrelated'] = h5py.ExternalLink('missing.h5', '/data')
        assert HEDMInstrument(stream['instrument']).beam_energy == 80
        restored = HEDMInstrument(stream)
        assert restored.active_beam_name == 'secondary'
        np.testing.assert_array_equal(restored.detectors['panel'].panel_buffer, buffer)
        primary = next(iter(instrument.beam_dict))
        assert HEDMInstrument(stream, active_beam_name=primary).active_beam_name == primary

    # A renamed single beam must not acquire an extra default beam on reload.
    del instrument.beam_dict[primary]
    instrument.write_config(str(path), style='hdf5')
    with h5py.File(path) as stream:
        single = HEDMInstrument(stream)
    assert single.beam_names == ['secondary']
    assert single.beam_energy == 80
