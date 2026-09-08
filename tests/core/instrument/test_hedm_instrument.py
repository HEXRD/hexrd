import numpy as np

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
