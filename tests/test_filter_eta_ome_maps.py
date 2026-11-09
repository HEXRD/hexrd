from types import SimpleNamespace

import numpy as np
import pytest

from hexrd.core import constants
from hexrd.hedm.find_orientations import _filter_maps


@pytest.mark.parametrize('filter_stdev', [False, True, 0.5, 2.0])
@pytest.mark.filterwarnings('ignore:All-NaN slice encountered:RuntimeWarning')
def test_filter_eta_ome_maps_preserves_unmeasured_bins(
    filter_stdev: bool | float,
) -> None:
    # Constant background, with peaks next to and away from the gap
    measured = np.full((31, 41), 12.0)
    measured[15, 19] = 112
    measured[15, 30] = 112
    masked = measured.copy()
    masked[:, 20] = np.nan
    masked[5, :] = np.nan
    masked[25, 10] = np.nan
    nan_mask = np.isnan(masked)

    reference = SimpleNamespace(ring_maps=[measured])
    eta_ome = SimpleNamespace(ring_maps=[masked, np.full_like(masked, np.nan)])
    log_fwhm = (float(filter_stdev) / constants.fwhm_to_sigma
                if filter_stdev else None)
    _filter_maps(reference, True, log_fwhm)
    _filter_maps(eta_ome, True, log_fwhm)

    # Missing background must not erase neighboring measurements or peaks
    np.testing.assert_array_equal(np.isnan(masked), nan_mask)
    np.testing.assert_allclose(masked[~nan_mask], measured[~nan_mask])
    assert masked[15, 19] > 0
    assert masked[15, 30] > 0
    assert np.isnan(eta_ome.ring_maps[1]).all()
