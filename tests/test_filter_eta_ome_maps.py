from types import SimpleNamespace

import numpy as np
import pytest

from hexrd.hedm.findorientations import _filter_eta_ome_maps


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

    reference = SimpleNamespace(dataStore=[measured])
    eta_ome = SimpleNamespace(dataStore=[masked, np.full_like(masked, np.nan)])
    _filter_eta_ome_maps(reference, filter_stdev)
    _filter_eta_ome_maps(eta_ome, filter_stdev)

    # Missing background must not erase neighboring measurements or peaks
    np.testing.assert_array_equal(np.isnan(masked), nan_mask)
    np.testing.assert_allclose(masked[~nan_mask], measured[~nan_mask])
    assert masked[15, 19] > 0
    assert masked[15, 30] > 0
    assert np.isnan(eta_ome.dataStore[1]).all()
