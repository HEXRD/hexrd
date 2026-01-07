from pathlib import Path
import pytest

import numpy as np

from hexrd.core import imageseries

# NOTE: this is in a separate format than the `test_formats.py` file
# so that we can utilize the pytest fixtures.


pytest.importorskip(
    "dectris.compression",
    reason="dectris.compression (Eiger decompression) is not installed",
)

def test_format_eiger_stream_v2(example_repo_path: Path):
    filepath = (
        example_repo_path
        / 'eiger/eiger_stream_v2/eiger_stream_v2_test_dataset.h5'
    )
    fmt = 'eiger-stream-v2'
    ims = imageseries.open(filepath, fmt)

    # Verify the metadata is there
    assert 'start' in ims.metadata
    assert 'output_metadata' in ims.metadata

    ims.set_option('threshold_setting', 'threshold_1')
    assert ims.option_values()['threshold_setting'] == 'threshold_1'
    array1 = ims[0]

    ims_shape = ims.shape
    assert ims_shape == (4362, 4148)

    # Verify that the array looks valid
    assert array1.shape == ims_shape
    assert array1.dtype == ims.dtype
    assert array1.mean() > 0
    assert np.std(array1) > 0

    ims.set_option('threshold_setting', 'threshold_2')
    assert ims.option_values()['threshold_setting'] == 'threshold_2'
    array2 = ims[0]

    # Verify that the array looks valid
    assert array2.shape == ims_shape
    assert array2.dtype == ims.dtype
    assert array2.mean() > 0
    assert np.std(array2) > 0

    # This should also be different from array1
    assert not np.allclose(array1, array2)

    multiplier = 0.75
    ims.set_option('threshold_setting', 'man_diff')
    ims.set_option('multiplier', multiplier)
    assert ims.option_values()['threshold_setting'] == 'man_diff'
    assert ims.option_values()['multiplier'] == multiplier

    diff_array = ims[0]

    # Verify that the array looks valid
    assert diff_array.shape == ims_shape
    assert diff_array.dtype == ims.dtype
    assert diff_array.mean() > 0
    assert np.std(diff_array) > 0

    # This should be different than the two earlier arrays
    assert not np.allclose(array1, diff_array)
    assert not np.allclose(array2, diff_array)

    # In fact, it should be equal to the following formula:
    expected_diff_array = array1.astype(np.float64) - multiplier * array2
    assert np.allclose(diff_array, expected_diff_array)
