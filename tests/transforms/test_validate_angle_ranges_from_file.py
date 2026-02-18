# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness

from __future__ import absolute_import
import numpy as np
from hexrd.core.transforms.xfcapi import validate_angle_ranges


def test_validate_angle_ranges_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_validate_angle_ranges.npy',
        allow_pickle=True,
    )
    for obj in arr:
        result = validate_angle_ranges(
            obj["angs_list"], obj["start_angs"], obj["stop_angs"], obj["ccw"]
        )
        assert np.all(result == obj["result"])

