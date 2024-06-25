# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness

from __future__ import absolute_import
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import validate_angle_ranges


def test_validate_angle_ranges_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_validate_angle_ranges.npy',
        allow_pickle=True
    )

    for obj in arr:

        result = validate_angle_ranges(
            obj["angs_list"],
            obj["start_angs"],
            obj["stop_angs"],
            obj["ccw"]
        )

        assert np.allclose(result, obj["result"])

# def test_correct_validate_angle_ranges(test_data_dir):
#     arr = [];
#     # Generate random xy_dets
#     for i in range(40):
#         angs_list = np.random.rand(20) * 2 * np.pi
#         start_angs = np.random.rand(2) * 2 * np.pi
#         stop_angs = np.random.rand(2) * 2 * np.pi
#         ccw = np.random.choice([True, False])
#         
#         
#         result = validate_angle_ranges(
#             angs_list,
#             start_angs,
#             stop_angs,
#             ccw
#         )
#         print(result)
#         # Add the result to the array=
#         obj = {
#             "angs_list": angs_list,
#             "start_angs": start_angs,
#             "stop_angs": stop_angs,
#             "ccw": ccw,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_validate_angle_ranges.npy", arr)
