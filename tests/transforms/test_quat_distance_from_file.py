# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness

from __future__ import absolute_import
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import quat_distance  # , unit_vector
# from hexrd.rotations import quatOfLaueGroup


def test_quat_distance_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_quat_distance.npy',
        allow_pickle=True
    )

    for obj in arr:

        result = quat_distance(
            obj["q1"],
            obj["q2"],
            obj["q_sym"]
        )

        assert np.allclose(result, obj["result"])

# def test_correct_quat_distance(test_data_dir):
#     arr = [];
#     # Generate random xy_dets
#     for _ in range(40):
#         laueGroup = np.random.choice([
#             "Ci",
#             "C2h",
#             "D2h",
#             "C4h",
#             "D4h",
#             "C3i",
#             "D3d",
#             "C6h",
#             "D6h",
#             "Th",
#             "Oh",
#         ])
#         q1 = unit_vector(np.random.rand(4))
#         q2 = unit_vector(np.random.rand(4))
#         q_sym = quatOfLaueGroup(laueGroup)
#        
#
#         # Gener
#         result = quat_distance(
#             q1,
#             q2,
#             q_sym,
#         )
#         # Add the result to the array=
#         obj = {
#             "q1": q1,
#             "q2": q2,
#             "q_sym": q_sym,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_quat_distance.npy", arr)
