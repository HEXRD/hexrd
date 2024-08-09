# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness of xy_to_gvec

from __future__ import absolute_import
import numpy as np
from hexrd.transforms.xfcapi import xy_to_gvec
# from common import random_rotation_matrix, random_unit_vectors


def test_xy_to_gvec_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_xy_to_gvec.npy',
        allow_pickle=True
    )

    for obj in arr:
        result = xy_to_gvec(
            obj["xy_det"],
            obj["rmat_d"],
            obj["rmat_s"],
            obj["tvec_d"],
            obj["tvec_s"],
            obj["tvec_c"],
            obj["rmat_b"]
        )

        assert np.allclose(result[0], obj["result"][0])
        assert np.allclose(result[1], obj["result"][1])


# def test_correct_xy_to_gvec(test_data_dir):
#     arr = []
#     # Generate random xy_dets
#     for i in range(40):
#         xy_det = np.random.rand(4, 2)
#         rmat_d = random_rotation_matrix()
#         rmat_s = random_rotation_matrix()
#         tvec_d = np.random.rand(3) * 2 - 1
#         tvec_s = np.random.rand(3) * 2 - 1
#         tvec_c = np.random.rand(3) * 2 - 1
#         rmat_b = random_rotation_matrix()

#         result = xy_to_gvec(xy_det,
#                             rmat_d,
#                             rmat_s,
#                             tvec_d,
#                             tvec_s,
#                             tvec_c,
#                             rmat_b)
#         # Add the result to the array
#         obj = {
#             "xy_det": xy_det,
#             "rmat_d": rmat_d,
#             "rmat_s": rmat_s,
#             "tvec_d": tvec_d,
#             "tvec_s": tvec_s,
#             "tvec_c": tvec_c,
#             "rmat_b": rmat_b,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_xy_to_gvec.npy", arr)
