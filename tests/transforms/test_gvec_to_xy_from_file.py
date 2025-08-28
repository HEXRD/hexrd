# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness of gvec_to_xy

from __future__ import absolute_import
import numpy as np
from hexrd.core.transforms.new_capi.xf_new_capi import gvec_to_xy

# from common import random_rotation_matrix, random_unit_vectors


def test_gvec_to_xy_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_gvec_to_xy.npy', allow_pickle=True
    )
    for obj in arr:
        result = gvec_to_xy(
            obj["gvec_c"],
            obj["rmat_d"],
            obj["rmat_s"],
            obj["rmat_c"],
            obj["tvec_d"],
            obj["tvec_s"],
            obj["tvec_c"],
            obj["beam_vec"],
        )
        assert np.allclose(result, obj["result"], equal_nan=True)


# def test_correct_gvec_to_xy(test_data_dir):
#     arr = [];

#     for i in range(40):
#         gvec_c = random_unit_vectors(10,3)
#         rmat_d = random_rotation_matrix()
#         rmat_s = [random_rotation_matrix() for j in range(10)]
#         rmat_c = random_rotation_matrix()
#         tvec_d = np.random.rand(3) * 2 - 1
#         tvec_s = np.random.rand(3) * 2 - 1
#         tvec_c = np.random.rand(3) * 2 - 1
#         beam_vec = random_unit_vectors()

#         result = gvec_to_xy(gvec_c,
#                             rmat_d,
#                             rmat_s,
#                             rmat_c,
#                             tvec_d,
#                             tvec_s,
#                             tvec_c,
#                             beam_vec)
#         # Add the result to the array=
#         obj = {
#             "gvec_c": gvec_c,
#             "rmat_d": rmat_d,
#             "rmat_s": rmat_s,
#             "rmat_c": rmat_c,
#             "tvec_d": tvec_d,
#             "tvec_s": tvec_s,
#             "tvec_c": tvec_c,
#             "beam_vec": beam_vec,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_gvec_to_xy.npy", arr)
