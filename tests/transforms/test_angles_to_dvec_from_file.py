# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness

from __future__ import absolute_import
import numpy as np
from hexrd.core.transforms.new_capi.xf_new_capi import angles_to_dvec

# from common import random_rotation_matrix, random_unit_vectors


def test_angles_to_dvec_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_angles_to_dvec.npy', allow_pickle=True
    )

    for obj in arr:
        result = angles_to_dvec(
            obj["angs"],
            obj["beam_vec"],
            obj["eta_vec"],
            obj["chi"],
            obj["rmat_c"],
        )
        assert np.allclose(result, obj["result"])


# def test_correct_xy_to_dvec(test_data_dir):
#     arr = [];

#     for i in range(40):
#         k =  np.random.choice([2,3])
#         angs = np.random.rand(10, k) * 2 * np.pi
#         beam_vec = random_unit_vectors()
#         eta_vec = random_unit_vectors()
#         chi = np.random.rand() * np.pi/2
#         rmat_c = random_rotation_matrix()

#         result = angles_to_dvec(
#             angs,
#             beam_vec,
#             eta_vec,
#             chi,
#             rmat_c
#         )
#         # Add the result to the array
#         obj = {
#             "angs": angs,
#             "beam_vec": beam_vec,
#             "eta_vec": eta_vec,
#             "rmat_c": rmat_c,
#             "chi": chi,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_angles_to_dvec.npy", arr)
