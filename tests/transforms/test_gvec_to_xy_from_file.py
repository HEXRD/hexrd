# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness of gvec_to_xy

from __future__ import absolute_import
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import gvec_to_xy

def test_gvec_to_xy_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_gvec_to_xy.npy',
        allow_pickle=True
    )
    for obj in arr:
        result = gvec_to_xy(obj["gvec_c"],
                            obj["rmat_d"],
                            obj["rmat_s"],
                            obj["rmat_c"],
                            obj["tvec_d"],
                            obj["tvec_s"],
                            obj["tvec_c"],
                            obj["beam_vec"])
        # Ignore nans
        mask = ~(np.isnan(obj["result"]))
        assert np.allclose(result[mask], obj["result"][mask])
        # Make sure unmasked stuff is all nan
        assert np.isnan(result[~mask]).all()

# def random_rotation_matrix():
#     # Generate a random unit quaternion
#     q = np.random.rand(4)
#     q /= np.linalg.norm(q)
#     q0 = q[0]
#     q1 = q[1]
#     q2 = q[2]
#     q3 = q[3]
     
#     # First row of the rotation matrix
#     r00 = 2 * (q0 * q0 + q1 * q1) - 1
#     r01 = 2 * (q1 * q2 - q0 * q3)
#     r02 = 2 * (q1 * q3 + q0 * q2)
     
#     # Second row of the rotation matrix
#     r10 = 2 * (q1 * q2 + q0 * q3)
#     r11 = 2 * (q0 * q0 + q2 * q2) - 1
#     r12 = 2 * (q2 * q3 - q0 * q1)
     
#     # Third row of the rotation matrix
#     r20 = 2 * (q1 * q3 - q0 * q2)
#     r21 = 2 * (q2 * q3 + q0 * q1)
#     r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
#     # 3x3 rotation matrix
#     rot_matrix = np.array([[r00, r01, r02],
#                            [r10, r11, r12],
#                            [r20, r21, r22]])
#     return rot_matrix

# def test_correct_xy_to_gvec(test_data_dir):
#     arr = [];
#     # Generate random xy_dets
#     for i in range(40):
#         gvec_c = np.random.rand(10, 3)
#         rmat_d = random_rotation_matrix()
#         rmat_s = [random_rotation_matrix() for j in range(10)]
#         rmat_c = random_rotation_matrix()
#         tvec_d = np.random.rand(3)
#         tvec_s = np.random.rand(3)
#         tvec_c = np.random.rand(3)
#         beam_vec = np.random.rand(3)
#         # Gener
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