# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness of xy_to_gvec

from __future__ import absolute_import
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import xy_to_gvec


def test_xy_to_gvec_from_file():
    # Load the array from a file
    arr = np.load(
        "tests/transforms/data/test_correct_xy_to_gvec.npy", allow_pickle=True
    )
    for obj in arr:
        result = xy_to_gvec(
            obj["xy_det"],
            obj["rmat_d"],
            obj["rmat_s"],
            obj["tvec_d"],
            obj["tvec_s"],
            obj["tvec_c"],
            obj["rmat_b"],
        )
        assert np.allclose(result[0], obj["result"][0])
        assert np.allclose(result[1], obj["result"][1])


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

# def test_correct_xy_to_gvec():
#     arr = [];
#     # Generate random xy_dets
#     for i in range(40):
#         xy_det = np.random.rand(4, 2)
#         rmat_d = random_rotation_matrix()
#         rmat_s = random_rotation_matrix()
#         tvec_d = np.random.rand(3)
#         tvec_s = np.random.rand(3)
#         tvec_c = np.random.rand(3)
#         rmat_b = random_rotation_matrix()
#         # Gener
#         result = xy_to_gvec(xy_det,
#                             rmat_d,
#                             rmat_s,
#                             tvec_d,
#                             tvec_s,
#                             tvec_c,
#                             rmat_b)
#         # Add the result to the array=
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
#     # Save the array to a file
#     np.save("tests/transforms//data/test_correct_xy_to_gvec.npy", arr)
