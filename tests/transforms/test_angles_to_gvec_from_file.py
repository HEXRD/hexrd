# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness


from __future__ import absolute_import
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import angles_to_gvec


def test_angles_to_gvec_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_angles_to_gvec.npy',
        allow_pickle=True
    )

    for obj in arr:

        result = angles_to_gvec(
            obj["angs"],
            obj["beam_vec"],
            obj["eta_vec"],
            obj["chi"],
            obj["rmat_c"]
        )

        assert np.allclose(result, obj["result"])

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
#         k =  np.random.choice([2,3])
#         angs = np.random.rand(10, k) * 2 * np.pi
#         beam_vec = np.random.rand(3)
#         beam_vec /= np.linalg.norm(beam_vec)
#         eta_vec = np.random.rand(3)
#         eta_vec /= np.linalg.norm(eta_vec)
#         chi = np.random.rand() * np.pi/2
#         rmat_c = random_rotation_matrix()

#         # Gener
#         result = angles_to_gvec(
#             angs,
#             beam_vec,
#             eta_vec,
#             chi,
#             rmat_c
#         )
#         # Add the result to the array=
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
#     np.save("test_correct_angles_to_gvec.npy", arr)
