# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness

from __future__ import absolute_import
import numpy as np
from hexrd.core.transforms.new_capi.xf_new_capi import make_sample_rmat


def test_make_sample_rmat_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_make_sample_rmat.npy', allow_pickle=True
    )

    for obj in arr:

        result = make_sample_rmat(obj["chi"], obj["omega"])

        assert np.allclose(result, obj["result"])


# def test_correct_make_sample_rmat(test_data_dir):
#     arr = [];

#     for i in range(40):
#         if np.random.choice([True, False]):
#             omega = np.random.rand(10) * np.pi
#         else:
#             omega = np.random.rand() * np.pi
#         chi = np.random.rand() * np.pi

#         result = make_sample_rmat(
#             chi,
#             omega
#         )
#         # Add the result to the array
#         obj = {
#             "omega": omega,
#             "chi": chi,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_make_sample_rmat.npy", arr)
