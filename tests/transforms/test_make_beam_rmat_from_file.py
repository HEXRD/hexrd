# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness


from __future__ import absolute_import
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import make_beam_rmat


def test_make_beam_rmat_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_make_beam_rmat.npy',
        allow_pickle=True
    )

    for obj in arr:

        result = make_beam_rmat(
            obj["bvec_l"],
            obj["evec_l"]
        )

        assert np.allclose(result, obj["result"])

# def test_make_beam_rmat(test_data_dir):
#     arr = [];
#     # Generate random xy_dets
#     for i in range(40):
#         bvec_l = np.random.rand(3)
#         evec_l = np.random.rand(3)
#         bvec_l /= np.linalg.norm(bvec_l)
#         evec_l /= np.linalg.norm(evec_l)
#         # Gener
#         result = make_beam_rmat(
#             bvec_l,
#             evec_l
#         )
#         # Add the result to the array=
#         obj = {
#             "bvec_l": bvec_l,
#             "evec_l": evec_l,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_make_beam_rmat.npy", arr)