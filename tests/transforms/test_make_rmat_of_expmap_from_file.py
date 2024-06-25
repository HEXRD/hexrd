# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness


from __future__ import absolute_import
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import make_rmat_of_expmap


def test_make_rmat_of_expmap_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_make_rmat_of_expmap.npy',
        allow_pickle=True
    )

    for obj in arr:

        result = make_rmat_of_expmap(
            obj["expmap"]
        )

        assert np.allclose(result, obj["result"])

# def test_correct_make_sample_rmat(test_data_dir):
#     arr = [];
#     # Generate random xy_dets
#     for i in range(40):
#         expmap = np.random.rand(3)

#         result = make_rmat_of_expmap(
#             expmap
#         )
#         # Add the result to the array=
#         obj = {
#             "expmap": expmap,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_make_rmat_of_expmap.npy", arr)
