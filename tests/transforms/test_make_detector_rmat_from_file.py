# Generated random test cases from commented out code
# Save the test cases to a file
# Loading to test correctness


from __future__ import absolute_import
import numpy as np
from hexrd.core.transforms.xfcapi import make_detector_rmat


def test_make_detector_rmat_from_file(test_data_dir):
    # Load the array from a file
    arr = np.load(
        test_data_dir / 'test_correct_make_detector_rmat.npy',
        allow_pickle=True,
    )

    for obj in arr:

        result = make_detector_rmat(obj["tilt_angles"])

        assert np.allclose(result, obj["result"])


# def test_correct_make_detector_rmat(test_data_dir):
#     arr = [];

#     for i in range(40):
#         tilt_angles = np.random.rand(3)

#         result = make_detector_rmat(
#             tilt_angles
#         )
#         # Add the result to the array=
#         obj = {
#             "tilt_angles": tilt_angles,
#             "result": result
#         }
#         arr.append(obj)
#     # Save the array to a file, move it to the tests/data folder
#     np.save(test_data_dir / "test_correct_make_detector_rmat.npy", arr)
