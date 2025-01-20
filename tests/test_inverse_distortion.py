import json

import numpy as np
from hexrd.core.extensions import inverse_distortion

RHO_MAX = 204.8
params = [
    -2.277777438488093e-05,
    -8.763805995117837e-05,
    -0.00047451698761967085,
]


def test_known_values():
    xy_in = np.array([[140.40087891, 117.74253845]])
    expected_output = np.array([[140.44540352, 117.77987754]])
    xy_out = inverse_distortion.ge_41rt_inverse_distortion(
        xy_in, RHO_MAX, params
    )
    assert np.allclose(xy_out, expected_output)


def test_large_input():
    xy_in = np.array([[1e5, 1e5]])
    xy_out = inverse_distortion.ge_41rt_inverse_distortion(
        xy_in, RHO_MAX, params
    )
    # No specific expected output here, just ensure it doesn't fail
    assert xy_out.shape == xy_in.shape


def test_logged_data(test_data_dir):
    with open(test_data_dir / 'inverse_distortion_in_out.json') as f:
        example_data = json.load(f)

    for example in example_data:
        xy_in = np.asarray(example['logged_inputs'])
        xy_out_expected = np.asarray(example['logged_outputs'])
        example_params = np.asarray(example['logged_params'])

        xy_out = inverse_distortion.ge_41rt_inverse_distortion(
            xy_in, RHO_MAX, example_params
        )
        assert np.allclose(xy_out, xy_out_expected, atol=1e-7)


def test_random_values():
    np.random.seed(42)
    xy_in = np.random.rand(10, 2) * 200
    xy_out = inverse_distortion.ge_41rt_inverse_distortion(
        xy_in, RHO_MAX, params
    )
    # Verify function doesn't raise any exceptions
    assert xy_out.shape == xy_in.shape
