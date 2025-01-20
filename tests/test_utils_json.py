import json

import numpy as np

from hexrd.core.utils.json import NumpyDecoder, NumpyEncoder, NumpyToNativeEncoder


def test_decode_encode():
    to_test = [
        {
            'floating': np.arange(50, dtype=np.float16),
            'complex': np.arange(20, dtype=np.complex128),
        },
        {
            'nested': {
                'int8': np.arange(27, dtype=np.int8).reshape((3, 3, 3)),
                'uint8': np.arange(8, dtype=np.uint8).reshape((2, 4)),
                'not_numpy': 3,
            }
        },
        np.array([0, 5, 4]),
        5,
        'string',
    ]

    output = json.dumps(to_test, cls=NumpyEncoder)
    converted_back = json.loads(output, cls=NumpyDecoder)

    assert _json_equal(to_test, converted_back)


def test_numpy_to_native():
    to_test = {
        'inside': np.arange(27, dtype=np.int8).reshape((3, 3, 3)),
        'nested': {
            'float': np.arange(4, dtype=np.float32).reshape((2, 2)),
        },
        'float': np.float64(8.3),
    }

    inside_to_list = to_test['inside'].tolist()
    nested_float_to_value = to_test['nested']['float'].tolist()
    float_to_value = to_test['float'].item()

    encoded = json.dumps(to_test, cls=NumpyToNativeEncoder)
    output = json.loads(encoded)

    assert (
        isinstance(output['inside'], list) and
        output['inside'] == to_test['inside'].tolist()
    )
    assert (
        isinstance(output['float'], float) and
        output['float'] == to_test['float'].item()
    )
    assert (
        isinstance(output['nested']['float'], list) and
        output['nested']['float'] == to_test['nested']['float'].tolist()
    )


def _json_equal(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b) and a.dtype == b.dtype

    if isinstance(a, dict):
        if list(a) != list(b):
            return False

        for k in a:
            return _json_equal(a[k], b[k])

    if isinstance(a, list):
        if len(a) != len(b):
            return False

        for i in range(len(a)):
            return _json_equal(a[i], b[i])

    return a == b
