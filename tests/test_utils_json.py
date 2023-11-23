import json

import numpy as np

from hexrd.utils.json import NumpyDecoder, NumpyEncoder


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
