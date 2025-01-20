import numpy as np
import yaml

from hexrd.core.utils.yaml import NumpyToNativeDumper


def test_numpy_to_native():
    to_test = {
        'inside': np.arange(27, dtype=np.int8).reshape((3, 3, 3)),
        'nested': {
            'float16': np.arange(4, dtype=np.float16).reshape((2, 2)),
        },
        'float32': np.float32(32.5),
        'float64': np.float64(8.3),
        'int64': np.int64(3),
        'str': 'string',
    }

    encoded = yaml.dump(to_test, Dumper=NumpyToNativeDumper)
    output = yaml.safe_load(encoded)

    assert (
        isinstance(output['inside'], list) and
        output['inside'] == to_test['inside'].tolist()
    )
    assert (
        isinstance(output['nested']['float16'], list) and
        output['nested']['float16'] == to_test['nested']['float16'].tolist()
    )
    assert (
        isinstance(output['float32'], float) and
        output['float32'] == to_test['float32'].item()
    )
    assert (
        isinstance(output['float64'], float) and
        output['float64'] == to_test['float64'].item()
    )
    assert (
        isinstance(output['int64'], int) and
        output['int64'] == to_test['int64'].item()
    )
    assert (
        isinstance(output['str'], str) and
        output['str'] == to_test['str']
    )
