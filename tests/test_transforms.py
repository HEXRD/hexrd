import json

import numpy as np

from hexrd.transforms.xfcapi import gvec_to_xy


def test_gvec_to_xy(test_data_dir):
    with open(test_data_dir / 'gvec_to_xy.json') as rf:
        test_data = json.load(rf)

    for entry in test_data:
        kwargs = entry['input']
        output = entry['output']

        kwargs = {k: np.asarray(v) for k, v in kwargs.items()}
        result = gvec_to_xy(**kwargs)
        assert np.allclose(result, output)
