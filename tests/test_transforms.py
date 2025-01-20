import copy
import json

import numpy as np

from hexrd.core.transforms.xfcapi import gvec_to_xy

from common import convert_axis_angle_to_rmat


def test_gvec_to_xy(test_data_dir):
    with open(test_data_dir / 'gvec_to_xy.json') as rf:
        test_data = json.load(rf)

    for entry in test_data:
        kwargs = entry['input']
        expected = entry['output']

        kwargs = {k: np.asarray(v) for k, v in kwargs.items()}
        result = gvec_to_xy(**kwargs)
        assert np.allclose(result, expected)

        # Verify that we get the correct answer with a rotation.
        rot = convert_axis_angle_to_rmat(np.r_[0.5, 0.2, 0.6], 1.0)

        rotated_kwargs = copy.deepcopy(kwargs)
        rotated_kwargs['beam_vec'] = np.r_[0.0, 0.0, -1.0]

        # The following are not rotated:
        # gvec_c are relative to the crystal frame
        # rMat_c is in sample frame
        # tvec_c is relative to sample frame
        to_rotate = ['rmat_d', 'rmat_s', 'tvec_d', 'tvec_s', 'beam_vec']
        for k in to_rotate:
            rotated_kwargs[k] = rot @ rotated_kwargs[k]

        result = gvec_to_xy(**rotated_kwargs)
        assert np.allclose(result, expected)
