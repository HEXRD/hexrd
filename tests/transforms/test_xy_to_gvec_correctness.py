# tests for capi implementation of xy_to_gvec.
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used
# From Oscar Villellas

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import xy_to_gvec, gvec_to_xy

Experiment = namedtuple('Experiment', ['xy_det', 'rmat_d', 'rmat_s',
                                       'tvec_d', 'tvec_s', 'tvec_c'])


@pytest.fixture(scope='module')
def experiment():
    '''This fixture only is about argument types and dimensions.
    There is no need for it to reflect a real problem
    '''
    yield Experiment(
        xy_det=np.array([[ 0.57735027,  0.57735028],
                         [ 0.57735027, -0.57735027],
                         [ 0.57735027, -0.57735028],
                         [ 0.57735028,  0.57735027]]),
        rmat_d=np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]]),
        rmat_s=np.array([[ 0.73506994,  0.        , -0.67799129],
                         [-0.        ,  1.        , -0.        ],
                         [ 0.67799129,  0.        ,  0.73506994]]),
        tvec_d=np.array([ 0. ,  1.5, -5. ]),
        tvec_s=np.array([0., 0., 0.]),
        tvec_c=np.array([-0.25, -0.25, -0.25])
    )


def test_correct_xy_to_gvec(experiment):
    result = xy_to_gvec(experiment.xy_det,
                        experiment.rmat_d,
                        experiment.rmat_s,
                        experiment.tvec_d,
                        experiment.tvec_s,
                        experiment.tvec_c)
    print(result[1][0])
    # Run inverse
    xy = gvec_to_xy(result[1],
                   experiment.rmat_d,
                   experiment.rmat_s,
                   np.eye(3),
                   experiment.tvec_d,
                   experiment.tvec_s,
                   experiment.tvec_c)
    
    print(xy)
    
    assert np.allclose(experiment.xy_det, xy)