# tests for capi implementation of xy_to_gvec.
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used
# From Oscar Villellas

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from hexrd.transforms.new_capi.xf_new_capi import xy_to_gvec

Experiment = namedtuple(
    'Experiment', ['xy_det', 'rmat_d', 'rmat_s', 'tvec_d', 'tvec_s', 'tvec_c']
)

@pytest.fixture(scope='module')
def experiment():
    '''This fixture only is about argument types and dimensions.
    There is no need for it to reflect a real problem
    '''
    yield Experiment(
        xy_det=np.array(
            [
                [0.57735027, 0.57735028],
                [0.57735027, -0.57735027],
                [0.57735027, -0.57735028],
                [0.57735028, 0.57735027],
            ]
        ),
        rmat_d=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        rmat_s=np.array(
            [
                [0.73506994, 0.0, -0.67799129],
                [-0.0, 1.0, -0.0],
                [0.67799129, 0.0, 0.73506994],
            ]
        ),
        tvec_d=np.array([0.0, 1.5, -5.0]),
        tvec_s=np.array([0.0, 0.0, 0.0]),
        tvec_c=np.array([-0.25, -0.25, -0.25]),
    )

def test_correct_xy_to_gvec(experiment):
    result = xy_to_gvec(
        experiment.xy_det,
        experiment.rmat_d,
        experiment.rmat_s,
        experiment.tvec_d,
        experiment.tvec_s,
        experiment.tvec_c,
    )

def test_incorrect_rmat_d(experiment):
    # bad number of dimensions (less than 2)
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            np.r_[0.0, 0.0, 1.0],
            experiment.rmat_s,
            experiment.tvec_d,
            experiment.tvec_s,
            experiment.tvec_c,
        )

    # bad number of dimensions (more than 2)
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            np.zeros((3, 3, 3), dtype=np.double),
            experiment.rmat_s,
            experiment.tvec_d,
            experiment.tvec_s,
            experiment.tvec_c,
        )

    # bad dimensions
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            np.eye(4),
            experiment.rmat_s,
            experiment.tvec_d,
            experiment.tvec_s,
            experiment.tvec_c,
        )

def test_incorrect_rmat_s(experiment):
    # bad number of dimensions (less than 2)
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            np.r_[0.0, 0.0, 1.0],
            experiment.tvec_d,
            experiment.tvec_s,
            experiment.tvec_c,
        )

    # bad number of dimensions (more than 2)
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            np.zeros((3, 3, 3), dtype=np.double),
            experiment.tvec_d,
            experiment.tvec_s,
            experiment.tvec_c,
        )

    # bad dimensions
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            np.eye(4),
            experiment.tvec_d,
            experiment.tvec_s,
            experiment.tvec_c,
        )

def test_incorrect_tvec_d(experiment):
    # bad number of dimensions (more than 1)
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            experiment.rmat_s,
            np.eye(3),
            experiment.tvec_s,
            experiment.tvec_c,
        )

    # bad dimensions
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            experiment.rmat_s,
            np.r_[0.0, 1.0],
            experiment.tvec_s,
            experiment.tvec_c,
        )

def test_incorrect_tvec_s(experiment):
    # bad number of dimensions (more than 1)
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            experiment.rmat_s,
            experiment.tvec_d,
            np.eye(3),
            experiment.tvec_c,
        )

    # bad dimensions
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            experiment.rmat_s,
            experiment.tvec_d,
            np.r_[0.0, 1.0],
            experiment.tvec_c,
        )

def test_incorrect_tvec_c(experiment):
    # bad number of dimensions (more than 1)
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            experiment.rmat_s,
            experiment.tvec_d,
            experiment.tvec_s,
            np.eye(3),
        )
    # bad dimensions
    with pytest.raises(ValueError):
        result = xy_to_gvec(
            experiment.xy_det,
            experiment.rmat_d,
            experiment.rmat_s,
            experiment.tvec_d,
            experiment.tvec_s,
            np.r_[0.0, 1.0],
        )