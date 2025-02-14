# tests for angles_to_gvec
# Written by Oscar Villellas

from __future__ import absolute_import

from collections import namedtuple

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import convert_axis_angle_to_rmat

from hexrd.core.transforms.new_capi.xf_new_capi import gvec_to_xy


# gvec_to_xy intersects vectors from crystal position with the detector plane.
#
# gvec_to_xy is always vectorized on gvecs. It can be used with either a single
# SAMPLE rotation matrix for all gvecs, or with a SAMPLE rotation matrix per
# gvec. Both cases should be exercised.

# LabSetup specifies the lab related transforms. That is, detector positions
# as well as beam orientation.
Experiment = namedtuple(
    'Experiment',
    [
        'rtol',
        'gvec_c',
        'rmat_d',
        'rmat_s',
        'rmat_c',
        'tvec_d',
        'tvec_s',
        'tvec_c',
        'result',
    ],
)


@pytest.fixture(scope='module')
def experiment():
    '''
    Note this fixture is data is actually extracted from some test runs. They
    are assumed to be correct, but no actual hard checking has been done.
    This is just a subset. It should be enough to exercise the gvec_to_xy
    function with different vector/scalar arrangements
    '''
    yield Experiment(
        rtol=1e-6,
        gvec_c=np.array(
            [
                [0.57735027, 0.57735028, 0.57735027],
                [0.57735027, -0.57735027, 0.57735028],
                [0.57735027, -0.57735028, -0.57735026],
                [0.57735028, 0.57735027, -0.57735027],
            ]
        ),
        rmat_d=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        rmat_s=np.array(
            [
                [
                    [0.77029942, 0.0, 0.63768237],
                    [0.0, 1.0, -0.0],
                    [-0.63768237, 0.0, 0.77029942],
                ],
                [
                    [0.97986016, 0.0, -0.19968493],
                    [-0.0, 1.0, -0.0],
                    [0.19968493, 0.0, 0.97986016],
                ],
                [
                    [0.30523954, 0.0, -0.9522756],
                    [-0.0, 1.0, -0.0],
                    [0.9522756, 0.0, 0.30523954],
                ],
                [
                    [0.73506994, 0.0, -0.67799129],
                    [-0.0, 1.0, -0.0],
                    [0.67799129, 0.0, 0.73506994],
                ],
            ]
        ),
        rmat_c=np.array(
            [
                [0.91734473, -0.08166131, 0.38962815],
                [0.31547749, 0.74606417, -0.58639766],
                [-0.24280159, 0.66084771, 0.71016033],
            ]
        ),
        tvec_d=np.array([0.0, 1.5, -5.0]),
        tvec_s=np.array([0.0, 0.0, 0.0]),
        tvec_c=np.array([-0.25, -0.25, -0.25]),
        result=np.array(
            [
                [0.13349048, -1.61131393],
                [0.19186549, -2.03119741],
                [0.63614123, -1.70709656],
                [0.12934705, -1.29999638],
            ]
        ),
    )


def test_scalar(experiment):
    '''A simple call using all scalar arguments'''
    result = gvec_to_xy(
        experiment.gvec_c[0],
        experiment.rmat_d,
        experiment.rmat_s[0],
        experiment.rmat_c,
        experiment.tvec_d,
        experiment.tvec_s,
        experiment.tvec_c,
    )
    assert_allclose(result, experiment.result[0], rtol=experiment.rtol)


def test_vector_c(experiment):
    '''a call using a vector of gvec_c and a single rmat_s. Result should be
    equivalent to multiple calls for each vector'''
    result = gvec_to_xy(
        experiment.gvec_c,
        experiment.rmat_d,
        experiment.rmat_s[0],
        experiment.rmat_c,
        experiment.tvec_d,
        experiment.tvec_s,
        experiment.tvec_c,
    )

    results = []
    for gvec_c in experiment.gvec_c:
        results.append(
            gvec_to_xy(
                gvec_c,
                experiment.rmat_d,
                experiment.rmat_s[0],
                experiment.rmat_c,
                experiment.tvec_d,
                experiment.tvec_s,
                experiment.tvec_c,
            )
        )

    assert_allclose(result, results, rtol=experiment.rtol)


def test_vector_cs(experiment):
    '''a call using a vector of gvec_c and a vector of rmat_s. Result should be
    equivalent to multiple calls for each vector and matrix pair'''
    result = gvec_to_xy(
        experiment.gvec_c,
        experiment.rmat_d,
        experiment.rmat_s,
        experiment.rmat_c,
        experiment.tvec_d,
        experiment.tvec_s,
        experiment.tvec_c,
    )

    results = []
    for gvec_c, rmat_s in zip(experiment.gvec_c, experiment.rmat_s):
        results.append(
            gvec_to_xy(
                gvec_c,
                experiment.rmat_d,
                rmat_s,
                experiment.rmat_c,
                experiment.tvec_d,
                experiment.tvec_s,
                experiment.tvec_c,
            )
        )

    assert_allclose(result, results, rtol=experiment.rtol)


def test_vector(experiment):
    '''A simple call using vector arguments for gvec_c and rmat_s, based on the
    results of the experiment'''
    result = gvec_to_xy(
        experiment.gvec_c,
        experiment.rmat_d,
        experiment.rmat_s,
        experiment.rmat_c,
        experiment.tvec_d,
        experiment.tvec_s,
        experiment.tvec_c,
    )
    assert_allclose(result, experiment.result, rtol=experiment.rtol)


def test_rotated_scalar(experiment):
    '''A simple call using all scalar arguments'''
    experiment_rot = convert_axis_angle_to_rmat(np.r_[0.5, 0.2, 0.6], 1.0)

    gvec_c = experiment.gvec_c  # gvec_c are relative to the crystal frame
    rmat_d = experiment_rot @ experiment.rmat_d
    rmat_s = experiment_rot @ experiment.rmat_s
    rmat_c = experiment.rmat_c  # rMat_c is in sample frame
    tvec_d = experiment_rot @ experiment.tvec_d
    tvec_s = experiment_rot @ experiment.tvec_s
    tvec_c = experiment.tvec_c  # tvec_c is relative to sample frame
    beam = experiment_rot @ np.r_[0.0, 0.0, -1.0]

    result = gvec_to_xy(
        gvec_c[0],
        rmat_d,
        rmat_s[0],
        rmat_c,
        tvec_d,
        tvec_s,
        tvec_c,
        beam_vec=beam,
    )

    assert_allclose(result, experiment.result[0], rtol=experiment.rtol)


def test_rotated_vector(experiment):
    '''Rotating the LAB frame and all associated experiment values should not
    affect the results. As some of the rotations are relative to other
    rotations, not everything should need to be rotated.

    This serves as a test for non-standard beam vector as well'''
    # note: rotation is  arbitrary...
    experiment_rot = convert_axis_angle_to_rmat(np.r_[0.5, 0.2, 0.6], 1.0)

    gvec_c = experiment.gvec_c  # gvec_c are relative to the crystal frame
    rmat_d = experiment_rot @ experiment.rmat_d
    rmat_s = experiment_rot @ experiment.rmat_s
    rmat_c = experiment.rmat_c  # rMat_c is in sample frame
    tvec_d = experiment_rot @ experiment.tvec_d
    tvec_s = experiment_rot @ experiment.tvec_s
    tvec_c = experiment.tvec_c  # tvec_c is relative to sample frame
    beam = experiment_rot @ np.r_[0.0, 0.0, -1.0]

    result = gvec_to_xy(
        gvec_c, rmat_d, rmat_s, rmat_c, tvec_d, tvec_s, tvec_c, beam_vec=beam
    )

    assert_allclose(result, experiment.result, rtol=experiment.rtol)
