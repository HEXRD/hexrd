# We will replace these functions with the new versions as we
# add and test them.
# NOTE: we are only importing what is currently being used in hexrd
# and hexrdgui. This is so that we can see clearly what is in use.
from .old_xfcapi import (
    # Old transform functions still in use
    anglesToDVec,
    anglesToGVec,  # new version provided below
    detectorXYToGvec,
    gvecToDetectorXY,  # new version provided below
    gvecToDetectorXYArray,  # new version provided below
    oscillAnglesOfHKLs,
    # Utility functions
    angularDifference,
    quat_distance,
    makeDetectorRotMat,
    makeEtaFrameRotMat,
    makeOscillRotMat,  # new version provided below
    makeOscillRotMatArray,  # new version provided below
    makeRotMatOfExpMap,
    makeRotMatOfQuat,
    mapAngle,
    rowNorm,
    unitRowVector,
    # Constants,
    bVec_ref,
    eta_ref,
    Xl,
    Yl,
)


import numpy as np

max_diff = 0
max_array_diff = 0


def gvec_to_xy(*args, **kwargs):
    from .new_capi import xf_new_capi

    new_result = xf_new_capi.gvec_to_xy(*args, **kwargs)

    if 'beam_vec' in kwargs:
        # Convert to older kwarg name
        kwargs['beamVec'] = kwargs.pop('beam_vec')

    old_result = gvecToDetectorXY(*args, **kwargs)

    global max_diff
    diff = np.nanmax(np.abs(new_result - old_result))
    if diff > max_diff:
        max_diff = diff
        print('New max diff for gvec_to_xy:', max_diff)

    return old_result


def gvec_to_xy_array(*args, **kwargs):
    from .new_capi import xf_new_capi

    new_result = xf_new_capi.gvec_to_xy(*args, **kwargs)

    if 'beam_vec' in kwargs:
        # Convert to older kwarg name
        kwargs['beamVec'] = kwargs.pop('beam_vec')

    old_result = gvecToDetectorXYArray(*args, **kwargs)

    global max_array_diff
    diff = np.nanmax(np.abs(new_result - old_result))
    if diff > max_array_diff:
        max_array_diff = diff
        print('New max diff for gvec_to_xy_array:', max_array_diff)

    return old_result


from .new_capi.xf_new_capi import(
    # New transform functions
    angles_to_gvec,
    # gvec_to_xy,  # this is gvecToDetectorXY and gvecToDetectorXYArray
    make_sample_rmat,  # this is makeOscillRotMat and makeOscillRotMatArray
)
