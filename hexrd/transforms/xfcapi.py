# We will replace these functions with the new versions as we
# add and test them.
# NOTE: we are only importing what is currently being used in hexrd
# and hexrdgui. This is so that we can see clearly what is in use.
from .old_xfcapi import (
    # Old transform functions still in use
    anglesToDVec,  # new version provided below
    anglesToGVec,  # new version provided below
    detectorXYToGvec,  # new version provided below
    gvecToDetectorXY,  # new version provided below
    gvecToDetectorXYArray,  # new version provided below
    oscillAnglesOfHKLs,
    # Utility functions
    angularDifference,
    # quat_distance, # new version provided below
    makeDetectorRotMat,
    makeEtaFrameRotMat,  # new version provided below (make_beam_rmat)
    makeOscillRotMat,  # new version provided below (make_sample_rmat)
    makeOscillRotMatArray,  # new version provided below (make_sample_rmat)
    makeRotMatOfExpMap,  # new version provided below
    makeRotMatOfQuat,
    mapAngle,
    rowNorm,
    unitRowVector,  # new version provided below (unit_vector)
    # Constants,
    bVec_ref,
    eta_ref,
    Xl,
    Yl,
)


from .new_capi.xf_new_capi import(
    # New transform functions
    angles_to_gvec,
    angles_to_dvec,
    gvec_to_xy,  # this is gvecToDetectorXY and gvecToDetectorXYArray
    xy_to_gvec,
    make_sample_rmat,  # this is makeOscillRotMat and makeOscillRotMatArray
    make_beam_rmat,
    unit_vector,
    quat_distance
)
