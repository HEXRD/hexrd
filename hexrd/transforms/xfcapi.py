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
    makeDetectorRotMat,  # New version provided below
    makeEtaFrameRotMat,  # new version provided below
    makeOscillRotMat,  # new version provided below
    makeOscillRotMatArray,  # new version provided below
    makeRotMatOfExpMap,
    makeRotMatOfQuat,  # Use rotations.rotMatOfQuat instead
    mapAngle,  # Use rotations.mapAngle instead
    rowNorm,  # use numpy.linalg.norm(..., axis=1) instead
    unitRowVector,  # new version below
    # Constants,
    bVec_ref,
    eta_ref,
    Xl,
    Yl,
)


from .new_capi.xf_new_capi import (
    # New transform functions
    angles_to_dvec,
    angles_to_gvec,
    gvec_to_xy,  # this is gvecToDetectorXY and gvecToDetectorXYArray
    make_beam_rmat,  # this is makeEtaFrameRotMat
    make_detector_rmat,
    make_rmat_of_expmap,
    make_sample_rmat,  # this is makeOscillRotMat and makeOscillRotMatArray
    oscill_angles_of_hkls,
    quat_distance,
    rotate_vecs_about_axis,
    unit_vector,  # this is unitRowVector
    validate_angle_ranges,
    xy_to_gvec,
)
