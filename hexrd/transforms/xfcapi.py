# We will replace these functions with the new versions as we
# add and test them.
# NOTE: we are only importing what is currently being used in hexrd
# and hexrdgui. This is so that we can see clearly what is in use.
from .old_xfcapi import (
    # Old transform functions still in use
    anglesToDVec,
    detectorXYToGvec,
    gvecToDetectorXY,
    gvecToDetectorXYArray,
    oscillAnglesOfHKLs,
    # Utility functions
    angularDifference,
    quat_distance,
    makeDetectorRotMat,
    makeEtaFrameRotMat,
    makeOscillRotMat,
    makeOscillRotMatArray,
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


from .new_capi.xf_new_capi import(
    # New transform functions
    angles_to_gvec,
)
