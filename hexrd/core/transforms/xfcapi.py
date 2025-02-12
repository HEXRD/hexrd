# We will replace these functions with the new versions as we
# add and test them.
# NOTE: we are only importing what is currently being used in hexrd
# and hexrdgui. This is so that we can see clearly what is in use.
from .old_xfcapi import anglesToDVec, anglesToGVec, detectorXYToGvec, gvecToDetectorXY, gvecToDetectorXYArray, oscillAnglesOfHKLs, angularDifference, makeDetectorRotMat, makeEtaFrameRotMat, makeOscillRotMat, makeOscillRotMatArray, makeRotMatOfExpMap, makeRotMatOfQuat, mapAngle, rowNorm, unitRowVector, bVec_ref, eta_ref, Xl, Yl


from .new_capi.xf_new_capi import angles_to_dvec, angles_to_gvec, gvec_to_xy, make_beam_rmat, make_detector_rmat, make_rmat_of_expmap, make_sample_rmat, oscill_angles_of_hkls, quat_distance, rotate_vecs_about_axis, unit_vector, validate_angle_ranges, xy_to_gvec
