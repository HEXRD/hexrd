# -*- coding: utf-8 -*-
"""
Transforms module implementation using a support C extension module.

Currently, this implementation contains code for the following functions:

 - angles_to_gvec
 - angles_to_dvec
 - gvec_to_xy
 - xy_to_gvec (partial)

 - unit_vector
 - make_rmat_of_exp_map
 - make_binary_rmat
 - make_beam_rmat
 - validate_angle_ranges
 - rotate_vecs_about_axis
 - quat_distance

There are also some functions that maybe would be needed in the transforms module:
 - makeGVector
 - makeRotMatOfQuat
 - homochoricOfQuat
"""
from . import constants as cnst
from .transforms_definitions import xf_api
from hexrd.transforms import _new_transforms_capi as _impl

import numpy as np

@xf_api
def angles_to_gvec(
        angs,
        beam_vec=None, eta_vec=None,
        chi=None, rmat_c=None):
    orig_ndim = angs.ndim

    # if only a pair is provided... converto to a triplet with omegas == 0
    # so that behavior is preserved.
    if angs.shape[-1] == 2:
        angs = np.hstack((angs, np.zeros(angs.shape[:-1]+(1,))))

    angs = np.ascontiguousarray( np.atleast_2d(angs) )
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    beam_vec = np.ascontiguousarray( beam_vec.flatten() )
    eta_vec = eta_vec if eta_vec is not None else cnst.eta_vec
    eta_vec = np.ascontiguousarray( eta_vec.flatten() )
    chi = 0.0 if chi is None else float(chi)
    rmat_c = cnst.identity_3x3 if rmat_c is None else np.ascontiguousarray( rmat_c )

    result = _impl.anglesToGVec(angs, beam_vec, eta_vec, chi, rmat_c)

    return result[0] if orig_ndim == 1 else result


@xf_api
def angles_to_dvec(
        angs,
        beam_vec=None, eta_vec=None,
        chi=None, rmat_c=None):
    # TODO: Improve capi to avoid multiplications when rmat_c is None
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    eta_vec = eta_vec if eta_vec is not None else cnst.eta_vec

    angs = np.ascontiguousarray( np.atleast_2d(angs) )
    beam_vec = np.ascontiguousarray( beam_vec.flatten() )
    eta_vec = np.ascontiguousarray( eta_vec.flatten() )
    rmat_c = np.ascontiguousarray(rmat_c) if rmat_c is not None else cnst.identity_3x3
    chi = 0.0 if chi is None else float(chi)

    return _impl.anglesToDVec(angs,
                                         beam_vec, eta_vec,
                                         chi, rmat_c)

def makeGVector(hkl, bMat):
    assert hkl.shape[0] == 3, 'hkl input must be (3, n)'
    return unitVector(np.dot(bMat, hkl))


@xf_api
def gvec_to_xy(gvec_c,
               rmat_d, rmat_s, rmat_c,
               tvec_d, tvec_s, tvec_c,
               beam_vec=None,
               vmat_inv=None,
               bmat=None):
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec

    orig_ndim = gvec_c.ndim
    gvec_c  = np.ascontiguousarray( np.atleast_2d(gvec_c) )
    rmat_s  = np.ascontiguousarray( rmat_s )
    tvec_d  = np.ascontiguousarray( tvec_d.flatten()  )
    tvec_s  = np.ascontiguousarray( tvec_s.flatten()  )
    tvec_c  = np.ascontiguousarray( tvec_c.flatten()  )
    beam_vec = np.ascontiguousarray( beam_vec.flatten() )

    # depending on the number of dimensions of rmat_s use either the array
    # version or the "scalar" (over rmat_s) version. Note that rmat_s is either
    # a 3x3 matrix (ndim 2) or an nx3x4 array of matrices (ndim 3)
    if rmat_s.ndim > 2:
        result =  _impl.gvecToDetectorXYArray(gvec_c,
                                              rmat_d, rmat_s, rmat_c,
                                              tvec_d, tvec_s, tvec_c,
                                              beam_vec)
    else:
        result =  _impl.gvecToDetectorXY(gvec_c,
                                         rmat_d, rmat_s, rmat_c,
                                         tvec_d, tvec_s, tvec_c,
                                         beam_vec)
    return result[0] if orig_ndim == 1 else result


@xf_api
def xy_to_gvec(xy_d,
               rmat_d, rmat_s,
               tvec_d, tvec_s, tvec_c,
               rmat_b=None,
               distortion=None,
               output_ref=False):
    # in the C library beam vector and eta vector are expected. However we receive
    # rmat_b. Please check this!
    #
    # It also seems that the output_ref version is not present as the argument gets
    # ignored

    rmat_b = rmat_b if rmat_b is not None else cnst.identity_3x3

    # the code seems to ignore this argument, assume output_ref == True not implemented
    assert not output_ref

    if distortion is not None:
        xy_d = distortion.unwarp(xy_d)

    xy_d  = np.ascontiguousarray( np.atleast_2d(xy_d) )
    rmat_d = np.ascontiguousarray( rmat_d )
    rmat_s = np.ascontiguousarray( rmat_s )
    tvec_d  = np.ascontiguousarray( tvec_d.flatten() )
    tvec_s  = np.ascontiguousarray( tvec_s.flatten() )
    tvec_c  = np.ascontiguousarray( tvec_c.flatten() )
    beam_vec = np.ascontiguousarray( (-rmat_b[:,2]).flatten() )
    eta_vec  = np.ascontiguousarray( rmat_b[:,0].flatten() ) #check this!
    return _impl.detectorXYToGvec(xy_det,
                                  rmat_d, rmat_s,
                                  tvec_d, tvec_s, tvec_c,
                                  beam_vec, eta_vec)


#@xf_api
def oscillAnglesOfHKLs(hkls, chi, rMat_c, bMat, wavelength,
                       vInv=None, beamVec=cnst.beam_vec, etaVec=cnst.eta_vec):
    # this was oscillAnglesOfHKLs
    hkls = np.array(hkls, dtype=float, order='C')
    if vInv is None:
        vInv = np.ascontiguousarray(vInv_ref.flatten())
    else:
        vInv = np.ascontiguousarray(vInv.flatten())
    beamVec = np.ascontiguousarray(beamVec.flatten())
    etaVec  = np.ascontiguousarray(etaVec.flatten())
    bMat = np.ascontiguousarray(bMat)
    return _impl.oscillAnglesOfHKLs(
        hkls, chi, rMat_c, bMat, wavelength, vInv, beamVec, etaVec
        )


@xf_api
def unit_vector(vec_in):
    vec_in = np.ascontiguousarray(vec_in)
    if vec_in.ndim == 1:
        return _impl.unitRowVector(vec_in)
    elif vec_in.ndim == 2:
        return _impl.unitRowVectors(vec_in)
    else:
        raise ValueError(
            "incorrect arg shape; must be 1-d or 2-d, yours is %d-d"
            % (vec_in.ndim)
        )


#@xf_api
def makeDetectorRotMat(tiltAngles):
    arg = np.ascontiguousarray(np.r_[tiltAngles].flatten())
    return _impl.makeDetectorRotMat(arg)


# make_sample_rmat in CAPI is split between makeOscillRotMat
# and makeOscillRotMatArray...

#@xf_api
def makeOscillRotMat(oscillAngles):
    chi, ome = oscillAngles
    ome = np.atleast1d(ome)
    result = _impl.makeOscillRotMat(chi, ome)
    return result.reshape((3, 3))


#@xf_api
def makeOscillRotMatArray(chi, omeArray):
    arg = np.ascontiguousarray(omeArray)
    return _impl.makeOscillRotMat(chi, arg)


@xf_api
def make_sample_rmat(chi, ome):
    ome_array = np.atleast_1d(ome)
    if ome is ome_array:
        ome_array = np.ascontiguousarray(ome_array)
        result = _impl.makeOscillRotMat(chi, ome_array)
    else:
        # converted to 1d array of 1 element, no need
        # to call ascontiguousarray, but need to remove
        # the outer dimension from the result
        result = _impl.makeOscillRotMat(chi, ome_array)
        result = result.reshape(3,3)

    return result

@xf_api
def make_rmat_of_expmap(exp_map):
    arg = np.ascontiguousarray(exp_map.flatten())
    return _impl.makeRotMatOfExpMap(arg)


@xf_api
def make_binary_rmat(axis):
    arg = np.ascontiguousarray(axis.flatten())
    return _impl.makeBinaryRotMat(arg)


@xf_api
def make_beam_rmat(bvec_l, evec_l):
    arg1 = np.ascontiguousarray(bvec_l.flatten())
    arg2 = np.ascontiguousarray(evec_l.flatten())
    return _impl.makeEtaFrameRotMat(arg1, arg2)


@xf_api
def validate_angle_ranges(ang_list, start_angs, stop_angs, ccw=True):
    ang_list = ang_list.astype(np.double, order="C")
    start_angs = start_angs.astype(np.double, order="C")
    stop_angs = stop_angs.astype(np.double, order="C")

    return _impl.validateAngleRanges(ang_list, start_angs, stop_angs, ccw)


@xf_api
def rotate_vecs_about_axis(angle, axis, vecs):
    angle = np.asarray(angle)
    axis = np.ascontiguousarray(axis.T)
    vecs = np.ascontiguousarray(vecs.T)
    result = _impl.rotate_vecs_about_axis(angle, axis, vecs)
    return result.T

@xf_api
def quat_distance(q1, q2, qsym):
    q1 = np.ascontiguousarray(q1.flatten())
    q2 = np.ascontiguousarray(q2.flatten())
    # C module expects quaternions in row major, numpy code in column major.
    qsym = np.ascontiguousarray(qsym.T)
    return _impl.quat_distance(q1, q2, qsym)
