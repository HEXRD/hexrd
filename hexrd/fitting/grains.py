"""Grain fitting functions"""

import numpy as np

from scipy import optimize

from hexrd import matrixutil as mutil

from hexrd import transforms as xf
from hexrd.transforms import xfcapi
from hexrd import distortion as dFuncs

from hexrd.xrdutil import extract_detector_transformation

return_value_flag = None

epsf = np.finfo(float).eps  # ~2.2e-16
sqrt_epsf = np.sqrt(epsf)  # ~1.5e-8


# for grain parameters
gFlag_ref = np.ones(12, dtype=bool)
gScl_ref = np.ones(12, dtype=bool)


def fitGrain(gFull, instrument, reflections_dict,
             bMat, wavelength,
             gFlag=gFlag_ref, gScl=gScl_ref,
             omePeriod=None,
             factor=0.1, xtol=sqrt_epsf, ftol=sqrt_epsf):
    """
    """
    # FIXME: will currently fail if omePeriod is specifed
    if omePeriod is not None:
        # xyo_det[:, 2] = xf.mapAngle(xyo_det[:, 2], omePeriod)
        raise(RuntimeError, "ome period must be specified")

    gFit = gFull[gFlag]

    fitArgs = (gFull, gFlag, instrument, reflections_dict,
               bMat, wavelength, omePeriod)
    results = optimize.leastsq(objFuncFitGrain, gFit, args=fitArgs,
                               diag=1./gScl[gFlag].flatten(),
                               factor=0.1, xtol=xtol, ftol=ftol)

    gFit_opt = results[0]

    retval = gFull
    retval[gFlag] = gFit_opt
    return retval


def objFuncFitGrain(gFit, gFull, gFlag,
                    instrument,
                    reflections_dict,
                    bMat, wavelength,
                    omePeriod,
                    simOnly=False,
                    return_value_flag=return_value_flag):
    """
    gFull[0]  = expMap_c[0]
    gFull[1]  = expMap_c[1]
    gFull[2]  = expMap_c[2]
    gFull[3]  = tVec_c[0]
    gFull[4]  = tVec_c[1]
    gFull[5]  = tVec_c[2]
    gFull[6]  = vInv_MV[0]
    gFull[7]  = vInv_MV[1]
    gFull[8]  = vInv_MV[2]
    gFull[9]  = vInv_MV[3]
    gFull[10] = vInv_MV[4]
    gFull[11] = vInv_MV[5]

    OLD CALL
    objFuncFitGrain(gFit, gFull, gFlag,
                    detectorParams,
                    xyo_det, hkls_idx, bMat, wavelength,
                    bVec, eVec,
                    dFunc, dParams,
                    omePeriod,
                    simOnly=False, return_value_flag=return_value_flag)
    """

    bVec = instrument.beam_vector
    eVec = instrument.eta_vector

    # fill out parameters
    gFull[gFlag] = gFit

    # map parameters to functional arrays
    rMat_c = xfcapi.makeRotMatOfExpMap(gFull[:3])
    tVec_c = gFull[3:6].reshape(3, 1)
    vInv_s = gFull[6:]
    vMat_s = mutil.vecMVToSymm(vInv_s)  # NOTE: Inverse of V from F = V * R

    # loop over instrument panels
    # CAVEAT: keeping track of key ordering in the "detectors" attribute of
    # instrument here because I am not sure if instatiating them using
    # dict.fromkeys() preserves the same order if using iteration...
    # <JVB 2017-10-31>
    calc_omes_dict = dict.fromkeys(instrument.detectors, [])
    calc_xy_dict = dict.fromkeys(instrument.detectors)
    meas_xyo_all = []
    det_keys_ordered = []
    for det_key, panel in instrument.detectors.iteritems():
        det_keys_ordered.append(det_key)

        rMat_d, tVec_d, chi, tVec_s = extract_detector_transformation(
            instrument.detector_parameters[det_key])

        results = reflections_dict[det_key]
        if len(results) == 0:
            continue

        """
        extract data from results list fields:
          refl_id, gvec_id, hkl, sum_int, max_int, pred_ang, meas_ang, meas_xy

        or array from spots tables:
          0:5    ID    PID    H    K    L
          5:7    sum(int)    max(int)
          7:10   pred tth    pred eta    pred ome
          10:13  meas tth    meas eta    meas ome
          13:15  pred X    pred Y
          15:17  meas X    meas Y
        """
        if isinstance(results, list):
            # WARNING: hkls and derived vectors below must be columnwise;
            # strictly necessary??? change affected APIs instead?
            # <JVB 2017-03-26>
            hkls = np.atleast_2d(
                np.vstack([x[2] for x in results])
            ).T

            meas_xyo = np.atleast_2d(
                np.vstack([np.r_[x[7], x[6][-1]] for x in results])
            )
        elif isinstance(results, np.ndarray):
            hkls = np.atleast_2d(results[:, 2:5]).T
            meas_xyo = np.atleast_2d(results[:, [15, 16, 12]])

        # FIXME: distortion handling must change to class-based
        if panel.distortion is not None:
            meas_omes = meas_xyo[:, 2]
            xy_unwarped = panel.distortion[0](
                    meas_xyo[:, :2], panel.distortion[1])
            meas_xyo = np.vstack([xy_unwarped.T, meas_omes]).T
            pass

        # append to meas_omes
        meas_xyo_all.append(meas_xyo)

        # G-vectors:
        #   1. calculate full g-vector components in CRYSTAL frame from B
        #   2. rotate into SAMPLE frame and apply stretch
        #   3. rotate back into CRYSTAL frame and normalize to unit magnitude
        # IDEA: make a function for this sequence of operations with option for
        # choosing ouput frame (i.e. CRYSTAL vs SAMPLE vs LAB)
        gVec_c = np.dot(bMat, hkls)
        gVec_s = np.dot(vMat_s, np.dot(rMat_c, gVec_c))
        gHat_c = mutil.unitVector(np.dot(rMat_c.T, gVec_s))

        # !!!: check that this operates on UNWARPED xy
        match_omes, calc_omes = matchOmegas(
            meas_xyo, hkls, chi, rMat_c, bMat, wavelength,
            vInv=vInv_s, beamVec=bVec, etaVec=eVec,
            omePeriod=omePeriod)

        # append to omes dict
        calc_omes_dict[det_key] = calc_omes

        # TODO: try Numba implementations
        rMat_s = xfcapi.makeOscillRotMatArray(chi, calc_omes)
        calc_xy = xfcapi.gvecToDetectorXYArray(gHat_c.T,
                                               rMat_d, rMat_s, rMat_c,
                                               tVec_d, tVec_s, tVec_c,
                                               beamVec=bVec)

        # append to xy dict
        calc_xy_dict[det_key] = calc_xy
        pass

    # stack results to concatenated arrays
    calc_omes_all = np.hstack([calc_omes_dict[k] for k in det_keys_ordered])
    tmp = []
    for k in det_keys_ordered:
        if calc_xy_dict[k] is not None:
            tmp.append(calc_xy_dict[k])
    calc_xy_all = np.vstack(tmp)
    meas_xyo_all = np.vstack(meas_xyo_all)

    npts = len(meas_xyo_all)
    if np.any(np.isnan(calc_xy)):
        raise RuntimeError(
            "infeasible pFull: may want to scale" +
            "back finite difference step size")

    # return values
    if simOnly:
        # return simulated values
        if return_value_flag in [None, 1]:
            retval = np.hstack([calc_xy_all, calc_omes_all.reshape(npts, 1)])
        else:
            rd = dict.fromkeys(det_keys_ordered)
            for det_key in det_keys_ordered:
                rd[det_key] = {'calc_xy': calc_xy_dict[det_key],
                               'calc_omes': calc_omes_dict[det_key]}
            retval = rd
    else:
        # return residual vector
        # IDEA: try angles instead of xys?
        diff_vecs_xy = calc_xy_all - meas_xyo_all[:, :2]
        diff_ome = xf.angularDifference(calc_omes_all, meas_xyo_all[:, 2])
        retval = np.hstack([diff_vecs_xy,
                            diff_ome.reshape(npts, 1)
                            ]).flatten()
        if return_value_flag == 1:
            # return scalar sum of squared residuals
            retval = sum(abs(retval))
        elif return_value_flag == 2:
            # return DOF-normalized chisq
            # TODO: check this calculation
            denom = 3*npts - len(gFit) - 1.
            if denom != 0:
                nu_fac = 1. / denom
            else:
                nu_fac = 1.
            retval = nu_fac * sum(retval**2)
    return retval
