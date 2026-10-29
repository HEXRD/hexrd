"""Grain fitting functions"""

import numpy as np

from scipy import optimize

from hexrd.core import matrixutil as mutil

from hexrd.core.transforms import xfcapi
from hexrd.core import constants
from hexrd.core import rotations
from hexrd.core.instrument._spotutils import _SpotData

from hexrd.hedm.xrdutil import (
    apply_correction_to_wavelength,
    extract_detector_transformation,
)

return_value_flag = None

epsf = np.finfo(float).eps  # ~2.2e-16
sqrt_epsf = np.sqrt(epsf)  # ~1.5e-8

bVec_ref = constants.beam_vec
eta_ref = constants.eta_vec
vInv_ref = np.r_[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


# for grain parameters
gFlag_ref = np.ones(12, dtype=bool)
gScl_ref = np.ones(12, dtype=bool)


def _fit_data(results):
    """Normalize the supported spot representations at the fitting boundary."""
    if isinstance(results, _SpotData):
        return results.as_fit_data() if len(results) else []
    if isinstance(results, list) and len(results):
        return {
            'hkls': np.atleast_2d(np.vstack([x[2] for x in results])).T,
            'meas_xyo': np.atleast_2d(
                np.vstack([np.r_[x[7], x[6][-1]] for x in results])
            ),
        }
    if isinstance(results, np.ndarray) and len(results):
        return {
            'hkls': np.atleast_2d(results[:, 2:5]).T,
            'meas_xyo': np.atleast_2d(results[:, [15, 16, 12]]),
        }
    return results


class _FitGrainData(dict):
    """A fit-local snapshot of reflection and detector quantities."""

    def __init__(self, reflections, instrument, bmat):
        super().__init__()
        self.transforms = {}
        self.reciprocal = {}
        panels = {}
        parameters = None
        for key, results in reflections.items():
            if not isinstance(results, dict):
                self[key] = results
                continue
            if parameters is None:
                parameters = instrument.detector_parameters
            self.transforms[key] = extract_detector_transformation(parameters[key])
            measured = results['meas_xyo']
            distortion = instrument.detectors[key].distortion
            if distortion is not None:
                measured = np.vstack(
                    [distortion.apply(measured[:, :2]).T, measured[:, 2]]
                ).T
            self[key] = {'hkls': results['hkls'], 'meas_xyo': measured}
            self.reciprocal[key] = np.dot(bmat, results['hkls'])
            rd, td, chi, ts = self.transforms[key]
            panels[key] = (
                results['hkls'].T,
                self.reciprocal[key],
                measured,
                rd,
                td,
                chi,
                ts,
            )
        from hexrd.core.extensions import transforms as native

        ordered = [panels[key] for key in instrument.detectors if key in panels]
        compatible = all(
            np.asarray(array).dtype.kind in 'biuf'
            and np.asarray(array).dtype.itemsize <= 8
            for array in [bmat] + [item[i] for item in ordered for i in (0, 1, 2)]
        )
        self.native = (
            native._GrainObjective(
                ordered, bmat, instrument.beam_vector, instrument.eta_vector
            )
            if ordered and compatible
            else None
        )


def fitGrain(
    gFull,
    instrument,
    reflections_dict,
    bMat,
    wavelength,
    gFlag=gFlag_ref,
    gScl=gScl_ref,
    omePeriod=None,
    factor=0.1,
    xtol=sqrt_epsf,
    ftol=sqrt_epsf,
):
    """Fit grain parameters to reflection positions with MINPACK least squares."""
    # FIXME: will currently fail if omePeriod is specifed
    if omePeriod is not None:
        # xyo_det[:, 2] = rotations.mapAngle(xyo_det[:, 2], omePeriod)
        raise RuntimeError

    gFit = gFull[gFlag]

    prepared = _FitGrainData(
        {key: _fit_data(value) for key, value in reflections_dict.items()},
        instrument,
        bMat,
    )
    fitArgs = (
        gFull,
        gFlag,
        instrument,
        prepared,
        bMat,
        wavelength,
        omePeriod,
    )
    from hexrd.hedm._progress import check_cancelled
    from hexrd.hedm._fit_progress import current_reporter

    reporter = current_reporter()

    def objective(*args):
        check_cancelled()
        if reporter is not None:
            reporter.publish()
        return objFuncFitGrain(*args)

    results = optimize.leastsq(
        objective,
        gFit,
        args=fitArgs,
        diag=1.0 / gScl[gFlag].flatten(),
        factor=0.1,
        xtol=xtol,
        ftol=ftol,
    )

    gFit_opt = results[0]

    retval = gFull
    retval[gFlag] = gFit_opt
    return retval


def objFuncFitGrain(
    gFit,
    gFull,
    gFlag,
    instrument,
    reflections_dict,
    bMat,
    wavelength,
    omePeriod,
    simOnly=False,
    return_value_flag=return_value_flag,
):
    """
    Calculate residual between measured and simulated ff-HEDM G-vectors.

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

    """
    bVec = instrument.beam_vector
    eVec = instrument.eta_vector
    tVec_s = instrument.tvec
    energy_correction = instrument.energy_correction

    # fill out parameters
    gFull[gFlag] = gFit

    # map parameters to functional arrays
    tVec_c = gFull[3:6].reshape(3, 1)

    # Apply an energy correction according to grain position
    corrected_wavelength = apply_correction_to_wavelength(
        wavelength,
        energy_correction,
        tVec_s,
        tVec_c,
    )

    if (
        isinstance(reflections_dict, _FitGrainData)
        and reflections_dict.native is not None
        and omePeriod is None
        and not simOnly
        and return_value_flag is None
    ):
        result = reflections_dict.native.residual(gFull, corrected_wavelength)
        if result is not None:
            return result

    rMat_c = xfcapi.make_rmat_of_expmap(gFull[:3])
    vInv_s = gFull[6:]
    vMat_s = mutil.vecMVToSymm(vInv_s)

    calc_omes_dict = dict.fromkeys(instrument.detectors, [])
    calc_xy_dict = dict.fromkeys(instrument.detectors)
    meas_xyo_all = []
    det_keys_ordered = list(instrument.detectors)
    for det_key, panel in instrument.detectors.items():
        results = reflections_dict.get(det_key, [])
        if not isinstance(results, dict) and len(results) == 0:
            continue
        prepared = isinstance(reflections_dict, _FitGrainData)
        rMat_d, tVec_d, chi, tVec_s = (
            reflections_dict.transforms[det_key]
            if prepared
            else extract_detector_transformation(
                instrument.detector_parameters[det_key]
            )
        )
        data = _fit_data(results)
        hkls, meas_xyo = data['hkls'], data['meas_xyo']

        # distortion handling
        if panel.distortion is not None and not prepared:
            meas_omes = meas_xyo[:, 2]
            xy_unwarped = panel.distortion.apply(meas_xyo[:, :2])
            meas_xyo = np.vstack([xy_unwarped.T, meas_omes]).T

        # append to meas_omes
        meas_xyo_all.append(meas_xyo)

        # G-vectors:
        #   1. calculate full g-vector components in CRYSTAL frame from B
        #   2. rotate into SAMPLE frame and apply stretch
        #   3. rotate back into CRYSTAL frame and normalize to unit magnitude
        # IDEA: make a function for this sequence of operations with option for
        # choosing ouput frame (i.e. CRYSTAL vs SAMPLE vs LAB)
        gVec_c = (
            reflections_dict.reciprocal[det_key] if prepared else np.dot(bMat, hkls)
        )
        gVec_s = np.dot(vMat_s, np.dot(rMat_c, gVec_c))
        gHat_c = mutil.unitVector(np.dot(rMat_c.T, gVec_s))

        # !!!: check that this operates on UNWARPED xy
        match_omes, calc_omes = matchOmegas(
            meas_xyo,
            hkls,
            chi,
            rMat_c,
            bMat,
            corrected_wavelength,
            vInv=vInv_s,
            beamVec=bVec,
            etaVec=eVec,
            omePeriod=omePeriod,
        )

        # append to omes dict
        calc_omes_dict[det_key] = calc_omes

        # TODO: try Numba implementations
        rMat_s = xfcapi.make_sample_rmat(chi, calc_omes)
        calc_xy = xfcapi.gvec_to_xy(
            gHat_c.T,
            rMat_d,
            rMat_s,
            rMat_c,
            tVec_d,
            tVec_s,
            tVec_c,
            beam_vec=bVec,
        )

        # append to xy dict
        calc_xy_dict[det_key] = calc_xy

    # stack results to concatenated arrays
    calc_omes_all = np.hstack([calc_omes_dict[k] for k in det_keys_ordered])
    calc_xy_all = np.vstack([xy for xy in calc_xy_dict.values() if xy is not None])
    meas_xyo_all = np.vstack(meas_xyo_all)

    npts = len(meas_xyo_all)
    if np.any(np.isnan(calc_xy)):
        raise RuntimeError(
            "infeasible pFull: may want to scale" + "back finite difference step size"
        )

    # return values
    if simOnly:
        # return simulated values
        if return_value_flag in [None, 1]:
            retval = np.hstack([calc_xy_all, calc_omes_all.reshape(npts, 1)])
        else:
            rd = dict.fromkeys(det_keys_ordered)
            for det_key in det_keys_ordered:
                rd[det_key] = {
                    'calc_xy': calc_xy_dict[det_key],
                    'calc_omes': calc_omes_dict[det_key],
                }
            retval = rd
    else:
        # return residual vector
        # IDEA: try angles instead of xys?
        diff_vecs_xy = calc_xy_all - meas_xyo_all[:, :2]
        diff_ome = rotations.angularDifference(calc_omes_all, meas_xyo_all[:, 2])
        retval = np.hstack([diff_vecs_xy, diff_ome.reshape(npts, 1)]).flatten()
        if return_value_flag == 1:
            # return scalar sum of squared residuals
            retval = sum(abs(retval))
        elif return_value_flag == 2:
            # return DOF-normalized chisq
            # TODO: check this calculation
            denom = 3 * npts - len(gFit) - 1.0
            if denom != 0:
                nu_fac = 1.0 / denom
            else:
                nu_fac = 1.0
            retval = nu_fac * sum(retval**2)
    return retval


def matchOmegas(
    xyo_det,
    hkls_idx,
    chi,
    rMat_c,
    bMat,
    wavelength,
    vInv=vInv_ref,
    beamVec=bVec_ref,
    etaVec=eta_ref,
    omePeriod=None,
):
    """
    For a given list of (x, y, ome) points, outputs the index into the results
    from oscillAnglesOfHKLs, including the calculated omega values.
    """
    # get omegas for rMat_s calculation
    if omePeriod is not None:
        meas_omes = rotations.mapAngle(xyo_det[:, 2], omePeriod)
    else:
        meas_omes = xyo_det[:, 2]

    oangs0, oangs1 = xfcapi.oscill_angles_of_hkls(
        hkls_idx.T,
        chi,
        rMat_c,
        bMat,
        wavelength,
        v_inv=vInv,
        beam_vec=beamVec,
        eta_vec=etaVec,
    )
    if np.any(np.isnan(oangs0)):
        # debugging
        # TODO: remove this
        # import pdb
        # pdb.set_trace()
        nanIdx = np.where(np.isnan(oangs0[:, 0]))[0]
        errorString = "Infeasible parameters for hkls:\n"
        for i in range(len(nanIdx)):
            errorString += "%d  %d  %d\n" % tuple(hkls_idx[:, nanIdx[i]])
        errorString += "you may need to deselect this hkl family."
        raise RuntimeError(errorString)
    else:
        # CAPI version gives vstacked angles... must be (2, nhkls)
        calc_omes = np.vstack([oangs0[:, 2], oangs1[:, 2]])
    if omePeriod is not None:
        calc_omes = np.vstack(
            [
                rotations.mapAngle(oangs0[:, 2], omePeriod),
                rotations.mapAngle(oangs1[:, 2], omePeriod),
            ]
        )
    # do angular difference
    diff_omes = rotations.angularDifference(meas_omes[None, :], calc_omes)
    choose_second = (diff_omes[1] < diff_omes[0]) | (
        np.isnan(diff_omes[0]) & ~np.isnan(diff_omes[1])
    )
    # Two alternatives need one comparison, not an argsort and two flattened
    # temporary arrays. Keep the first branch for ties and sort NaNs last.
    match_omes = np.vstack((~choose_second, choose_second))
    calc_omes = np.where(choose_second, calc_omes[1], calc_omes[0])

    return match_omes, calc_omes
