import logging

import numpy as np

from hexrd.core import matrixutil as mutil
from hexrd.core.rotations import angularDifference
from hexrd.core.transforms import xfcapi
from hexrd.core import xrdutil

from .abstract_grain import AbstractGrainCalibrator
from .lmfit_param_handling import (
    DEFAULT_EULER_CONVENTION,
)
from .. import grains as grainutil

logger = logging.getLogger(__name__)


class GrainCalibrator(AbstractGrainCalibrator):
    """This is for HEDM grain calibration"""

    type = 'grain'

    def __init__(
        self,
        instr,
        material,
        grain_params,
        ome_period,
        index=0,
        default_refinements=None,
        calibration_picks=None,
        euler_convention=DEFAULT_EULER_CONVENTION,
    ):
        super().__init__(
            instr,
            material,
            grain_params,
            default_refinements,
            calibration_picks,
            euler_convention,
        )
        self.ome_period = ome_period
        self.index = index

    @property
    def name(self):
        return f'{self.material.name}_{self.index}'

    def autopick_points(self):
        # We could call `pull_spots()` here to perform auto-picking.
        raise NotImplementedError

    def _evaluate(self):
        data_dict = self.data_dict

        # grab reflection data from picks input
        pick_hkls_dict = {}
        pick_xys_dict = {}
        for det_key in self.instr.detectors:
            # find valid reflections and recast hkls to int
            xys = np.asarray(data_dict['pick_xys'][det_key], dtype=float)
            hkls = np.asarray(data_dict['hkls'][det_key], dtype=int)

            valid_idx = ~np.isnan(xys[:, 0])

            # fill local dicts
            pick_hkls_dict[det_key] = [np.atleast_2d(hkls[valid_idx, :])]
            pick_xys_dict[det_key] = [np.atleast_2d(xys[valid_idx, :])]

        return pick_hkls_dict, pick_xys_dict

    def residual(self):
        pick_hkls_dict, pick_xys_dict = self._evaluate()

        return sxcal_obj_func(
            [self.grain_params],
            self.instr,
            pick_xys_dict,
            pick_hkls_dict,
            self.bmatx,
            self.ome_period,
        )

    def model(self):
        pick_hkls_dict, pick_xys_dict = self._evaluate()

        return sxcal_obj_func(
            [self.grain_params],
            self.instr,
            pick_xys_dict,
            pick_hkls_dict,
            self.bmatx,
            self.ome_period,
            sim_only=True,
        )


# Objective function for multigrain fitting
def sxcal_obj_func(
    grain_params, instr, xyo_det, hkls_idx, bmat, ome_period, sim_only=False
):
    ngrains = len(grain_params)

    # assign some useful params
    wavelength = instr.beam_wavelength
    bvec = instr.beam_vector
    chi = instr.chi
    tvec_s = instr.tvec
    energy_correction = instr.energy_correction

    # right now just stuck on the end and assumed
    # to all be the same length... FIX THIS
    xy_unwarped = {}
    meas_omes = {}
    calc_omes = {}
    calc_xy = {}

    # loop over panels
    npts_tot = 0
    for det_key, panel in instr.detectors.items():
        rmat_d = panel.rmat
        tvec_d = panel.tvec

        xy_unwarped[det_key] = []
        meas_omes[det_key] = []
        calc_omes[det_key] = []
        calc_xy[det_key] = []

        for ig, grain in enumerate(grain_params):
            ghkls = hkls_idx[det_key][ig]
            xyo = xyo_det[det_key][ig]

            npts_tot += len(xyo)

            xy_unwarped[det_key].append(xyo[:, :2])
            meas_omes[det_key].append(xyo[:, 2])
            if panel.distortion is not None:  # do unwarping
                xy_unwarped[det_key][ig] = panel.distortion.apply(
                    xy_unwarped[det_key][ig]
                )

            # transform G-vectors:
            # 1) convert inv. stretch tensor from MV notation in to 3x3
            # 2) take reciprocal lattice vectors from CRYSTAL to SAMPLE frame
            # 3) apply stretch tensor
            # 4) normalize reciprocal lattice vectors in SAMPLE frame
            # 5) transform unit reciprocal lattice vetors back to CRYSAL frame
            rmat_c = xfcapi.make_rmat_of_expmap(grain[:3])
            tvec_c = grain[3:6]
            vinv_s = grain[6:]
            gvec_c = np.dot(bmat, ghkls.T)
            vmat_s = mutil.vecMVToSymm(vinv_s)
            ghat_s = mutil.unitVector(np.dot(vmat_s, np.dot(rmat_c, gvec_c)))
            ghat_c = np.dot(rmat_c.T, ghat_s)

            # Apply an energy correction according to grain position
            corrected_wavelength = xrdutil.apply_correction_to_wavelength(
                wavelength,
                energy_correction,
                tvec_s,
                tvec_c,
            )

            match_omes, calc_omes_tmp = grainutil.matchOmegas(
                xyo, ghkls.T,
                chi, rmat_c, bmat, corrected_wavelength,
                vInv=vinv_s,
                beamVec=bvec,
                omePeriod=ome_period,
            )

            rmat_s_arr = xfcapi.make_sample_rmat(
                chi, np.ascontiguousarray(calc_omes_tmp)
            )
            calc_xy_tmp = xfcapi.gvec_to_xy(
                ghat_c.T, rmat_d, rmat_s_arr, rmat_c, tvec_d, tvec_s, tvec_c
            )
            if np.any(np.isnan(calc_xy_tmp)):
                logger.warning(
                    "infeasible parameters: may want to scale back "
                    "finite difference step size"
                )

            calc_omes[det_key].append(calc_omes_tmp)
            calc_xy[det_key].append(calc_xy_tmp)

    # return values
    if sim_only:
        retval = {}
        for det_key in calc_xy.keys():
            # ??? calc_xy is always 2-d
            retval[det_key] = []
            for ig in range(ngrains):
                retval[det_key].append(
                    np.vstack(
                        [calc_xy[det_key][ig].T, calc_omes[det_key][ig]]
                    ).T
                )
    else:
        meas_xy_all = []
        calc_xy_all = []
        meas_omes_all = []
        calc_omes_all = []
        for det_key in xy_unwarped.keys():
            meas_xy_all.append(np.vstack(xy_unwarped[det_key]))
            calc_xy_all.append(np.vstack(calc_xy[det_key]))
            meas_omes_all.append(np.hstack(meas_omes[det_key]))
            calc_omes_all.append(np.hstack(calc_omes[det_key]))
        meas_xy_all = np.vstack(meas_xy_all)
        calc_xy_all = np.vstack(calc_xy_all)
        meas_omes_all = np.hstack(meas_omes_all)
        calc_omes_all = np.hstack(calc_omes_all)

        diff_vecs_xy = calc_xy_all - meas_xy_all
        diff_ome = angularDifference(calc_omes_all, meas_omes_all)
        retval = np.hstack(
            [diff_vecs_xy, diff_ome.reshape(npts_tot, 1)]
        ).flatten()
    return retval
