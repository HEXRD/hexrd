from abc import abstractmethod
import logging

import lmfit
import numpy as np

import hexrd.constants as cnst
from hexrd import matrixutil as mutil
from hexrd.rotations import (
    angleAxisOfRotMat,
    angularDifference,
    RotMatEuler,
)
from hexrd.transforms import xfcapi
from hexrd.utils.hkl import hkl_to_str, str_to_hkl

from .calibrator import Calibrator
from .lmfit_param_handling import (
    create_grain_params,
    DEFAULT_EULER_CONVENTION,
    rename_to_avoid_collision,
)
from .. import grains as grainutil

logger = logging.getLogger(__name__)


class AbstractGrainCalibrator(Calibrator):
    def __init__(self, instr, material, grain_params,
                 default_refinements=None, calibration_picks=None,
                 euler_convention=DEFAULT_EULER_CONVENTION):
        self.instr = instr
        self.material = material
        self.grain_params = grain_params
        self.default_refinements = default_refinements
        self.euler_convention = euler_convention

        self.data_dict = None
        if calibration_picks is not None:
            self.calibration_picks = calibration_picks

        self.param_names = []

    @property
    @abstractmethod
    def name(self):
        pass

    def create_lmfit_params(self, current_params):
        params = create_grain_params(
            self.name,
            self.grain_params_euler,
            refinements=self.default_refinements,
        )

        # Ensure there are no name collisions
        params, _ = rename_to_avoid_collision(params, current_params)
        self.param_names = [x[0] for x in params]

        return params

    def update_from_lmfit_params(self, params_dict):
        grain_params = []
        for i, name in enumerate(self.param_names):
            grain_params.append(params_dict[name].value)

        self.grain_params_euler = np.asarray(grain_params)

    def fix_strain_to_identity(self, params_dict: lmfit.Parameters):
        identity = cnst.identity_6x1
        for i, name in enumerate(self.strain_param_names):
            param = params_dict[name]
            force_param_value(param, identity[i])
            param.value = identity[i]
            param.vary = False

    def fix_translation_to_origin(self, params_dict: lmfit.Parameters):
        origin = cnst.zeros_3
        for i, name in enumerate(self.translation_param_names):
            param = params_dict[name]
            force_param_value(param, origin[i])
            param.vary = False

    def fix_y_to_zero(self, params_dict: lmfit.Parameters):
        name = self.translation_param_names[1]
        param = params_dict[name]
        force_param_value(param, 0)
        param.vary = False

    @property
    def orientation_param_names(self) -> list[str]:
        return self.param_names[:3]

    @property
    def translation_param_names(self) -> list[str]:
        return self.param_names[3:6]

    @property
    def strain_param_names(self) -> list[str]:
        return self.param_names[6:]

    @property
    def grain_params_euler(self):
        # Grain parameters with orientation set using Euler angle convention
        if self.euler_convention is None:
            return self.grain_params

        grain_params = self.grain_params.copy()
        rme = RotMatEuler(np.zeros(3), **self.euler_convention)
        rme.rmat = xfcapi.make_rmat_of_expmap(grain_params[:3])
        grain_params[:3] = np.degrees(rme.angles)
        return grain_params

    @grain_params_euler.setter
    def grain_params_euler(self, v):
        # Grain parameters with orientation set using Euler angle convention
        grain_params = v.copy()
        if self.euler_convention is not None:
            rme = RotMatEuler(np.zeros(3,), **self.euler_convention)
            rme.angles = np.radians(grain_params[:3])
            phi, n = angleAxisOfRotMat(rme.rmat)
            grain_params[:3] = phi * n.flatten()

        self.grain_params = grain_params

    @property
    def plane_data(self):
        return self.material.planeData

    @property
    def bmatx(self):
        return self.plane_data.latVecOps['B']

    @property
    def calibration_picks(self):
        # Convert this from our internal data dict format
        picks = {}
        for det_key in self.instr.detectors:
            picks[det_key] = {}

            # find valid reflections and recast hkls to int
            xys = self.data_dict['pick_xys'][det_key]
            hkls = self.data_dict['hkls'][det_key]

            for hkl, xy in zip(hkls, xys):
                picks[det_key][hkl_to_str(hkl)] = xy

        return picks

    @calibration_picks.setter
    def calibration_picks(self, v):
        # Convert this to our internal data dict format
        data_dict = {
            'pick_xys': {},
            'hkls': {},
        }
        for det_key, det_picks in v.items():
            data_dict['hkls'][det_key] = [str_to_hkl(x) for x in det_picks]
            data_dict['pick_xys'][det_key] = list(det_picks.values())

        self.data_dict = data_dict

    @abstractmethod
    def autopick_points(self):
        pass

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
            [self.grain_params], self.instr, pick_xys_dict, pick_hkls_dict,
            self.bmatx, self.ome_period
        )

    def model(self):
        pick_hkls_dict, pick_xys_dict = self._evaluate()

        return sxcal_obj_func(
            [self.grain_params], self.instr, pick_xys_dict, pick_hkls_dict,
            self.bmatx, self.ome_period, sim_only=True
        )


def force_param_value(param: lmfit.Parameter, val: float):
    # This ensures the min/max are adjusted so the parameter can be set
    # We can't set the min/max to be exactly the same value, or lmfit
    # will panic.
    tol = 1e-4

    # Ensure we can set this
    if param.min > val:
        param.min = val - tol
    elif param.max < val:
        param.max = val + tol

    param.value = val


# Objective function for multigrain fitting
def sxcal_obj_func(grain_params, instr, xyo_det, hkls_idx,
                   bmat, ome_period, sim_only=False):
    ngrains = len(grain_params)

    # assign some useful params
    wavelength = instr.beam_wavelength
    bvec = instr.beam_vector
    chi = instr.chi
    tvec_s = instr.tvec

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
            if panel.distortion is not None:    # do unwarping
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

            match_omes, calc_omes_tmp = grainutil.matchOmegas(
                xyo, ghkls.T,
                chi, rmat_c, bmat, wavelength,
                vInv=vinv_s,
                beamVec=bvec,
                omePeriod=ome_period)

            rmat_s_arr = xfcapi.make_sample_rmat(
                chi, np.ascontiguousarray(calc_omes_tmp)
            )
            calc_xy_tmp = xfcapi.gvec_to_xy(
                    ghat_c.T, rmat_d, rmat_s_arr, rmat_c,
                    tvec_d, tvec_s, tvec_c
            )
            if np.any(np.isnan(calc_xy_tmp)):
                logger.warning("infeasible parameters: may want to scale back "
                               "finite difference step size")

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
            [diff_vecs_xy,
             diff_ome.reshape(npts_tot, 1)]
        ).flatten()
    return retval
