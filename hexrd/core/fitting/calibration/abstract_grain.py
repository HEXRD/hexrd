from abc import abstractmethod
import logging

import lmfit
import numpy as np

import hexrd.core.constants as cnst
from hexrd.core.rotations import angleAxisOfRotMat, RotMatEuler
from hexrd.core.transforms import xfcapi
from hexrd.core.utils.hkl import hkl_to_str, str_to_hkl

from .calibrator import Calibrator
from .lmfit_param_handling import create_grain_params, DEFAULT_EULER_CONVENTION, rename_to_avoid_collision

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

    @abstractmethod
    def residual(self):
        pass

    @abstractmethod
    def model(self):
        pass


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
