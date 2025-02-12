import logging
from typing import Optional

import lmfit
import numpy as np

from .lmfit_param_handling import add_engineering_constraints, create_instr_params, DEFAULT_EULER_CONVENTION, update_instrument_from_params, validate_params_list
from ....core.fitting.calibration.relative_constraints import create_relative_constraints, RelativeConstraints, RelativeConstraintsType

logger = logging.getLogger()
logger.setLevel('INFO')


def _normalized_ssqr(resd):
    return np.sum(resd * resd) / len(resd)


class InstrumentCalibrator:
    def __init__(self, *args, engineering_constraints=None,
                 set_refinements_from_instrument_flags=True,
                 euler_convention=DEFAULT_EULER_CONVENTION,
                 relative_constraints_type=RelativeConstraintsType.none):
        """
        Model for instrument calibration class as a function of

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        Notes
        -----
        Flags are set on calibrators
        """
        assert len(args) > 0, "must have at least one calibrator"
        self.calibrators = args
        for calib in self.calibrators:
            assert calib.instr is self.instr, \
                "all calibrators must refer to the same instrument"
        self._engineering_constraints = engineering_constraints
        self._relative_constraints = create_relative_constraints(
            relative_constraints_type, self.instr)
        self.euler_convention = euler_convention

        self.params = self.make_lmfit_params()
        if set_refinements_from_instrument_flags:
            self.instr.set_calibration_flags_to_lmfit_params(self.params)

        self.fitter = lmfit.Minimizer(self.minimizer_function,
                                      self.params,
                                      nan_policy='omit')

    def make_lmfit_params(self):
        params = create_instr_params(
            self.instr,
            euler_convention=self.euler_convention,
            relative_constraints=self.relative_constraints,
        )

        for calibrator in self.calibrators:
            # We pass the params to the calibrator so it can ensure it
            # creates unique parameter names. The calibrator will keep
            # track of the names it chooses itself.
            params += calibrator.create_lmfit_params(params)

        # Perform validation on the params before proceeding
        validate_params_list(params)

        params_dict = lmfit.Parameters()
        params_dict.add_many(*params)

        add_engineering_constraints(params_dict, self.engineering_constraints)
        return params_dict

    def update_all_from_params(self, params):
        # Update instrument and material from the lmfit parameters
        update_instrument_from_params(
            self.instr,
            params,
            self.euler_convention,
            self.relative_constraints,
        )

        for calibrator in self.calibrators:
            calibrator.update_from_lmfit_params(params)

    @property
    def instr(self):
        return self.calibrators[0].instr

    @property
    def tth_distortion(self):
        return self.calibrators[0].tth_distortion

    @tth_distortion.setter
    def tth_distortion(self, v):
        for calibrator in self.calibrators:
            calibrator.tth_distortion = v

    def minimizer_function(self, params):
        self.update_all_from_params(params)
        return self.residual()

    def residual(self):
        return np.hstack([x.residual() for x in self.calibrators])

    def minimize(self, method='least_squares', odict=None):
        if odict is None:
            odict = {}

        if method == 'least_squares':
            # Set defaults to the odict, if they are missing
            odict = {
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8,
                "verbose": 2,
                "max_nfev": 1000,
                "x_scale": "jac",
                "method": "trf",
                "jac": "3-point",
                **odict,
            }

            result = self.fitter.least_squares(self.params, **odict)
        else:
            result = self.fitter.scalar_minimize(method=method,
                                                 params=self.params,
                                                 max_nfev=50000,
                                                 **odict)

        return result

    @property
    def engineering_constraints(self):
        return self._engineering_constraints

    @engineering_constraints.setter
    def engineering_constraints(self, v):
        if v == self._engineering_constraints:
            return

        valid_settings = [
            None,
            'None',
            'TARDIS',
        ]
        if v not in valid_settings:
            valid_str = ', '.join(map(valid_settings, str))
            msg = (
                f'Invalid engineering constraint "{v}". Valid constraints '
                f'are: "{valid_str}"'
            )
            raise Exception(msg)

        self._engineering_constraints = v
        self.params = self.make_lmfit_params()

    @property
    def relative_constraints_type(self):
        return self._relative_constraints.type

    @relative_constraints_type.setter
    def relative_constraints_type(self, v: Optional[RelativeConstraintsType]):
        v = v if v is not None else RelativeConstraintsType.none

        current = getattr(self, '_relative_constraints', None)
        if current is None or current.type != v:
            self.relative_constraints = create_relative_constraints(
                v, self.instr)

    @property
    def relative_constraints(self) -> RelativeConstraints:
        return self._relative_constraints

    @relative_constraints.setter
    def relative_constraints(self, v: RelativeConstraints):
        self._relative_constraints = v
        self.params = self.make_lmfit_params()

    def reset_lmfit_params(self):
        self.params = self.make_lmfit_params()

    def reset_relative_constraint_params(self):
        # Set them back to zero.
        self.relative_constraints.reset()

    def run_calibration(self, odict):
        resd0 = self.residual()
        nrm_ssr_0 = _normalized_ssqr(resd0)

        result = self.minimize(odict=odict)

        resd1 = self.residual()

        nrm_ssr_1 = _normalized_ssqr(resd1)

        delta_r = 1. - nrm_ssr_1/nrm_ssr_0

        if delta_r > 0:
            logger.info('OPTIMIZATION SUCCESSFUL')
        else:
            logger.warning('no improvement in residual')

        logger.info('normalized initial ssr: %.4e' % nrm_ssr_0)
        logger.info('normalized final ssr: %.4e' % nrm_ssr_1)
        logger.info('change in resdiual: %.4e' % delta_r)

        self.params = result.params
        self.update_all_from_params(self.params)

        return result
