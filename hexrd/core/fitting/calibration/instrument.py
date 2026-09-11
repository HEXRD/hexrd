import logging
from typing import Any, Optional, Sequence

import lmfit
import numpy as np
from numpy.typing import NDArray

from .lmfit_param_handling import (
    add_engineering_constraints,
    create_instr_params,
    DEFAULT_EULER_CONVENTION,
    update_instrument_from_params,
    validate_params_list,
)
from .relative_constraints import (
    create_relative_constraints,
    RelativeConstraints,
    RelativeConstraintsType,
)

logger = logging.getLogger()
logger.setLevel('INFO')


def _normalized_ssqr(resd):
    return np.sum(resd * resd) / len(resd)


def _params_equal(a: Any, b: Any) -> bool:
    # Recursively compare relative-constraint params, which may nest dicts of
    # numpy arrays (a plain ``==`` raises on array truthiness).
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_params_equal(a[k], b[k]) for k in a)

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)

    return a == b


class InstrumentCalibrator:
    def __init__(
        self,
        *args,
        engineering_constraints=None,
        euler_convention=DEFAULT_EULER_CONVENTION,
        relative_constraints_type=RelativeConstraintsType.none,
    ):
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
            assert (
                calib.instr is self.instr
            ), "all calibrators must refer to the same instrument"
        self._engineering_constraints = engineering_constraints
        self._relative_constraints = create_relative_constraints(
            relative_constraints_type, self.instr
        )
        self.euler_convention = euler_convention

        self.params = self.make_lmfit_params()
        self.fitter = lmfit.Minimizer(
            self.minimizer_function, self.params, nan_policy='omit'
        )

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
            result = self.fitter.scalar_minimize(
                method=method, params=self.params, max_nfev=50000, **odict
            )

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
            self.relative_constraints = create_relative_constraints(v, self.instr)

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

        delta_r = 1.0 - nrm_ssr_1 / nrm_ssr_0

        if delta_r > 0:
            logger.debug('OPTIMIZATION SUCCESSFUL')
        else:
            logger.warning('no improvement in residual')

        logger.info('normalized initial ssr: %.4e' % nrm_ssr_0)
        logger.info('normalized final ssr: %.4e' % nrm_ssr_1)
        logger.info('change in resdiual: %.4e' % delta_r)

        self.params = result.params
        self.update_all_from_params(self.params)

        return result


class MultiInstrumentCalibrator:
    """Calibrate several instruments simultaneously with a shared geometry.

    Each entry is an ``InstrumentCalibrator`` describing one dataset (for
    example, one rotation scan), with its own instrument and its own
    sub-calibrators (grains, powder rings, ...). The detector geometry (panel
    tilts/translations/distortion and the beam) is shared across every
    instrument, while the oscillation stage (``instr.chi`` and ``instr.tvec``)
    and the sub-calibrator parameters (grains, lattice, ...) remain independent
    per dataset.

    Sharing is achieved through lmfit parameter names: the geometry parameters
    are created once (from the first calibrator) and every other instrument
    reads them by the same name. The stage parameters are created per dataset
    with a unique prefix so each dataset keeps its own rotation axis and sample
    offset. All residuals are stacked into a single vector and minimized
    together.
    """

    def __init__(
        self,
        calibrators: Sequence['InstrumentCalibrator'],
        labels: Optional[Sequence[str]] = None,
        engineering_constraints: Optional[str] = None,
    ):
        assert len(calibrators) > 0, "must have at least one calibrator"
        self.calibrators = list(calibrators)

        # The geometry (detectors + beam) is created only from calibrators[0]
        # and every other instrument reads it back by the same (unprefixed)
        # detector key / beam name. A mismatch would KeyError deep in the fit,
        # so require an identical detector layout across all calibrators.
        reference_detectors = set(self.calibrators[0].instr.detectors)
        reference_beams = set(self.calibrators[0].instr.beam_dict)
        for i, calib in enumerate(self.calibrators[1:], start=1):
            detectors = set(calib.instr.detectors)
            if detectors != reference_detectors:
                extra = detectors - reference_detectors
                missing = reference_detectors - detectors
                raise ValueError(
                    f"calibrator {i} has a different detector layout than "
                    f"calibrator 0; extra detectors: {sorted(extra)}, "
                    f"missing detectors: {sorted(missing)}"
                )

            beams = set(calib.instr.beam_dict)
            if beams != reference_beams:
                extra = beams - reference_beams
                missing = reference_beams - beams
                raise ValueError(
                    f"calibrator {i} has a different beam layout than "
                    f"calibrator 0; extra beams: {sorted(extra)}, "
                    f"missing beams: {sorted(missing)}"
                )

        # Only calibrators[0]'s relative_constraints actually take effect for
        # the shared geometry, so require every calibrator to agree on them.
        reference_constraints = self.calibrators[0].relative_constraints
        for i, calib in enumerate(self.calibrators[1:], start=1):
            constraints = calib.relative_constraints
            if type(constraints) is not type(
                reference_constraints
            ) or not _params_equal(
                constraints.params, reference_constraints.params
            ):
                raise ValueError(
                    f"calibrator {i} has different relative_constraints than "
                    f"calibrator 0; all calibrators must share the same "
                    f"relative_constraints"
                )

        if labels is None:
            labels = [f'scan_{i}' for i in range(len(self.calibrators))]
        assert len(labels) == len(self.calibrators), (
            "must have one label per calibrator"
        )
        if len(set(labels)) != len(labels):
            raise ValueError(f"labels must be unique, got: {labels}")
        self.labels = list(labels)

        self._engineering_constraints = engineering_constraints

        self.params = self.make_lmfit_params()

        # Sync every instrument to the shared (first calibrator) geometry so
        # the initial residual is consistent before the first minimizer step.
        self.update_all_from_params(self.params)

        self.fitter = lmfit.Minimizer(
            self.minimizer_function, self.params, nan_policy='omit'
        )

    def _stage_prefix(self, index: int) -> str:
        return f'{self.labels[index]}_'

    def make_lmfit_params(self) -> lmfit.Parameters:
        all_params = []
        for i, calib in enumerate(self.calibrators):
            # Geometry (beam + detectors) is shared, so it is only created for
            # the first instrument. Every other instrument references it by the
            # same parameter names.
            instr_params = create_instr_params(
                calib.instr,
                euler_convention=calib.euler_convention,
                relative_constraints=calib.relative_constraints,
                include_geometry=(i == 0),
                stage_prefix=self._stage_prefix(i),
            )
            all_params += instr_params

            # Sub-calibrator params (grains, lattice, ...). Threading the
            # accumulated list means shared material params dedupe while
            # per-dataset grain params stay unique.
            for sub_calibrator in calib.calibrators:
                all_params += sub_calibrator.create_lmfit_params(all_params)

        validate_params_list(all_params)

        params_dict = lmfit.Parameters()
        params_dict.add_many(*all_params)

        add_engineering_constraints(params_dict, self.engineering_constraints)
        return params_dict

    def update_all_from_params(self, params: lmfit.Parameters) -> None:
        for i, calib in enumerate(self.calibrators):
            update_instrument_from_params(
                calib.instr,
                params,
                calib.euler_convention,
                calib.relative_constraints,
                stage_prefix=self._stage_prefix(i),
            )

            for sub_calibrator in calib.calibrators:
                sub_calibrator.update_from_lmfit_params(params)

    def minimizer_function(
        self, params: lmfit.Parameters
    ) -> NDArray[np.float64]:
        self.update_all_from_params(params)
        return self.residual()

    def residual(self) -> NDArray[np.float64]:
        return np.hstack([calib.residual() for calib in self.calibrators])

    def minimize(
        self,
        method: str = 'least_squares',
        odict: Optional[dict[str, Any]] = None,
    ) -> lmfit.minimizer.MinimizerResult:
        if odict is None:
            odict = {}

        if method == 'least_squares':
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
            result = self.fitter.scalar_minimize(
                method=method, params=self.params, max_nfev=50000, **odict
            )

        return result

    @property
    def engineering_constraints(self) -> Optional[str]:
        return self._engineering_constraints

    def reset_lmfit_params(self) -> None:
        self.params = self.make_lmfit_params()

    def run_calibration(
        self, odict: Optional[dict[str, Any]] = None
    ) -> lmfit.minimizer.MinimizerResult:
        resd0 = self.residual()
        nrm_ssr_0 = _normalized_ssqr(resd0)

        result = self.minimize(odict=odict)

        resd1 = self.residual()
        nrm_ssr_1 = _normalized_ssqr(resd1)

        delta_r = 1.0 - nrm_ssr_1 / nrm_ssr_0

        if delta_r > 0:
            logger.debug('OPTIMIZATION SUCCESSFUL')
        else:
            logger.warning('no improvement in residual')

        logger.info('normalized initial ssr: %.4e' % nrm_ssr_0)
        logger.info('normalized final ssr: %.4e' % nrm_ssr_1)
        logger.info('change in resdiual: %.4e' % delta_r)

        self.params = result.params
        self.update_all_from_params(self.params)

        return result
