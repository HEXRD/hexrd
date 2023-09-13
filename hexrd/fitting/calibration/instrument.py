import logging

import numpy as np
from scipy.optimize import leastsq, least_squares

logger = logging.getLogger()
logger.setLevel('INFO')


def _normalized_ssqr(resd):
    return np.sum(resd*resd)/len(resd)


class InstrumentCalibrator:
    def __init__(self, *args):
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
        assert len(args) > 0, \
            "must have at least one calibrator"
        self._calibrators = args
        self._instr = self._calibrators[0].instr
        self.npi = len(self._instr.calibration_parameters)
        self._full_params = self._instr.calibration_parameters
        calib_data = []
        for calib in self._calibrators:
            assert calib.instr is self._instr, \
                "all calibrators must refer to the same instrument"
            self._full_params = np.hstack([self._full_params, calib.params])
            calib_data.append(calib.calibration_data)
        self._calibration_data = calib_data

    @property
    def instr(self):
        return self._instr

    @property
    def calibrators(self):
        return self._calibrators

    @property
    def calibration_data(self):
        return self._calibration_data

    @property
    def flags(self):
        # additional params come next
        flags = [self.instr.calibration_flags, ]
        for calib_class in self.calibrators:
            flags.append(calib_class.flags[calib_class.npi:])
        return np.hstack(flags)

    @property
    def full_params(self):
        return self._full_params

    @full_params.setter
    def full_params(self, x):
        assert len(x) == len(self._full_params), \
            "input must have length %d; you gave %d" \
            % (len(self._full_params), len(x))
        self._full_params = x

    @property
    def reduced_params(self):
        return self.full_params[self.flags]

    # =========================================================================
    # METHODS
    # =========================================================================

    def _reduced_params_flag(self, cidx):
        assert cidx >= 0 and cidx < len(self.calibrators), \
            "index must be in %s" % str(np.arange(len(self.calibrators)))

        calib_class = self.calibrators[cidx]

        # instrument params come first
        npi = calib_class.npi

        # additional params come next
        cparams_flags = [calib_class.flags[:npi], ]
        for i, calib_class in enumerate(self.calibrators):
            if i == cidx:
                cparams_flags.append(calib_class.flags[npi:])
            else:
                cparams_flags.append(np.zeros(calib_class.npe, dtype=bool))
        return np.hstack(cparams_flags)

    def extract_points(self, fit_tth_tol, int_cutoff=1e-4):
        # !!! list in the same order as dict looping
        calib_data_list = []
        for calib_class in self.calibrators:
            calib_class._extract_powder_lines(
                fit_tth_tol=fit_tth_tol, int_cutoff=int_cutoff
            )
            calib_data_list.append(calib_class.calibration_data)
        self._calibration_data = calib_data_list
        return calib_data_list

    def residual(self, x0):
        # !!! list in the same order as dict looping
        resd = []
        for i, calib_class in enumerate(self.calibrators):
            # !!! need to grab the param set
            #     specific to this calibrator class
            fp = np.array(self.full_params)  # copy full_params
            fp[self.flags] = x0  # assign new global values
            this_x0 = fp[self._reduced_params_flag(i)]  # select these
            resd.append(calib_class.residual(this_x0))
        return np.hstack(resd)

    def run_calibration(self,
                        fit_tth_tol=None, int_cutoff=1e-4,
                        conv_tol=1e-4, max_iter=5,
                        use_robust_optimization=False):
        """


        Parameters
        ----------
        fit_tth_tol : TYPE, optional
            DESCRIPTION. The default is None.
        int_cutoff : TYPE, optional
            DESCRIPTION. The default is 1e-4.
        conv_tol : TYPE, optional
            DESCRIPTION. The default is 1e-4.
        max_iter : TYPE, optional
            DESCRIPTION. The default is 5.
        use_robust_optimization : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        x1 : TYPE
            DESCRIPTION.

        """

        delta_r = np.inf
        step_successful = True
        iter_count = 0
        nrm_ssr_prev = np.inf
        rparams_prev = np.array(self.reduced_params)
        while delta_r > conv_tol \
            and step_successful \
                and iter_count < max_iter:

            # extract data
            check_cal = [i is None for i in self.calibration_data]
            if np.any(check_cal) or iter_count > 0:
                self.extract_points(
                    fit_tth_tol=fit_tth_tol,
                    int_cutoff=int_cutoff
                )

            # grab reduced params for optimizer
            x0 = np.array(self.reduced_params)  # !!! copy
            resd0 = self.residual(x0)
            nrm_ssr_0 = _normalized_ssqr(resd0)
            if nrm_ssr_0 > nrm_ssr_prev:
                logger.warning('No residual improvement; exiting')
                self.full_params = rparams_prev
                break

            if use_robust_optimization:
                oresult = least_squares(
                    self.residual, x0,
                    method='trf', loss='soft_l1'
                )
                x1 = oresult['x']
            else:
                x1, cox_x, infodict, mesg, ierr = leastsq(
                    self.residual, x0,
                    full_output=True
                )

            # eval new residual
            # !!! I thought this should update the underlying class params?
            resd1 = self.residual(x1)

            nrm_ssr_1 = _normalized_ssqr(resd1)

            delta_r = 1. - nrm_ssr_1/nrm_ssr_0

            if delta_r > 0:
                logger.info('OPTIMIZATION SUCCESSFUL')
                logger.info('normalized initial ssr: %.4e' % nrm_ssr_0)
                logger.info('normalized final ssr: %.4e' % nrm_ssr_1)
                logger.info('change in resdiual: %.4e' % delta_r)
                # FIXME: WHY IS THIS UPDATE NECESSARY?
                #        Thought the cal to self.residual below did this, but
                #        appeasr not to.
                new_params = np.array(self.full_params)
                new_params[self.flags] = x1
                self.full_params = new_params

                nrm_ssr_prev = nrm_ssr_0
                rparams_prev = np.array(self.full_params)  # !!! careful
                rparams_prev[self.flags] = x0
            else:
                logger.warning('no improvement in residual; exiting')
                step_successful = False
                break

            iter_count += 1

        # handle exit condition in case step failed
        if not step_successful:
            x1 = x0
            _ = self.residual(x1)

        # update the full_params
        # FIXME: this class is still hard-coded for one calibrator
        fp = np.array(self.full_params, dtype=float)
        fp[self.flags] = x1
        self.full_params = fp

        return x1
