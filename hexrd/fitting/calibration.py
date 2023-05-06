import logging
import os

import numpy as np

from scipy.optimize import leastsq, least_squares

from hexrd import constants as cnst
from hexrd import matrixutil as mutil
from hexrd.transforms import xfcapi

from . import grains as grainutil
from . import spectrum

logger = logging.getLogger()
logger.setLevel('INFO')


# =============================================================================
# %% PARAMETERS
# =============================================================================

# instrument
instr_flags_DFLT = np.ones(7, dtype=bool)

# panel
panel_flags_DFLT = np.ones(6, dtype=bool)

# grains
grain_flags_DFLT = np.array(
    [1, 1, 1,
     1, 0, 1,
     0, 0, 0, 0, 0, 0],
    dtype=bool
)

nfields_powder_data = 8

ext_eta_tol = np.radians(5.)  # for HEDM cal, may make this a user param

# =============================================================================
# %% POWDER CALIBRATION
# =============================================================================


def _normalized_ssqr(resd):
    return np.sum(resd*resd)/len(resd)


class PowderCalibrator(object):
    def __init__(self, instr, plane_data, img_dict, flags,
                 tth_tol=None, eta_tol=0.25,
                 fwhm_estimate=None, min_pk_sep=1e-3, min_ampl=0.,
                 pktype='pvoigt', bgtype='linear',
                 tth_distortion=None):
        assert list(instr.detectors.keys()) == list(img_dict.keys()), \
            "instrument and image dict must have the same keys"
        self._instr = instr
        self._plane_data = plane_data
        self._tth_distortion = tth_distortion
        self._fwhm_estimate = fwhm_estimate
        self._min_pk_sep = min_pk_sep
        self._min_ampl = min_ampl
        self._plane_data.wavelength = self._instr.beam_energy  # force
        self._img_dict = img_dict
        self._params = np.asarray(plane_data.lparms, dtype=float)
        self._full_params = np.hstack(
            [self._instr.calibration_parameters, self._params]
        )
        assert len(flags) == len(self._full_params), \
            "flags must have %d elements" % len(self._full_params)
        self._flags = flags

        nparams_instr = len(self._instr.calibration_parameters)
        # nparams_extra = len(self._params)
        # nparams = nparams_instr + nparams_extra
        self._instr.calibration_flags = self._flags[:nparams_instr]

        # for polar interpolation
        if tth_tol is None:
            self._tth_tol = np.degrees(plane_data.tThWidth)
        else:
            self._tth_tol = tth_tol
            self._plane_data.tThWidth = np.radians(tth_tol)
        self._eta_tol = eta_tol

        # for peak fitting
        # ??? fitting only, or do alternative peak detection?
        self._pktype = pktype
        self._bgtype = bgtype

        # container for calibration data
        self._calibration_data = None

    @property
    def npi(self):
        return len(self._instr.calibration_parameters)

    @property
    def instr(self):
        return self._instr

    @property
    def plane_data(self):
        self._plane_data.wavelength = self._instr.beam_energy
        self._plane_data.tThWidth = np.radians(self.tth_tol)
        return self._plane_data

    @property
    def tth_distortion(self):
        return self._tth_distortion

    @property
    def img_dict(self):
        return self._img_dict

    @property
    def tth_tol(self):
        return self._tth_tol

    @tth_tol.setter
    def tth_tol(self, x):
        assert np.isscalar(x), "tth_tol must be a scalar value"
        self._tth_tol = x

    @property
    def eta_tol(self):
        return self._eta_tol

    @eta_tol.setter
    def eta_tol(self, x):
        assert np.isscalar(x), "eta_tol must be a scalar value"
        self._eta_tol = x

    @property
    def fwhm_estimate(self):
        return self._fwhm_estimate

    @fwhm_estimate.setter
    def fwhm_estimate(self, x):
        if x is not None:
            assert np.isscalar(x), "fwhm_estimate must be a scalar value"
        self._fwhm_estimate = x

    @property
    def min_pk_sep(self):
        return self._min_pk_sep

    @min_pk_sep.setter
    def min_pk_sep(self, x):
        if x is not None:
            assert x > 0., "min_pk_sep must be greater than zero"
        self._min_pk_sep = x

    @property
    def min_ampl(self):
        return self._min_ampl

    @min_ampl.setter
    def min_ampl(self, x):
        if x is not None:
            assert x > 0., "min_ampl must be greater than zero"
        self._min_ampl = x

    @property
    def spectrum_kwargs(self):
        return dict(pktype=self.pktype,
                    bgtype=self.bgtype,
                    fwhm_init=self.fwhm_estimate,
                    min_ampl=self.min_ampl,
                    min_pk_sep=self.min_pk_sep)

    @property
    def calibration_data(self):
        return self._calibration_data

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        x = np.atleast_1d(x)
        if len(x) != len(self.plane_data.lparms):
            raise RuntimeError("params must have %d elements"
                               % len(self.plane_data.lparms))
        self._params = x
        self._plane_data.lparms = x

    @property
    def full_params(self):
        return self._full_params

    @property
    def npe(self):
        return len(self._params)

    @property
    def flags(self):
        return self._flags

    @flags.setter
    def flags(self, x):
        x = np.atleast_1d(x)
        nparams_instr = len(self.instr.calibration_parameters)
        nparams_extra = len(self.params)
        nparams = nparams_instr + nparams_extra
        if len(x) != nparams:
            raise RuntimeError("flags must have %d elements" % nparams)
        self._flags = np.asarrasy(x, dtype=bool)
        self._instr.calibration_flags = self._flags[:nparams_instr]

    @property
    def pktype(self):
        return self._pktype

    @pktype.setter
    def pktype(self, x):
        """
        currently only
            'gaussian', 'lorentzian,
            'pvoigt', 'split_pvoigt',
            or 'pink_beam_dcs'
        """
        assert x in spectrum._function_dict_1d, \
            "pktype '%s' not understood"
        self._pktype = x

    @property
    def bgtype(self):
        return self._bgtype

    @bgtype.setter
    def bgtype(self, x):
        """
        currently only
            'gaussian', 'lorentzian,
            'pvoigt', 'split_pvoigt',
            or 'pink_beam_dcs'
        """
        assert x in spectrum._function_dict_1d, \
            "pktype '%s' not understood"
        self._bgtype = x

    def _extract_powder_lines(self, fit_tth_tol=5., int_cutoff=1e-4):
        """
        return the RHS for the instrument DOF and image dict

        The format is a dict over detectors, each containing

        [index over ring sets]
            [index over azimuthal patch]
                [xy_meas, tth_meas, hkl, dsp_ref, eta_ref]

        FIXME: can not yet handle tth ranges with multiple peaks!
        """
        # ideal tth
        dsp_ideal = np.atleast_1d(self.plane_data.getPlaneSpacings())
        hkls_ref = self.plane_data.hkls.T
        dsp0 = []
        hkls = []
        for idx in self.plane_data.getMergedRanges()[0]:
            if len(idx) > 1:
                eqv, uidx = mutil.findDuplicateVectors(
                    np.atleast_2d(dsp_ideal[idx])
                )
                if len(uidx) < len(idx):
                    # if here, at least one peak is degenerate
                    uidx = np.asarray(idx)[uidx]
                else:
                    uidx = np.asarray(idx)
            else:
                uidx = np.asarray(idx)
            dsp0.append(dsp_ideal[uidx])
            hkls.append(hkls_ref[uidx])

        # Perform interpolation and fitting
        fitting_kwargs = {
            'int_cutoff': int_cutoff,
            'fit_tth_tol': fit_tth_tol,
            'spectrum_kwargs': self.spectrum_kwargs,
        }
        kwargs = {
            'plane_data': self.plane_data,
            'imgser_dict': self.img_dict,
            'tth_tol': self.tth_tol,
            'eta_tol': self.eta_tol,
            'npdiv': 2,
            'collapse_eta': True,
            'collapse_tth': False,
            'do_interpolation': True,
            'do_fitting': True,
            'fitting_kwargs': fitting_kwargs,
            'tth_distortion': self.tth_distortion,
        }
        powder_lines = self.instr.extract_line_positions(**kwargs)

        # Now loop over the ringsets and convert to the calibration format
        rhs = {}
        for det_key, panel in self.instr.detectors.items():
            rhs[det_key] = []
            for i_ring, ringset in enumerate(powder_lines[det_key]):
                this_dsp0 = dsp0[i_ring]
                this_hkl = hkls[i_ring]
                npeaks = len(this_dsp0)

                ret = []
                for angs, intensities, tth_meas in ringset:
                    if len(intensities) == 0:
                        continue

                    # We only run this on one image. Grab that one.
                    tth_meas = tth_meas[0]
                    if tth_meas is None:
                        continue

                    # Convert to radians
                    tth_meas = np.radians(tth_meas)

                    # reference eta
                    eta_ref_tile = np.tile(angs[1], npeaks)

                    # push back through mapping to cartesian (x, y)
                    xy_meas = panel.angles_to_cart(
                        np.vstack([tth_meas, eta_ref_tile]).T,
                        tvec_s=self.instr.tvec,
                        apply_distortion=True,
                    )

                    # cat results
                    output = np.hstack([
                                xy_meas,
                                tth_meas.reshape(npeaks, 1),
                                this_hkl,
                                this_dsp0.reshape(npeaks, 1),
                                eta_ref_tile.reshape(npeaks, 1),
                             ])
                    ret.append(output)

                if not ret:
                    ret.append(np.empty((0, nfields_powder_data)))

                rhs[det_key].append(np.vstack(ret))

        # assign attribute
        self._calibration_data = rhs

    def _evaluate(self, reduced_params,
                  calibration_data=None, output='residual'):
        """
        Evaluate the powder diffraction model.

        Parameters
        ----------
        reduced_params : TYPE
            DESCRIPTION.
        calibration_data : TYPE
            DESCRIPTION.
        output : TYPE, optional
            DESCRIPTION. The default is 'residual'.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # first update full parameters from input reduced parameters
        # TODO: make this process a class method
        full_params = np.asarray(self.full_params)
        full_params[self.flags] = reduced_params

        # !!! properties update from parameters
        self.instr.update_from_parameter_list(full_params[:self.npi])
        self.params = full_params[self.npi:]

        # need this for dsp
        bmat = self.plane_data.latVecOps['B']
        wlen = self.instr.beam_wavelength

        # calibration data
        if calibration_data is None:
            if self.calibration_data is None:
                raise RuntimeError(
                    "calibration data has not been provided; " +
                    "run extraction method or assign table from picks."
                )
        else:
            self._calibration_data = calibration_data

        # build residual
        retval = np.array([], dtype=float)
        for det_key, panel in self.instr.detectors.items():
            if len(self.calibration_data[det_key]) == 0:
                continue
            else:
                # recast as array
                pdata = np.vstack(self.calibration_data[det_key])

                """
                Here is the strategy:
                    1. remap the feature points from raw cartesian to
                       (tth, eta) under the current mapping
                    2. use the lattice and hkls to calculate the ideal tth0
                    3. push the (tth0, eta) values back through the mapping to
                       raw cartesian coordinates
                    4. build residual on the measured and recalculated (x, y)
                """
                # push measured (x, y) ring points through current mapping
                # to (tth, eta)
                meas_xy = pdata[:, :2]
                updated_angles, _ = panel.cart_to_angles(
                    meas_xy,
                    tvec_s=self.instr.tvec,
                    apply_distortion=True
                )

                # derive ideal tth positions from additional ring point info
                hkls = pdata[:, 3:6]
                gvecs = np.dot(hkls, bmat.T)
                dsp0 = 1./np.sqrt(np.sum(gvecs*gvecs, axis=1))

                # updated reference Bragg angles
                tth0 = 2.*np.arcsin(0.5*wlen/dsp0)

                # !!! get eta from mapped markers rather than ref
                # eta0 = pdata[:, -1]
                eta0 = updated_angles[:, 1]

                # apply tth distortion
                if self.tth_distortion is not None:
                    # !!! sd has ref to detector so is updated
                    sd = self.tth_distortion[det_key]
                    tmp = sd.apply(meas_xy, return_nominal=False)
                    corr_angs = tmp + np.vstack([tth0, np.zeros_like(tth0)]).T
                    tth0, eta0 = corr_angs.T
                    pass

                # map updated (tth0, eta0) back to cartesian coordinates
                tth_eta = np.vstack([tth0, eta0]).T

                # output
                if output == 'residual':
                    # retval = np.append(
                    #     retval,
                    #     meas_xy.flatten() - calc_xy.flatten()
                    # )
                    retval = np.append(
                        retval,
                        updated_angles[:, 0].flatten() - tth0.flatten()
                    )
                elif output == 'model':
                    calc_xy = panel.angles_to_cart(
                        tth_eta,
                        tvec_s=self.instr.tvec,
                        apply_distortion=True
                    )
                    retval = np.append(
                        retval,
                        calc_xy.flatten()
                    )
                else:
                    raise RuntimeError(
                        "unrecognized output flag '%s'"
                        % output
                    )

        return retval

    def residual(self, reduced_params, calibration_data=None):
        return self._evaluate(reduced_params,
                              calibration_data=calibration_data,
                              output='residual')

    def model(self, reduced_params, calibration_data=None):
        return self._evaluate(reduced_params,
                              calibration_data=calibration_data,
                              output='model')


class InstrumentCalibrator(object):
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

# =============================================================================
# %% STRUCTURE-LESS CALIBRATION
# =============================================================================
class StructureLessCalibrator():
    """
    this class implements the equivalent of the
    powder calibrator but without constraining
    the optimization to a structure. in this 
    implementation, the location of the constant
    two theta line that a set of points lie on
    is also an optimization parameter.

    unlike the previous implementations, this routine
    is based on the lmfit module to implement the 
    more complicated constraints for the TARDIS box
    """
    def __init__(self, 
                 instr,
                 data,
                 tth_distortion=None):

        self._instr = instr
        self._data = data

    def make_lmfit_params(self):
        self.params = lmfit.Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        all_params = []
        add_instr_params(all_params)
        nrng = len(self.data)
        add_tth_parameters(nrng, self.data, all_params)
        all_params = tuple(all_params)
        self.params.add_many(*all_params)

    def add_instr_params(self, parms_list):
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        instr = self.instr
        parms_list.append(('beam_energy',instr.beam_energy, 
                            False, instr.beam_energy-0.2, 
                            instr.beam_energy+0.2))
        azim, pol = calc_angles_from_beam_vec(instr.beam_vector)
        parms_list.append(('beam_polar', pol, False, pol-2, pol+2))
        parms_list.append(('beam_azimuth', azim, False, azim-2, azim+2))
        parms_list.append(('instr_chi', np.degrees(instr.chi), 
                           False, instr.chi-1, 
                           instr.chi+1))
        parms_list.append(('instr_tvec_x', instr.tvec[0], False, -np.inf, np.inf))
        parms_list.append(('instr_tvec_y', instr.tvec[1], False, -np.inf, np.inf))
        parms_list.append(('instr_tvec_z', instr.tvec[2], False, -np.inf, np.inf))
        for det, panel in instr.detectors.items():
            parms_list.append((f'{det}_tilt_x', panel.tilt[0],
                               False, panel.tilt[0]-0.1, panel.tilt[0]+0.1))
            parms_list.append((f'{det}_tilt_y', panel.tilt[1],
                               False, panel.tilt[1]-0.1, panel.tilt[0]+0.1))
            parms_list.append((f'{det}_tilt_z', panel.tilt[2],
                               False, panel.tilt[2]-0.1, panel.tilt[0]+0.1))
            parms_list.append((f'{det}_tvec_x', panel.tvec[0], False, -np.inf, np.inf))
            parms_list.append((f'{det}_tvec_y', panel.tvec[1], False, -np.inf, np.inf))
            parms_list.append((f'{det}_tvec_z', panel.tvec[2], False, -np.inf, np.inf))
            if instr.detectors['ge3'].distortion is not None:
                p = instr.detectors['ge3'].distortion.params
                for ii,pp in enumerate(p):
                    parms_list.append((f'{det}_distortion_param_{ii}',pp,
                                       False, -np.inf, np.inf))

    def add_tth_parameters(parms_list):
        for ii in range(self.nrings):
            val = np.mean(data[ii][:,2])
            parms_list.append((f'DS_ring_{ii}', 
                               val, 
                               True, 
                               val-np.radians(3.),
                               val+np.radians(3.)))

    def calc_residual(self):
        self.instr.update_from_lmfit_parameter_list(self.params)
        residual = np.empty([0,])
        for ii,rng in enumerate(self.data):
            tth_rng = self.params[f'DSring_{ii}']
            meas_xy = rng[:, :2]
            for det, panel in self.instr.detectors.items():
                updated_angles, _ = panel.cart_to_angles(
                                    meas_xy,
                                    tvec_s=instr.tvec,
                                    apply_distortion=True)
                tth_updated = updated_angles[:,0]
                delta_tth = tth_updated - tth_rng
                residual = np.concatenate((residual, delta_tth))
        return residual

    def set_minimizer(self):
        self.fitter = lmfit.Minimizer(self.calc_residual, 
                                      self.params)

    def run_calibration(self, odict=None):
        """
        odict is the optionas dictionary
        """
        fdict = {
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8,
                "verbose": 2,
                "max_nfev": 1000,
                "x_scale": "jac",
                "method": "trf",
                "jac": "3-point",
                }
        for k, v in odict.items():
            if k in fdict:
                fdict[k] = v
            else:
                fdict.update({k, v})
        return self.fitter.least_squares(**fdict)

    @property
    def nrings(self):
        return len(self.data)

    @property
    def instr(self):
        return self._instr

    @instr.setter
    def instr(self, ins):
        self._instr = ins
        self.make_lmfit_params()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, dat):
        self._data = dat
        self.make_lmfit_params()

    @property
    def residual(self):
        return self.calc_residual()
    

# =============================================================================
# %% LAUE CALIBRATION
# =============================================================================


# =============================================================================
# %% COMPOSITE CALIBRATION (WITH PICKS)
# =============================================================================


# =============================================================================
# %% MULTIGRAIN CALIBRATION
# =============================================================================

def calibrate_instrument_from_sx(
        instr, grain_params, bmat, xyo_det, hkls_idx,
        param_flags=None, grain_flags=None,
        ome_period=None,
        xtol=cnst.sqrt_epsf, ftol=cnst.sqrt_epsf,
        factor=10., sim_only=False, use_robust_lsq=False):
    """
    arguments xyo_det, hkls_idx are DICTs over panels

    """
    grain_params = np.atleast_2d(grain_params)
    ngrains = len(grain_params)
    pnames = generate_parameter_names(instr, grain_params)

    # reset parameter flags for instrument as specified
    if param_flags is None:
        param_flags = instr.calibration_flags
    else:
        # will throw an AssertionError if wrong length
        instr.calibration_flags = param_flags

    # re-map omegas if need be
    if ome_period is not None:
        for det_key in instr.detectors:
            for ig in range(ngrains):
                xyo_det[det_key][ig][:, 2] = xfcapi.mapAngle(
                        xyo_det[det_key][ig][:, 2],
                        ome_period
                )

    # first grab the instrument parameters
    # 7 global
    # 6*num_panels for the detectors
    # num_panels*ndp in case of distortion
    plist_full = instr.calibration_parameters

    # now handle grains
    # reset parameter flags for grains as specified
    if grain_flags is None:
        grain_flags = np.tile(grain_flags_DFLT, ngrains)

    plist_full = np.concatenate(
        [plist_full, np.hstack(grain_params)]
    )
    plf_copy = np.copy(plist_full)

    # concatenate refinement flags
    refine_flags = np.hstack([param_flags, grain_flags])
    plist_fit = plist_full[refine_flags]
    fit_args = (plist_full,
                param_flags, grain_flags,
                instr, xyo_det, hkls_idx,
                bmat, ome_period)
    if sim_only:
        return sxcal_obj_func(
            plist_fit, plist_full,
            param_flags, grain_flags,
            instr, xyo_det, hkls_idx,
            bmat, ome_period,
            sim_only=True)
    else:
        logger.info("Set up to refine:")
        for i in np.where(refine_flags)[0]:
            logger.info("\t%s = %1.7e" % (pnames[i], plist_full[i]))

        # run optimization
        if use_robust_lsq:
            result = least_squares(
                sxcal_obj_func, plist_fit, args=fit_args,
                xtol=xtol, ftol=ftol,
                loss='soft_l1', method='trf'
            )
            x = result.x
            resd = result.fun
            mesg = result.message
            ierr = result.status
        else:
            # do least squares problem
            x, cov_x, infodict, mesg, ierr = leastsq(
                sxcal_obj_func, plist_fit, args=fit_args,
                factor=factor, xtol=xtol, ftol=ftol,
                full_output=1
            )
            resd = infodict['fvec']
        if ierr not in [1, 2, 3, 4]:
            raise RuntimeError(f"solution not found: {ierr=}")
        else:
            logger.info(f"optimization fininshed successfully with {ierr=}")
            logger.info(mesg)

        # ??? output message handling?
        fit_params = plist_full
        fit_params[refine_flags] = x

        # run simulation with optimized results
        sim_final = sxcal_obj_func(
            x, plist_full,
            param_flags, grain_flags,
            instr, xyo_det, hkls_idx,
            bmat, ome_period,
            sim_only=True)

        # ??? reset instrument here?
        instr.update_from_parameter_list(fit_params)

        # report final
        logger.info("Optimization Reults:")
        for i in np.where(refine_flags)[0]:
            logger.info("\t%s = %1.7e --> %1.7e"
                        % (pnames[i], plf_copy[i], fit_params[i]))

        return fit_params, resd, sim_final


def generate_parameter_names(instr, grain_params):
    pnames = [
        '{:>24s}'.format('beam energy'),
        '{:>24s}'.format('beam azimuth'),
        '{:>24s}'.format('beam polar'),
        '{:>24s}'.format('chi'),
        '{:>24s}'.format('tvec_s[0]'),
        '{:>24s}'.format('tvec_s[1]'),
        '{:>24s}'.format('tvec_s[2]'),
    ]

    for det_key, panel in instr.detectors.items():
        pnames += [
            '{:>24s}'.format('%s tilt[0]' % det_key),
            '{:>24s}'.format('%s tilt[1]' % det_key),
            '{:>24s}'.format('%s tilt[2]' % det_key),
            '{:>24s}'.format('%s tvec[0]' % det_key),
            '{:>24s}'.format('%s tvec[1]' % det_key),
            '{:>24s}'.format('%s tvec[2]' % det_key),
        ]
        # now add distortion if there
        if panel.distortion is not None:
            for j in range(len(panel.distortion.params)):
                pnames.append(
                    '{:>24s}'.format('%s dparam[%d]' % (det_key, j))
                )

    grain_params = np.atleast_2d(grain_params)
    for ig, grain in enumerate(grain_params):
        pnames += [
            '{:>24s}'.format('grain %d xi[0]' % ig),
            '{:>24s}'.format('grain %d xi[1]' % ig),
            '{:>24s}'.format('grain %d xi[2]' % ig),
            '{:>24s}'.format('grain %d tvec_c[0]' % ig),
            '{:>24s}'.format('grain %d tvec_c[1]' % ig),
            '{:>24s}'.format('grain %d tvec_c[2]' % ig),
            '{:>24s}'.format('grain %d vinv_s[0]' % ig),
            '{:>24s}'.format('grain %d vinv_s[1]' % ig),
            '{:>24s}'.format('grain %d vinv_s[2]' % ig),
            '{:>24s}'.format('grain %d vinv_s[3]' % ig),
            '{:>24s}'.format('grain %d vinv_s[4]' % ig),
            '{:>24s}'.format('grain %d vinv_s[5]' % ig)
        ]

    return pnames


def sxcal_obj_func(plist_fit, plist_full,
                   param_flags, grain_flags,
                   instr, xyo_det, hkls_idx,
                   bmat, ome_period,
                   sim_only=False, return_value_flag=None):
    """
    """
    npi = len(instr.calibration_parameters)
    NP_GRN = 12

    # stack flags and force bool repr
    refine_flags = np.array(
        np.hstack([param_flags, grain_flags]),
        dtype=bool)

    # fill out full parameter list
    # !!! no scaling for now
    plist_full[refine_flags] = plist_fit

    # instrument update
    instr.update_from_parameter_list(plist_full)

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

    # grain params
    grain_params = plist_full[npi:]
    if np.mod(len(grain_params), NP_GRN) != 0:
        raise RuntimeError("parameter list length is not consistent")
    ngrains = len(grain_params) // NP_GRN
    grain_params = grain_params.reshape((ngrains, NP_GRN))

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
                pass

            # transform G-vectors:
            # 1) convert inv. stretch tensor from MV notation in to 3x3
            # 2) take reciprocal lattice vectors from CRYSTAL to SAMPLE frame
            # 3) apply stretch tensor
            # 4) normalize reciprocal lattice vectors in SAMPLE frame
            # 5) transform unit reciprocal lattice vetors back to CRYSAL frame
            rmat_c = xfcapi.makeRotMatOfExpMap(grain[:3])
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
            pass
        pass

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
            pass
        meas_xy_all = np.vstack(meas_xy_all)
        calc_xy_all = np.vstack(calc_xy_all)
        meas_omes_all = np.hstack(meas_omes_all)
        calc_omes_all = np.hstack(calc_omes_all)

        diff_vecs_xy = calc_xy_all - meas_xy_all
        diff_ome = xfcapi.angularDifference(calc_omes_all, meas_omes_all)
        retval = np.hstack(
            [diff_vecs_xy,
             diff_ome.reshape(npts_tot, 1)]
        ).flatten()
        if return_value_flag == 1:
            retval = sum(abs(retval))
        elif return_value_flag == 2:
            denom = npts_tot - len(plist_fit) - 1.
            if denom != 0:
                nu_fac = 1. / denom
            else:
                nu_fac = 1.
            nu_fac = 1 / (npts_tot - len(plist_fit) - 1.)
            retval = nu_fac * sum(retval**2)
    return retval


def parse_reflection_tables(cfg, instr, grain_ids, refit_idx=None):
    """
    make spot dictionaries
    """
    hkls = {}
    xyo_det = {}
    idx_0 = {}
    for det_key, panel in instr.detectors.items():
        hkls[det_key] = []
        xyo_det[det_key] = []
        idx_0[det_key] = []
        for ig, grain_id in enumerate(grain_ids):
            spots_filename = os.path.join(
                cfg.analysis_dir, os.path.join(
                    det_key, 'spots_%05d.out' % grain_id
                )
            )

            # load pull_spots output table
            gtable = np.loadtxt(spots_filename, ndmin=2)
            if len(gtable) == 0:
                gtable = np.nan*np.ones((1, 17))

            # apply conditions for accepting valid data
            valid_reflections = gtable[:, 0] >= 0  # is indexed
            not_saturated = gtable[:, 6] < panel.saturation_level
            # throw away extremem etas
            p90 = xfcapi.angularDifference(gtable[:, 8], cnst.piby2)
            m90 = xfcapi.angularDifference(gtable[:, 8], -cnst.piby2)
            accept_etas = np.logical_or(p90 > ext_eta_tol,
                                        m90 > ext_eta_tol)
            logger.info(f"panel '{det_key}', grain {grain_id}")
            logger.info(f"{sum(valid_reflections)} of {len(gtable)} "
                        "reflections are indexed")
            logger.info(f"{sum(not_saturated)} of {sum(valid_reflections)}"
                        " valid reflections be are below" +
                        f" saturation threshold of {panel.saturation_level}")
            logger.info(f"{sum(accept_etas)} of {len(gtable)}"
                        " reflections be are greater than " +
                        f" {np.degrees(ext_eta_tol)} from the rotation axis")

            # valid reflections index
            if refit_idx is None:
                idx = np.logical_and(
                    valid_reflections,
                    np.logical_and(not_saturated, accept_etas)
                )
                idx_0[det_key].append(idx)
            else:
                idx = refit_idx[det_key][ig]
                idx_0[det_key].append(idx)
                logger.info(f"input reflection specify {sum(idx)} of "
                            f"{len(gtable)} total valid reflections")

            hkls[det_key].append(gtable[idx, 2:5])
            meas_omes = gtable[idx, 12].reshape(sum(idx), 1)
            xyo_det[det_key].append(np.hstack([gtable[idx, -2:], meas_omes]))
    return hkls, xyo_det, idx_0
