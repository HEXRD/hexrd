import numpy as np

from hexrd import matrixutil as mutil

from .. import spectrum


nfields_powder_data = 8


class PowderCalibrator:
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
