import numpy as np

from hexrd.matrixutil import findDuplicateVectors
from hexrd.fitting import fitpeak


class PowderCalibrator(object):
    def __init__(self, instr, plane_data, img_dict, flags,
                 tth_tol=None, eta_tol=0.25,
                 pktype='pvoigt'):
        assert list(instr.detectors.keys()) == list(img_dict.keys()), \
            "instrument and image dict must have the same keys"
        self._instr = instr
        self._plane_data = plane_data.makeNew()  # need a copy to munge
        self._plane_data.wavelength = self._instr.beam_energy  # force
        self._img_dict = img_dict
        self._params = np.asarray(self.plane_data.lparms, dtype=float)
        self._full_params = np.hstack(
            [self._instr.calibration_parameters, self._params]
        )
        assert len(flags) == len(self._full_params), \
            "flags must have %d elements" % len(self._full_params)
        self._flags = flags

        # for polar interpolation
        self._tth_tol = tth_tol or np.degrees(plane_data.tThWidth)
        self._eta_tol = eta_tol

        # for peak fitting
        # ??? fitting only, or do alternative peak detection?
        self._pktype = pktype

    @property
    def npi(self):
        return len(self._instr.calibration_parameters)

    @property
    def instr(self):
        return self._instr

    @property
    def plane_data(self):
        self._plane_data.wavelength = self._instr.beam_energy
        return self._plane_data

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
        currently only 'gaussian', 'lorentzian, 'pvoigt' or 'split_pvoigt'
        """
        assert x in ['gaussian', 'lorentzian', 'pvoigt', 'split_pvoigt'], \
            "pktype '%s' not understood"
        self._pktype = x

    def _interpolate_images(self):
        """
        returns the iterpolated powder line data from the images in img_dict

        ??? interpolation necessary?
        """
        return self.instr.extract_line_positions(
                self.plane_data, self.img_dict,
                tth_tol=self.tth_tol, eta_tol=self.eta_tol,
                npdiv=2, collapse_eta=False, collapse_tth=False,
                do_interpolation=True)

    def _extract_powder_lines(self, fit_tth_tol=None):
        """
        return the RHS for the instrument DOF and image dict

        The format is a dict over detectors, each containing

        [index over ring sets]
            [index over azimuthal patch]
                [xy_meas, tth_meas, tth_ref, eta_ref]

        FIXME: can not yet handle tth ranges with multiple peaks!
        """
        if fit_tth_tol is None:
            fit_tth_tol = self.tth_tol/4.
        fit_tth_tol = np.radians(fit_tth_tol)
        # ideal tth
        wlen = self.instr.beam_wavelength
        dsp_ideal = np.atleast_1d(self.plane_data.getPlaneSpacings())
        hkls_ref = self.plane_data.hkls.T
        dsp0 = []
        hkls = []
        for idx in self.plane_data.getMergedRanges()[0]:
            if len(idx) > 1:
                eqv, uidx = findDuplicateVectors(np.atleast_2d(dsp_ideal[idx]))
                if len(uidx) > 1:
                    raise NotImplementedError("can not handle multipeak yet")
                else:
                    # if here, only degenerate ring case
                    uidx = idx[0]
            else:
                uidx = idx[0]
            dsp0.append(dsp_ideal[uidx])
            hkls.append(hkls_ref[uidx])

        powder_lines = self._interpolate_images()

        # GRAND LOOP OVER PATCHES
        rhs = dict.fromkeys(self.instr.detectors)
        for det_key, panel in self.instr.detectors.items():
            rhs[det_key] = []
            for i_ring, ringset in enumerate(powder_lines[det_key]):
                tmp = []
                for angs, intensities in ringset:
                    tth_centers = np.average(
                        np.vstack([angs[0][:-1], angs[0][1:]]),
                        axis=0)
                    eta_ref = angs[1]
                    int1d = np.sum(np.array(intensities).squeeze(), axis=0)

                    # peak profile fitting
                    p0 = fitpeak.estimate_pk_parms_1d(
                            tth_centers, int1d, self.pktype
                         )

                    p = fitpeak.fit_pk_parms_1d(
                            p0, tth_centers, int1d, self.pktype
                        )

                    # !!! this is where we can kick out bunk fits
                    tth_meas = p[1]
                    tth_pred = 2.*np.arcsin(0.5*wlen/dsp0[i_ring])
                    center_err = abs(tth_meas - tth_pred)
                    if p[0] < 0.01 or center_err > fit_tth_tol:
                        tmp.append(np.empty((0, 5)))
                        continue
                    xy_meas = panel.angles_to_cart([[tth_meas, eta_ref], ])

                    # distortion
                    if panel.distortion is not None:
                        xy_meas = panel.distortion.apply_inverse(xy_meas)

                    # cat results
                    tmp.append(
                        np.hstack(
                            [xy_meas.squeeze(),
                             tth_meas,
                             hkls[i_ring],
                             dsp0[i_ring],
                             eta_ref]
                        )
                    )
                    pass
                if len(tmp) == 0:
                    rhs[det_key].append(np.empty((0, 5)))
                else:
                    rhs[det_key].append(np.vstack(tmp))
                pass
            pass
        return rhs

    def _evaluate(self, reduced_params, data_dict, output='residual'):
        """
        """
        # need this for dsp
        bmat = self.plane_data.latVecOps['B']

        # first update instrument from input parameters
        full_params = np.asarray(self.full_params)
        full_params[self.flags] = reduced_params

        self.instr.update_from_parameter_list(full_params[:self.npi])
        self.params = full_params[self.npi:]

        wlen = self.instr.beam_wavelength

        # build residual
        retval = []
        for det_key, panel in self.instr.detectors.items():
            pdata = np.vstack(data_dict[det_key])
            if len(pdata) > 0:
                hkls = pdata[:, 3:6]
                gvecs = np.dot(hkls, bmat.T)
                dsp0 = 1./np.sqrt(np.sum(gvecs*gvecs, axis=1))
                # dsp0 = pdata[:, -2]
                eta0 = pdata[:, -1]

                # derive reference tth
                tth0 = 2.*np.arcsin(0.5*wlen/dsp0)
                calc_xy = panel.angles_to_cart(np.vstack([tth0, eta0]).T)

                # distortion if applicable, from ideal --> warped
                if panel.distortion is not None:
                    calc_xy = panel.distortion.apply_inverse(calc_xy)

                if output == 'residual':
                    retval.append(
                        (pdata[:, :2].flatten() - calc_xy.flatten())
                    )
                elif output == 'model':
                    retval.append(
                        calc_xy.flatten()
                    )
                else:
                    raise RuntimeError("unrecognized output flag '%s'"
                                       % output)
            else:
                continue
        return np.hstack(retval)

    def residual(self, reduced_params, data_dict):
        return self._evaluate(reduced_params, data_dict)

    def model(self, reduced_params, data_dict):
        return self._evaluate(reduced_params, data_dict, output='model')
