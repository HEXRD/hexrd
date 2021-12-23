import os

import numpy as np

from scipy.optimize import leastsq, least_squares

from hexrd import constants as cnst
from hexrd.fitting import fitpeak
from hexrd.fitting.peakfunctions import mpeak_nparams_dict
from hexrd import matrixutil as mutil
from hexrd.transforms import xfcapi

from . import grains as grainutil


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

# =============================================================================
# %% POWDER CALIBRATION
# =============================================================================

nfields_powder_data = 8


class PowderCalibrator(object):
    def __init__(self, instr, plane_data, img_dict, flags,
                 tth_tol=None, eta_tol=0.25,
                 pktype='pvoigt'):
        assert list(instr.detectors.keys()) == list(img_dict.keys()), \
            "instrument and image dict must have the same keys"
        self._instr = instr
        self._plane_data = plane_data
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
        currently only
            'gaussian', 'lorentzian,
            'pvoigt', 'split_pvoigt',
            or 'pink_beam_dcs'
        """
        assert x in ['gaussian',
                     'lorentzian',
                     'pvoigt',
                     'split_pvoigt',
                     'pink_beam_dcs'], \
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
                npdiv=2, collapse_eta=True, collapse_tth=False,
                do_interpolation=True)

    def _extract_powder_lines(self, fit_tth_tol=None, int_cutoff=1e-4):
        """
        return the RHS for the instrument DOF and image dict

        The format is a dict over detectors, each containing

        [index over ring sets]
            [index over azimuthal patch]
                [xy_meas, tth_meas, hkl, dsp_ref, eta_ref]

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
            pass
        powder_lines = self._interpolate_images()

        # GRAND LOOP OVER PATCHES
        rhs = dict.fromkeys(self.instr.detectors)
        for det_key, panel in self.instr.detectors.items():
            rhs[det_key] = []
            for i_ring, ringset in enumerate(powder_lines[det_key]):
                tmp = []
                if len(ringset) == 0:
                    continue
                else:
                    for angs, intensities in ringset:
                        # tth_centers = np.average(
                        #     np.vstack([angs[0][:-1], angs[0][1:]]),
                        #     axis=0)
                        # eta_ref = angs[1]
                        # int1d = np.sum(
                        #     np.array(intensities).squeeze(),
                        #     axis=0
                        # )
                        tth_centers = angs[0]
                        eta_ref = angs[1]
                        int1d = intensities[0]

                        # peak profile fitting
                        if len(dsp0[i_ring]) == 1:
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
                            if p[0] < int_cutoff or center_err > fit_tth_tol:
                                tmp.append(np.empty((0, nfields_powder_data)))
                                continue

                            # push back through mapping to cartesian (x, y)
                            xy_meas = panel.angles_to_cart(
                                [[tth_meas, eta_ref], ],
                                tvec_s=self.instr.tvec,
                                apply_distortion=True
                            )

                            # cat results
                            tmp.append(
                                np.hstack(
                                    [xy_meas.squeeze(),
                                     tth_meas,
                                     hkls[i_ring].squeeze(),
                                     dsp0[i_ring],
                                     eta_ref]
                                )
                            )
                        else:
                            # multiple peaks
                            tth_pred = 2.*np.arcsin(0.5*wlen/dsp0[i_ring])
                            npeaks = len(tth_pred)
                            eta_ref_tile = np.tile(eta_ref, npeaks)

                            # !!! these hueristics merit checking
                            fwhm_guess = self.plane_data.tThWidth/4.
                            center_bnd = self.plane_data.tThWidth/2./npeaks

                            p0, bnds = fitpeak.estimate_mpk_parms_1d(
                                    tth_pred, tth_centers, int1d,
                                    pktype=self.pktype, bgtype='linear',
                                    fwhm_guess=fwhm_guess,
                                    center_bnd=center_bnd
                                 )

                            p = fitpeak.fit_mpk_parms_1d(
                                    p0, tth_centers, int1d, self.pktype,
                                    npeaks, bgtype='linear', bnds=bnds
                                )

                            nparams_per_peak = mpeak_nparams_dict[self.pktype]
                            just_the_peaks = \
                                p[:npeaks*nparams_per_peak].reshape(
                                    npeaks, nparams_per_peak
                                )

                            # !!! this is where we can kick out bunk fits
                            tth_meas = just_the_peaks[:, 1]
                            center_err = abs(tth_meas - tth_pred)
                            if np.any(
                                    np.logical_or(
                                        just_the_peaks[:, 0] < int_cutoff,
                                        center_err > fit_tth_tol
                                    )
                            ):
                                tmp.append(np.empty((0, nfields_powder_data)))
                                continue

                            # push back through mapping to cartesian (x, y)
                            xy_meas = panel.angles_to_cart(
                                np.vstack([tth_meas, eta_ref_tile]).T,
                                tvec_s=self.instr.tvec,
                                apply_distortion=True
                            )

                            # cat results
                            tmp.append(
                                np.hstack(
                                    [xy_meas,
                                     tth_meas.reshape(npeaks, 1),
                                     hkls[i_ring],
                                     dsp0[i_ring].reshape(npeaks, 1),
                                     eta_ref_tile.reshape(npeaks, 1)]
                                )
                            )
                if len(tmp) == 0:
                    rhs[det_key].append(np.empty((0, nfields_powder_data)))
                else:
                    rhs[det_key].append(np.vstack(tmp))
                pass
            pass
        return rhs

    def _evaluate(self, reduced_params, data_dict, output='residual'):
        """
        Evaluate the powder diffraction model.

        Parameters
        ----------
        reduced_params : TYPE
            DESCRIPTION.
        data_dict : TYPE
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

        # build residual
        retval = []
        for det_key, panel in self.instr.detectors.items():
            if len(data_dict[det_key]) == 0:
                continue

            pdata = np.vstack(data_dict[det_key])
            if len(pdata) > 0:
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
                tth0 = 2.*np.arcsin(0.5*wlen/dsp0)

                # !!! get eta from mapped markers rather than ref
                # eta0 = pdata[:, -1]
                eta0 = updated_angles[:, 1]

                # map updated (tth0, eta0) back to cartesian coordinates
                tth_eta = np.vstack([tth0, eta0]).T
                calc_xy = panel.angles_to_cart(
                    tth_eta,
                    tvec_s=self.instr.tvec,
                    apply_distortion=True
                )

                # output
                if output == 'residual':
                    # retval.append(
                    #     (meas_xy.flatten() - calc_xy.flatten())
                    # )
                    retval.append(
                        updated_angles[:, 0].flatten() -  tth0.flatten()
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


class InstrumentCalibrator(object):
    def __init__(self, *args):
        assert len(args) > 0, \
            "must have at least one calibrator"
        self._calibrators = args
        self._instr = self._calibrators[0].instr

    @property
    def instr(self):
        return self._instr

    @property
    def calibrators(self):
        return self._calibrators

    # =========================================================================
    # METHODS
    # =========================================================================

    def run_calibration(self,
                        conv_tol=1e-4, max_iter=3, fit_tth_tol=None,
                        use_robust_optimization=False):
        """
        FIXME: only coding serial powder case to get things going.  Will
        eventually figure out how to loop over multiple calibrator classes.
        All will have a reference the same instrument, but some -- like single
        crystal -- will have to add parameters as well as contribute to the RHS
        """
        calib_class = self.calibrators[0]

        obj_func = calib_class.residual

        delta_r = np.inf
        step_successful = True
        iters = 0
        while (delta_r > conv_tol and step_successful) and iters < max_iter:
            data_dict = calib_class._extract_powder_lines(
                fit_tth_tol=fit_tth_tol)

            # grab reduced optimizaion parameter set
            x0 = self._instr.calibration_parameters[
                    self._instr.calibration_flags
                ]

            resd0 = obj_func(x0, data_dict)

            if use_robust_optimization:
                oresult = least_squares(
                    obj_func, x0, args=(data_dict, ),
                    method='trf', loss='soft_l1'
                )
                x1 = oresult['x']
            else:
                x1, cox_x, infodict, mesg, ierr = leastsq(
                    obj_func, x0, args=(data_dict, ),
                    full_output=True
                )
            resd1 = obj_func(x1, data_dict)

            nrm_ssr_0 = sum(resd0**2)/float(len(resd0))
            nrm_ssr_1 = sum(resd1**2)/float(len(resd1))

            delta_r = nrm_ssr_0 - nrm_ssr_1

            if delta_r > 0:
                print('OPTIMIZATION SUCCESSFUL')
                print('normalized initial ssr: %f\nnormalized final ssr: %f'
                      % (nrm_ssr_0, nrm_ssr_1))
                print(('change in resdiual: %f' % delta_r))

            else:
                print('no improvement in residual!!!')
                step_successful = False
            iters += 1
        return x1


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
    ngrains = len(grain_params)
    for ig, grain in enumerate(grain_params):
        pnames += [
            '{:>24s}'.format('grain %d expmap_c[0]' % ig),
            '{:>24s}'.format('grain %d expmap_c[0]' % ig),
            '{:>24s}'.format('grain %d expmap_c[0]' % ig),
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
        print("Set up to refine:")
        for i in np.where(refine_flags)[0]:
            print("\t%s = %1.7e" % (pnames[i], plist_full[i]))

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
            raise RuntimeError("solution not found: ierr = %d" % ierr)
        else:
            print("INFO: optimization fininshed successfully with ierr=%d"
                  % ierr)
            print("INFO: %s" % mesg)

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

        return fit_params, resd, sim_final


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

            rmat_s_arr = xfcapi.makeOscillRotMatArray(
                chi, np.ascontiguousarray(calc_omes_tmp)
            )
            calc_xy_tmp = xfcapi.gvecToDetectorXYArray(
                    ghat_c.T, rmat_d, rmat_s_arr, rmat_c,
                    tvec_d, tvec_s, tvec_c
            )
            if np.any(np.isnan(calc_xy_tmp)):
                print("infeasible parameters: "
                      + "may want to scale back finite difference step size")

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
            print("INFO: panel '%s', grain %d" % (det_key, grain_id))
            print("INFO: %d of %d reflections are indexed"
                  % (sum(valid_reflections), len(gtable))
                  )
            print("INFO: %d of %d"
                  % (sum(not_saturated), sum(valid_reflections)) +
                  " valid reflections be are below" +
                  " saturation threshold of %d" % (panel.saturation_level)
                  )

            # valid reflections index
            if refit_idx is None:
                idx = np.logical_and(valid_reflections, not_saturated)
                idx_0[det_key].append(idx)
            else:
                idx = refit_idx[det_key][ig]
                idx_0[det_key].append(idx)
                print("INFO: input reflection specify " +
                      "%d of %d total valid reflections"
                      % (sum(idx), len(gtable))
                      )

            hkls[det_key].append(gtable[idx, 2:5])
            meas_omes = gtable[idx, 12].reshape(sum(idx), 1)
            xyo_det[det_key].append(np.hstack([gtable[idx, -2:], meas_omes]))
    return hkls, xyo_det, idx_0
