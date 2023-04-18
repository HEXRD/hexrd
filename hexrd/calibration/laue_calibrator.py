import numpy as np

from hexrd.matrixutil import findDuplicateVectors
from hexrd.fitting import fitpeak


class LaueCalibrator(object):
    def __init__(self, instr, plane_data, img_dict, flags, crystal_params,
                 min_energy=5., max_energy=25., tth_tol=2.0, eta_tol=2.0,
                 pktype='pvoigt'):
        assert list(instr.detectors.keys()) == list(img_dict.keys()), \
            "instrument and image dict must have the same keys"
        self._instr = instr
        self._plane_data = plane_data.makeNew()  # need a copy to munge
        self._plane_data.wavelength = max_energy  # force
        self._plane_data.exclusions = None  # don't need exclusions for Laue
        self._img_dict = img_dict
        self._params = np.asarray(crystal_params, dtype=float)
        self._full_params = np.hstack(
            [self._instr.calibration_parameters, self._params]
        )
        assert len(flags) == len(self._full_params), \
            "flags must have %d elements" % len(self._full_params)
        self._flags = flags

        # !!! scalar cutoffs for now
        # TODO: make a list or spectral input
        self._min_energy = min_energy
        self._max_energy = max_energy

        # for polar interpolation
        self._tth_tol = tth_tol
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
    def min_energy(self):
        return self._min_energy

    @min_energy.setter
    def min_energy(self, x):
        assert np.isscalar(x), "min_energy must be a scalar value"
        self._min_energy = x

    @property
    def max_energy(self):
        return self._max_energy

    @max_energy.setter
    def max_energy(self, x):
        assert np.isscalar(x), "max_energy must be a scalar value"
        self._max_energy = x

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

    # =========================================================================
    # METHODS
    # =========================================================================

    def _extract_peaks(self):
        refl_patches = dict.fromkeys(self.instr.detectors)
        for ip_key, ip in self.instr.detectors.items():
            patches = xrdutil.make_reflection_patches(
                ip.config_dict(chi=self.instr.chi, tvec=self.instr.tvec,
                    beam_vector=self.instr.beam_vector),
                valid_angs[ip_key], ip.angularPixelSize(valid_xy[ip_key]),
                rmat_c=xfcapi.makeRotMatOfExpMap(expmap_c),
                tth_tol=tth_tol, eta_tol=eta_tol,
                npdiv=npdiv, quiet=True)
            refl_patches[ip_key] = list(patches)


"""for labeling"""
labelStructure = ndimage.generate_binary_structure(2, 1)
refl_dict = dict.fromkeys(instr.detectors)
for ip_key, ip in instr.detectors.items():
    reflInfoList = []
    img = self.img_dict[ip_key]
    native_area = ip.pixel_area
    num_patches = len(refl_patches[ip_key])
    meas_xy = np.nan*np.ones((num_patches, 2))
    meas_angs = np.nan*np.ones((num_patches, 2))
    for iRefl, patch in enumerate(refl_patches[ip_key]):
        # check for overrun
        irow = patch[-1][0]
        jcol = patch[-1][1]
        if np.any([irow < 0, irow >= ip.rows, jcol < 0, jcol >= ip.cols]):
            continue
        if not np.all(ip.clip_to_panel(np.vstack([patch[1][0].flatten(),
                                                  patch[1][1].flatten()]).T)[1]
                      ):
            continue
        # use nearest interpolation
        spot_data = img[irow, jcol] * patch[3] * npdiv**2 / native_area
        fsd = snip2d_py3.snip2d(spot_data, w=5, order=2, numiter=2)
        spot_data -= fsd

        spot_data -= np.amin(spot_data)
        patch_size = spot_data.shape

        sigmax = np.round(0.5*np.min(spot_data.shape)/sigma_to_FWHM)

        # optional gaussian smoothing
        if do_smoothing:
            spot_data = filters.gaussian(spot_data, stdev)

        if use_blob_detection:
            spot_data_scl = 2.*spot_data/np.max(spot_data) - 1.

            # Compute radii in the 3rd column.
            blobs_log = blob_log(spot_data_scl,
                                 min_sigma=5,
                                 max_sigma=max(sigmax, 6),
                                 num_sigma=int(sigmax-5),
                                 threshold=blob_thresh, overlap=0.1)
            numPeaks = len(blobs_log)
        else:
            labels, numPeaks = ndimage.label(
                spot_data > np.percentile(spot_data, 99),
                structure=labelStructure
            )
            slabels = np.arange(1, numPeaks + 1)
        tth_edges = patch[0][0][0, :]
        eta_edges = patch[0][1][:, 0]
        delta_tth = tth_edges[1] - tth_edges[0]
        delta_eta = eta_edges[1] - eta_edges[0]
        if numPeaks > 0:
            peakId = iRefl
            if use_blob_detection:
                coms = blobs_log[:, :2]
            else:
                coms = np.array(
                    ndimage.center_of_mass(
                        spot_data, labels=labels, index=slabels
                        )
                    )
            if numPeaks > 1:
                #
                center = np.r_[spot_data.shape]*0.5
                com_diff = coms - np.tile(center, (numPeaks, 1))
                closest_peak_idx = np.argmin(np.sum(com_diff**2, axis=1))
                #
            else:
                closest_peak_idx = 0
                pass   # end multipeak conditional
            #
            coms = coms[closest_peak_idx]
            '''
            #
            fig, ax = plt.subplots()
            ax.imshow(spot_data, cmap=cm.hot, interpolation='none')
            ax.plot(blobs_log[:, 1], blobs_log[:, 0], 'c+')
            ax.plot(coms[1], coms[0], 'ms', mfc='none')
            fig.suptitle("panel %s, reflection %d" % (ip_key, iRefl))
            print("%s, %d, (%.2f, %.2f), (%d, %d)"
                  % (ip_key, iRefl, coms[0], coms[1],
                     patch_size[0], patch_size[1]))
            #
            '''
            assert(coms.ndim == 1), "oops"
            #
            if fit_peaks:
                sigm = 0.2*np.min(spot_data.shape)
                if use_blob_detection:
                    sigm = min(blobs_log[closest_peak_idx, 2], sigm)
                y0, x0 = coms.flatten()
                ampl = float(spot_data[int(y0), int(x0)])
                # y0, x0 = 0.5*np.array(spot_data.shape)
                # ampl = np.max(spot_data)
                a_par = c_par = 0.5/float(sigm**2)
                b_par = 0.
                bgx = bgy = 0.
                bkg = np.min(spot_data)
                params = [ampl, a_par, b_par, c_par, x0, y0, bgx, bgy, bkg]
                #
                result = leastsq(gaussian_2d, params, args=(spot_data))
                #
                fit_par = result[0]
                #
                coms = np.array([fit_par[5], fit_par[4]])
                print("%s, %d, (%.2f, %.2f), (%d, %d)"
                      % (ip_key, iRefl, coms[0], coms[1],
                         patch_size[0], patch_size[1]))
                row_cen = 0.1*patch_size[0]
                col_cen = 0.1*patch_size[1]
                #
                if np.any(
                    [coms[0] < row_cen, coms[0] >= patch_size[0] - row_cen,
                     coms[1] < col_cen, coms[1] >= patch_size[1] - col_cen]
                ):
                    continue
                if (fit_par[0] < 1.):
                    continue
                #
                if plot_fits:
                    fit_g = (
                        gaussian_2d(fit_par, spot_data) + spot_data.flatten()
                    ).reshape(spot_data.shape)
                    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
                    fig.suptitle("panel %s, reflection %d" % (ip_key, iRefl))
                    ax[0].imshow(spot_data, cmap=cm.hot, interpolation='none')
                    ax[0].plot(fit_par[4], fit_par[5], 'c+')
                    ax[1].imshow(fit_g, cmap=cm.hot, interpolation='none')
                    ax[2].imshow(spot_data - fit_g,
                                 cmap=cm.hot, interpolation='none')

                # intensities
                spot_intensity, int_err = nquad(
                    gaussian_2d_int,
                    [[0., 2.*y0], [0., 2.*x0]],
                    args=fit_par)
                pass
            com_angs = np.hstack([
                tth_edges[0] + (0.5 + coms[1])*delta_tth,
                eta_edges[0] + (0.5 + coms[0])*delta_eta
                ])

            # grab intensities
            if not fit_peaks:
                if use_blob_detection:
                    spot_intensity = 10
                    max_intensity = 10
                else:
                    spot_intensity = np.sum(
                        spot_data[labels == slabels[closest_peak_idx]]
                    )
                    max_intensity = np.max(
                        spot_data[labels == slabels[closest_peak_idx]]
                    )
            else:
                max_intensity = np.max(spot_data)
            # need xy coords
            cmv = np.atleast_2d(np.hstack([com_angs, omega]))
            gvec_c = xfcapi.angles_to_gvec(
                cmv,
                chi=instr.chi,
                rmat_c=rmat_c,
                beam_vec=instr.beam_vector)
            new_xy = xfcapi.gvec_to_xy(
                gvec_c,
                ip.rmat, rmat_s, rmat_c,
                ip.tvec, instr.tvec, tvec_c,
                beam_vec=instr.beam_vector)
            meas_xy[iRefl, :] = new_xy
            meas_angs[iRefl, :] = com_angs
        else:
            peakId = -999
            #
            spot_intensity = np.nan
            max_intensity = np.nan
            pass

        reflInfoList.append([peakId, valid_hkls[ip_key][:, iRefl],
                             (spot_intensity, max_intensity),
                             valid_energy[ip_key][iRefl],
                             valid_angs[ip_key][iRefl, :],
                             meas_angs[iRefl, :],
                             meas_xy[iRefl, :]])
        pass
    reflInfo = np.array(
        [tuple(i) for i in reflInfoList],
        dtype=__reflInfo_dtype)
    refl_dict[ip_key] = reflInfo


    def _update_from_reduced(self, reduced_parameters):
        # first update instrument from input parameters
        full_params = np.asarray(self.full_params)
        full_params[self.flags] = reduced_params

        self.instr.update_from_parameter_list(full_params[:self.npi])
        self.params = full_params[self.npi:]


    def model(self, reduced_parameters):

        self._update_from_reduced(reduced_parameters)

        # Laue pattern simulation
        laue_simulation = self.instr.simulate_laue_pattern(
            self.plane_data,
            minEnergy=self.min_energy,
            maxEnergy=self.max_energy,
            grain_params=grain_params
        )

        # parse simulation results
        valid_xy, valid_hkls, valid_energy, valid_angs = \
            parse_laue_simulation(laue_simulation)

        return valid_xy
