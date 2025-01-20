import copy
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.integrate import nquad
from scipy.optimize import leastsq
from skimage import filters
from skimage.feature import blob_log

# TODO: Resolve extra-workflow-dependency
from hexrd.hedm import xrdutil
from hexrd.core.constants import fwhm_to_sigma
from hexrd.core.instrument import switch_xray_source
from hexrd.core.rotations import angleAxisOfRotMat, RotMatEuler
from hexrd.core.transforms import xfcapi
from hexrd.core.utils.hkl import hkl_to_str, str_to_hkl

# TODO: Resolve extra-workflow-dependency
from ....core.fitting.calibration.calibrator import Calibrator
from ....core.fitting.calibration.abstract_grain import AbstractGrainCalibrator
from ....core.fitting.calibration.lmfit_param_handling import create_grain_params, DEFAULT_EULER_CONVENTION, rename_to_avoid_collision


class LaueCalibrator(AbstractGrainCalibrator):
    """A Laue calibrator "is-a" specific case for a grain calibrator.

    Just like a grain calibrator, a Laue calibrator is calibrating
    grain parameters.

    There are some unique properties for Laue, though, such as having a
    varying energy range rather than a constant energy value. Also, we
    do not utilize any omega periods.
    """
    type = 'laue'

    def __init__(self, instr, material, grain_params, default_refinements=None,
                 min_energy=5, max_energy=25, tth_distortion=None,
                 calibration_picks=None,
                 euler_convention=DEFAULT_EULER_CONVENTION,
                 xray_source: Optional[str] = None):
        super().__init__(
            instr, material, grain_params, default_refinements,
            calibration_picks, euler_convention,
        )
        self.energy_cutoffs = [min_energy, max_energy]
        self.xray_source = xray_source

        self._tth_distortion = tth_distortion
        self._update_tth_distortion_panels()

    @property
    def name(self):
        return self.material.name

    @property
    def tth_distortion(self):
        return self._tth_distortion

    @tth_distortion.setter
    def tth_distortion(self, v):
        self._tth_distortion = v
        self._update_tth_distortion_panels()

    def _update_tth_distortion_panels(self):
        # Make sure the panels in the tth distortion are the same
        # as those on the instrument, so their beam vectors get modified
        # accordingly.
        if self._tth_distortion is None:
            return

        self._tth_distortion = copy.deepcopy(self._tth_distortion)
        for det_key, obj in self._tth_distortion.items():
            obj.panel = self.instr.detectors[det_key]

    @property
    def energy_cutoffs(self):
        return self._energy_cutoffs

    @energy_cutoffs.setter
    def energy_cutoffs(self, x):
        assert len(x) == 2, "input must have 2 elements"
        assert x[1] > x[0], "first element must be < than second"
        self._energy_cutoffs = x
        self.plane_data.wavelength = self.energy_cutoffs[-1]
        self.plane_data.exclusions = None

    def autopick_points(self, raw_img_dict, tth_tol=5., eta_tol=5.,
                        npdiv=2, do_smoothing=True, smoothing_sigma=2,
                        use_blob_detection=True, blob_threshold=0.25,
                        fit_peaks=True, min_peak_int=1., fit_tth_tol=0.1):
        """
        Parameters
        ----------
        raw_img_dict : TYPE
            DESCRIPTION.
        tth_tol : TYPE, optional
            DESCRIPTION. The default is 5..
        eta_tol : TYPE, optional
            DESCRIPTION. The default is 5..
        npdiv : TYPE, optional
            DESCRIPTION. The default is 2.
        do_smoothing : TYPE, optional
            DESCRIPTION. The default is True.
        smoothing_sigma : TYPE, optional
            DESCRIPTION. The default is 2.
        use_blob_detection : TYPE, optional
            DESCRIPTION. The default is True.
        blob_threshold : TYPE, optional
            DESCRIPTION. The default is 0.25.
        fit_peaks : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """

        with switch_xray_source(self.instr, self.xray_source):
            return self._autopick_points(
                raw_img_dict=raw_img_dict,
                tth_tol=tth_tol,
                eta_tol=eta_tol,
                npdiv=npdiv,
                do_smoothing=do_smoothing,
                smoothing_sigma=smoothing_sigma,
                use_blob_detection=use_blob_detection,
                blob_threshold=blob_threshold,
                fit_peaks=fit_peaks,
                min_peak_int=min_peak_int,
                fit_tth_tol=fit_tth_tol,
            )

    def _autopick_points(self, raw_img_dict, tth_tol=5., eta_tol=5.,
                         npdiv=2, do_smoothing=True, smoothing_sigma=2,
                         use_blob_detection=True, blob_threshold=0.25,
                         fit_peaks=True, min_peak_int=1., fit_tth_tol=0.1):
        labelStructure = ndimage.generate_binary_structure(2, 1)
        rmat_s = np.eye(3)  # !!! forcing to identity
        omega = 0.  # !!! same ^^^

        rmat_c = xfcapi.make_rmat_of_expmap(self.grain_params[:3])
        tvec_c = self.grain_params[3:6]
        # vinv_s = self.grain_params[6:12]  # !!!: patches don't take this yet

        # run simulation
        # ???: could we get this from overlays?
        laue_sim = self.instr.simulate_laue_pattern(
            self.plane_data,
            minEnergy=self.energy_cutoffs[0],
            maxEnergy=self.energy_cutoffs[1],
            rmat_s=None, grain_params=np.atleast_2d(self.grain_params),
        )

        # loop over detectors for results
        refl_dict = dict.fromkeys(self.instr.detectors)
        for det_key, det in self.instr.detectors.items():
            det_config = det.config_dict(
                chi=self.instr.chi,
                tvec=self.instr.tvec,
                beam_vector=self.instr.beam_vector
            )

            xy_det, hkls, angles, dspacing, energy = laue_sim[det_key]
            '''
            valid_xy = []
            valid_hkls = []
            valid_angs = []
            valid_energy = []
            '''
            # !!! not necessary to loop over grains since we can only handle 1
            # for gid in range(len(xy_det)):
            gid = 0
            # find valid reflections
            valid_refl = ~np.isnan(xy_det[gid][:, 0])
            valid_xy = xy_det[gid][valid_refl, :]
            valid_hkls = hkls[gid][:, valid_refl]
            valid_angs = angles[gid][valid_refl, :]
            valid_energy = energy[gid][valid_refl]

            # make patches
            refl_patches = xrdutil.make_reflection_patches(
                det_config,
                valid_angs, det.angularPixelSize(valid_xy),
                rmat_c=rmat_c, tvec_c=tvec_c,
                tth_tol=tth_tol, eta_tol=eta_tol,
                npdiv=npdiv, quiet=True)

            reflInfoList = []
            img = raw_img_dict[det_key]
            native_area = det.pixel_area
            num_patches = len(valid_angs)
            meas_xy = np.nan*np.ones((num_patches, 2))
            meas_angs = np.nan*np.ones((num_patches, 2))
            for iRefl, patch in enumerate(refl_patches):
                # check for overrun
                irow = patch[-1][0]
                jcol = patch[-1][1]
                if np.any([irow < 0, irow >= det.rows,
                           jcol < 0, jcol >= det.cols]):
                    continue
                if not np.all(
                        det.clip_to_panel(
                            np.vstack([patch[1][0].flatten(),
                                       patch[1][1].flatten()]).T
                            )[1]
                        ):
                    continue
                # use nearest interpolation
                spot_data = img[irow, jcol] * patch[3] * npdiv**2 / native_area
                spot_data -= np.amin(spot_data)
                patch_size = spot_data.shape

                sigmax = 0.25*np.min(spot_data.shape) * fwhm_to_sigma

                # optional gaussian smoothing
                if do_smoothing:
                    spot_data = filters.gaussian(spot_data, smoothing_sigma)

                if use_blob_detection:
                    spot_data_scl = 2.*spot_data/np.max(spot_data) - 1.

                    # Compute radii in the 3rd column.
                    blobs_log = blob_log(spot_data_scl,
                                         min_sigma=2,
                                         max_sigma=min(sigmax, 20),
                                         num_sigma=10,
                                         threshold=blob_threshold,
                                         overlap=0.1)
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
                        closest_peak_idx = np.argmin(
                            np.sum(com_diff**2, axis=1)
                        )
                        #
                    else:
                        closest_peak_idx = 0
                    #
                    coms = coms[closest_peak_idx]
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
                        params = [ampl,
                                  a_par, b_par, c_par,
                                  x0, y0, bgx, bgy, bkg]
                        #
                        result = leastsq(gaussian_2d, params, args=(spot_data))
                        #
                        fit_par = result[0]
                        #
                        coms = np.array([fit_par[5], fit_par[4]])
                        '''
                        print("%s, %d, (%.2f, %.2f), (%d, %d)"
                              % (det_key, iRefl, coms[0], coms[1],
                                 patch_size[0], patch_size[1]))
                        '''
                        row_cen = fit_tth_tol * patch_size[0]
                        col_cen = fit_tth_tol * patch_size[1]
                        if np.any(
                            [coms[0] < row_cen,
                             coms[0] >= patch_size[0] - row_cen,
                             coms[1] < col_cen,
                             coms[1] >= patch_size[1] - col_cen]
                        ):
                            continue
                        if (fit_par[0] < min_peak_int):
                            continue

                        # intensities
                        spot_intensity, int_err = nquad(
                            gaussian_2d_int,
                            [[0., 2.*y0], [0., 2.*x0]],
                            args=fit_par)
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
                    # !!! forcing ome = 0. -- could be inconsistent with rmat_s
                    cmv = np.atleast_2d(np.hstack([com_angs, omega]))
                    gvec_c = xfcapi.angles_to_gvec(
                        cmv,
                        chi=self.instr.chi,
                        rmat_c=rmat_c,
                        beam_vec=self.instr.beam_vector)
                    new_xy = xfcapi.gvec_to_xy(
                        gvec_c,
                        det.rmat, rmat_s, rmat_c,
                        det.tvec, self.instr.tvec, tvec_c,
                        beam_vec=self.instr.beam_vector)
                    meas_xy[iRefl, :] = new_xy
                    if det.distortion is not None:
                        meas_xy[iRefl, :] = det.distortion.apply_inverse(
                            meas_xy[iRefl, :]
                        )
                    meas_angs[iRefl, :] = com_angs
                else:
                    peakId = -999
                    #
                    spot_intensity = np.nan
                    max_intensity = np.nan
                reflInfoList.append([peakId, valid_hkls[:, iRefl],
                                     (spot_intensity, max_intensity),
                                     valid_energy[iRefl],
                                     valid_angs[iRefl, :],
                                     meas_angs[iRefl, :],
                                     meas_xy[iRefl, :]])
            reflInfo = np.array(
                [tuple(i) for i in reflInfoList],
                dtype=reflInfo_dtype)
            refl_dict[det_key] = reflInfo

        # Convert to our data_dict format
        data_dict = {
            'pick_xys': {},
            'hkls': {},
        }
        for det, det_picks in refl_dict.items():
            data_dict['pick_xys'].setdefault(det, [])
            data_dict['hkls'].setdefault(det, [])
            for entry in det_picks:
                hkl = entry[1].astype(int).tolist()
                cart = entry[6]
                data_dict['hkls'][det].append(hkl)
                data_dict['pick_xys'][det].append(cart)

        self.data_dict = data_dict
        return data_dict

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
            pick_hkls_dict[det_key] = np.atleast_2d(hkls[valid_idx, :]).T
            pick_xys_dict[det_key] = np.atleast_2d(xys[valid_idx, :])

        return pick_hkls_dict, pick_xys_dict

    def residual(self):
        with switch_xray_source(self.instr, self.xray_source):
            return self._residual()

    def _residual(self):
        # need this for laue obj
        pick_hkls_dict, pick_xys_dict = self._evaluate()

        # munge energy cutoffs
        energy_cutoffs = np.r_[0.5, 1.5] * np.asarray(self.energy_cutoffs)

        return sxcal_obj_func(
            [self.grain_params], self.instr, pick_xys_dict, pick_hkls_dict,
            self.bmatx, energy_cutoffs
        )

    def model(self):
        with switch_xray_source(self.instr, self.xray_source):
            return self._model()

    def _model(self):
        # need this for laue obj
        pick_hkls_dict, pick_xys_dict = self._evaluate()

        return sxcal_obj_func(
            [self.grain_params], self.instr, pick_xys_dict, pick_hkls_dict,
            self.bmatx, self.energy_cutoffs, sim_only=True
        )


# Objective function for Laue fitting
def sxcal_obj_func(grain_params, instr, meas_xy, hkls_idx,
                   bmat, energy_cutoffs, sim_only=False):
    """
    Objective function for Laue-based fitting.


    energy_cutoffs are [minEnergy, maxEnergy] where min/maxEnergy can be lists

    """
    # right now just stuck on the end and assumed
    # to all be the same length... FIX THIS
    calc_xy = {}
    calc_ang = {}
    for det_key, panel in instr.detectors.items():
        # Simulate Laue pattern:
        # returns xy_det, hkls_in, angles, dspacing, energy
        sim_results = panel.simulate_laue_pattern(
            [hkls_idx[det_key], bmat],
            minEnergy=energy_cutoffs[0], maxEnergy=energy_cutoffs[1],
            grain_params=grain_params,
            beam_vec=instr.beam_vector
        )

        calc_xy_tmp = sim_results[0][0]

        idx = ~np.isnan(calc_xy_tmp[:, 0])
        calc_xy[det_key] = calc_xy_tmp[idx, :]

        if sim_only:
            # Grab angles too. We dont use them otherwise.
            # FIXME: might need tth correction if there is a distortion.
            calc_angs_tmp = sim_results[2][0]
            calc_ang[det_key] = calc_angs_tmp[idx, :]

    # return values
    if sim_only:
        return {k: [calc_xy[k], calc_ang[k]] for k in calc_xy}

    meas_xy_all = np.vstack(list(meas_xy.values()))
    calc_xy_all = np.vstack(list(calc_xy.values()))

    diff_vecs_xy = calc_xy_all - meas_xy_all
    return diff_vecs_xy.flatten()


def gaussian_2d(p, data):
    shape = data.shape
    x, y = np.meshgrid(range(shape[1]), range(shape[0]))
    func = p[0]*np.exp(
        -(p[1]*(x-p[4])*(x-p[4])
          + p[2]*(x-p[4])*(y-p[5])
          + p[3]*(y-p[5])*(y-p[5]))
        ) + p[6]*(x-p[4]) + p[7]*(y-p[5]) + p[8]
    return func.flatten() - data.flatten()


def gaussian_2d_int(y, x, *p):
    func = p[0]*np.exp(
        -(p[1]*(x-p[4])*(x-p[4])
          + p[2]*(x-p[4])*(y-p[5])
          + p[3]*(y-p[5])*(y-p[5]))
        )
    return func.flatten()


reflInfo_dtype = [
    ('iRefl', int),
    ('hkl', (int, 3)),
    ('intensity', (float, 2)),
    ('energy', float),
    ('predAngles', (float, 2)),
    ('measAngles', (float, 2)),
    ('measXY', (float, 2)),
]
