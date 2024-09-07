from typing import Optional

import numpy as np

from hexrd import matrixutil as mutil
from hexrd.instrument import calc_angles_from_beam_vec, switch_xray_source
from hexrd.utils.hkl import hkl_to_str, str_to_hkl

from .calibrator import Calibrator
from .lmfit_param_handling import (
    create_material_params,
    update_material_from_params,
)

nfields_powder_data = 8


class PowderCalibrator(Calibrator):
    type = 'powder'

    def __init__(self, instr, material, img_dict, default_refinements=None,
                 tth_tol=None, eta_tol=0.25,
                 fwhm_estimate=None, min_pk_sep=1e-3, min_ampl=0.,
                 pktype='pvoigt', bgtype='linear',
                 tth_distortion=None, calibration_picks=None,
                 xray_source: Optional[str] = None):
        assert list(instr.detectors.keys()) == list(img_dict.keys()), \
            "instrument and image dict must have the same keys"

        self.instr = instr
        self.material = material
        self.img_dict = img_dict
        self.default_refinements = default_refinements
        self.xray_source = xray_source

        # for polar interpolation
        if tth_tol is not None:
            # This modifies the width on the plane data. Default to whatever
            # is on the plane data, so only set it if it is not None.
            self.tth_tol = tth_tol

        self.eta_tol = eta_tol
        self.fwhm_estimate = fwhm_estimate
        self.min_pk_sep = min_pk_sep
        self.min_ampl = min_ampl
        self.pktype = pktype
        self.bgtype = bgtype
        self.tth_distortion = tth_distortion

        self.plane_data.wavelength = instr.xrs_beam_energy(xray_source)

        self.param_names = []

        self.data_dict = None
        if calibration_picks is not None:
            # container for calibration data
            self.calibration_picks = calibration_picks

    def create_lmfit_params(self, current_params):
        # There shouldn't be more than one calibrator for a given material, so
        # just assume we have a unique name...
        params = create_material_params(self.material,
                                        self.default_refinements)

        # If multiple powder calibrators were used for the same material (such
        # as in 2XRS), then don't add params again.
        param_names = [x[0] for x in current_params]
        params = [x for x in params if x[0] not in param_names]

        self.param_names = [x[0] for x in params]
        return params

    def update_from_lmfit_params(self, params_dict):
        if self.param_names:
            update_material_from_params(params_dict, self.material)

    @property
    def plane_data(self):
        return self.material.planeData

    @property
    def tth_tol(self):
        tth_tol = self.plane_data.tThWidth
        return np.degrees(tth_tol) if tth_tol is not None else tth_tol

    @tth_tol.setter
    def tth_tol(self, x):
        assert np.isscalar(x), "tth_tol must be a scalar value"
        self.plane_data.tThWidth = np.radians(self.tth_tol)

    @property
    def spectrum_kwargs(self):
        return dict(pktype=self.pktype,
                    bgtype=self.bgtype,
                    fwhm_init=self.fwhm_estimate,
                    min_ampl=self.min_ampl,
                    min_pk_sep=self.min_pk_sep)

    @property
    def calibration_picks(self):
        # Convert this from our internal data dict format
        picks = {}
        for det_key, data in self.data_dict.items():
            picks[det_key] = {}
            for ringset in data:
                for row in ringset:
                    # Rows 3, 4, and 5 are the hkl
                    hkl_str = hkl_to_str(row[3:6].astype(int))
                    picks[det_key].setdefault(hkl_str, [])
                    # Rows 0 and 1 are the xy coordinates
                    picks[det_key][hkl_str].append(row[:2].tolist())

        return picks

    @calibration_picks.setter
    def calibration_picks(self, v):
        # Convert this to our internal data dict format
        data_dict = {}
        for det_key, hkl_picks in v.items():
            data_dict[det_key] = []
            for hkl_str, picks in hkl_picks.items():
                if len(picks) == 0:
                    # Just skip over it
                    continue

                data = np.zeros((len(picks), 8), dtype=np.float64)
                # Rows 0 and 1 are the xy coordinates
                data[:, :2] = np.asarray(picks)
                # Rows 3, 4, and 5 are the hkl
                data[:, 3:6] = str_to_hkl(hkl_str)
                data_dict[det_key].append(data)

        self.data_dict = data_dict

    def autopick_points(self, fit_tth_tol=5., int_cutoff=1e-4):
        """
        return the RHS for the instrument DOF and image dict

        The format is a dict over detectors, each containing

        [index over ring sets]
            [index over azimuthal patch]
                [xy_meas, tth_meas, hkl, dsp_ref, eta_ref]

        FIXME: can not yet handle tth ranges with multiple peaks!
        """
        # If needed, change the x-ray source before proceeding.
        # This does nothing for single x-ray sources.
        with switch_xray_source(self.instr, self.xray_source):
            return self._autopick_points(fit_tth_tol, int_cutoff)

    def _autopick_points(self, fit_tth_tol=5., int_cutoff=1e-4):
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

        self.data_dict = rhs
        return rhs

    def _evaluate(self, output='residual'):
        """
        Evaluate the powder diffraction model.

        Parameters
        ----------
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
        # In case the beam energy was modified, ensure it is updated
        # on the plane data as well.
        self.plane_data.wavelength = self.instr.beam_energy

        # need this for dsp
        bmat = self.plane_data.latVecOps['B']
        wlen = self.instr.beam_wavelength

        # build residual
        retval = np.array([], dtype=float)
        for det_key, panel in self.instr.detectors.items():
            if len(self.data_dict[det_key]) == 0:
                continue
            else:
                # recast as array
                pdata = np.vstack(self.data_dict[det_key])

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

    def residual(self):
        # If needed, change the x-ray source before proceeding.
        # This does nothing for single x-ray sources.
        with switch_xray_source(self.instr, self.xray_source):
            return self._evaluate(output='residual')

    def model(self):
        # If needed, change the x-ray source before proceeding.
        # This does nothing for single x-ray sources.
        with switch_xray_source(self.instr, self.xray_source):
            return self._evaluate(output='model')
