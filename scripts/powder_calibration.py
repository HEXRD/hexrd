#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:45:01 2019

@author: joel
"""

import numpy as np

from hexrd.fitting import fitpeak
from hexrd.matrixutil import findDuplicateVectors


# %%
class InstrumentCalibrator(object):
    def __init__(self, instr, *args):
        self._instr = instr
        assert len(args) > 0, \
            "must have at least one calibrator"
        self._calibrators = *args

    @property
    def instr(self):
        return self._instr


# %%
class PowderCalibrator(object):
    def __init__(self, instr, plane_data, img_dict,
                 tth_tol=None, eta_tol=0.25,
                 pktype='pvoigt'):
        assert list(instr.detectors.keys()) == list(img_dict.keys()), \
            "instrument and image dict must have the same keys"
        self._instr = instr
        self._plane_data = plane_data
        self._img_dict = img_dict

        # for polar interpolation
        self._tth_tol = tth_tol or np.degrees(plane_data.tThWidth)
        self._eta_tol = eta_tol

        # for peak fitting
        # ??? fitting only, or do alternative peak detection?
        self._pktype = pktype

    @property
    def instr(self):
        return self._instr

    @property
    def plane_data(self):
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
    def pktype(self):
        return self._pktype

    @pktype.setter
    def pktype(self, x):
        """
        currently only 'pvoigt' or 'gaussian'
        """
        assert isinstance(x, str), "tth_tol must be a scalar value"
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

    def _extract_powder_lines(self):
        """
        return the RHS for the instrument DOF and image dict

        The format is a dict over detectors, each containing

        [index over ring sets]
            [index over azimuthal patch]
                [xy_meas, tth_meas, tth_ref, eta_ref]

        FIXME: can not yet handle tth ranges with multiple peaks!
        """
        # ideal tth
        tth_ideal = self.plane_data.getTTh()
        tth0 = []
        for idx in self.plane_data.getMergedRanges()[0]:
            if len(idx) > 1:
                eqv, uidx = findDuplicateVectors(np.atleast_2d(tth_ideal[idx]))
                if len(uidx) > 1:
                    raise NotImplementedError("can not handle multipeak yet")
                else:
                    # if here, only degenerate ring case
                    uidx = idx[0]
            else:
                uidx = idx[0]
            tth0.append(tth_ideal[uidx])

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
                    """
                    DARREN: FIT [tth_centers, intensities[0]] HERE

                    RETURN TTH0
                    rhs.append([tth0, eta_ref])
                    """
                    p0 = fitpeak.estimate_pk_parms_1d(
                            tth_centers, int1d, self.pktype
                         )

                    p = fitpeak.fit_pk_parms_1d(
                            p0, tth_centers, int1d, self.pktype
                        )
                    # !!! this is where we can kick out bunk fits
                    tth_meas = p[1]
                    center_err = abs(tth_meas - tth0[i_ring])
                    if p[0] < 0.1 or center_err > np.radians(self.tth_tol):
                        continue
                    xy_meas = panel.angles_to_cart([[tth_meas, eta_ref], ])
                    tmp.append(
                        np.hstack(
                            [xy_meas.squeeze(),
                             tth_meas,
                             tth0[i_ring],
                             eta_ref]
                        )
                    )
                    pass
                rhs[det_key].append(np.vstack(tmp))
                pass
            rhs[det_key] = np.vstack(rhs[det_key])
            pass
        return rhs

    def residual(self, reduced_params, data_dict):
        """
        """

        # first update instrument from input parameters
        full_params = self.instr.calibration_parameters
        full_params[self.instr.calibration_flags] = reduced_params
        self.instr.update_from_parameter_list(full_params)

        # build residual
        resd = []
        for det_key, panel in self.instr.detectors.items():
            pdata = np.vstack(data_dict[det_key])
            if len(pdata) > 0:
                calc_xy = panel.angles_to_cart(pdata[:, -2:])

                resd.append(
                    (pdata[:, :2].flatten() - calc_xy.flatten())
                )
            else:
                continue
        return np.hstack(resd)


