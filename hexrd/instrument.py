# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# =============================================================================
"""
Created on Fri Dec  9 13:05:27 2016

@author: bernier2
"""
import os
from concurrent.futures import ThreadPoolExecutor
import functools

import yaml

import h5py

import numpy as np

from io import IOBase

from scipy import ndimage
from scipy.linalg.matfuncs import logm

from hexrd.gridutil import cellIndices, make_tolerance_grid
from hexrd import matrixutil as mutil
from hexrd.transforms.xfcapi import \
    anglesToGVec, \
    angularDifference, \
    detectorXYToGvec, \
    gvecToDetectorXY, \
    makeOscillRotMat, \
    makeRotMatOfExpMap, \
    mapAngle, \
    oscillAnglesOfHKLs, \
    rowNorm, \
    unitRowVector
from hexrd import xrdutil
from hexrd.crystallography import PlaneData
from hexrd import constants as ct
from hexrd.rotations import angleAxisOfRotMat, RotMatEuler
from hexrd.config.dumper import NumPyIncludeDumper

# FIXME: distortion kludge
from hexrd.distortion import GE_41RT  # BAD, VERY BAD!!!

from skimage.draw import polygon

try:
    from fast_histogram import histogram1d
    fast_histogram = True
except(ImportError):
    from numpy import histogram as histogram1d
    fast_histogram = False

# =============================================================================
# PARAMETERS
# =============================================================================

instrument_name_DFLT = 'instrument'

beam_energy_DFLT = 65.351
beam_vec_DFLT = ct.beam_vec

eta_vec_DFLT = ct.eta_vec

panel_id_DFLT = 'generic'
nrows_DFLT = 2048
ncols_DFLT = 2048
pixel_size_DFLT = (0.2, 0.2)

tilt_params_DFLT = np.zeros(3)
t_vec_d_DFLT = np.r_[0., 0., -1000.]

chi_DFLT = 0.
t_vec_s_DFLT = np.zeros(3)

"""
Calibration parameter flags

 for instrument level, len is 7

 [beam energy,
  beam azimuth,
  beam elevation,
  chi,
  tvec[0],
  tvec[1],
  tvec[2],
  ]
"""
instr_calibration_flags_DFLT = np.zeros(7, dtype=bool)

"""
 for each panel, order is:

 [tilt[0],
  tilt[1],
  tilt[2],
  tvec[0],
  tvec[1],
  tvec[2],
  <dparams>,
  ]

 len is 6 + len(dparams) for each panel
 by default, dparams are not set for refinement
"""
panel_calibration_flags_DFLT = np.array(
    [1, 1, 1, 1, 1, 1],
    dtype=bool
)

buffer_key = 'buffer'
distortion_key = 'distortion'

# =============================================================================
# UTILITY METHODS
# =============================================================================


def _fix_indices(idx, lo, hi):
    nidx = np.array(idx)
    off_lo = nidx < lo
    off_hi = nidx > hi
    nidx[off_lo] = lo
    nidx[off_hi] = hi
    return nidx


def calc_beam_vec(azim, pola):
    """
    Calculate unit beam propagation vector from
    spherical coordinate spec in DEGREES.

    ...MAY CHANGE; THIS IS ALSO LOCATED IN XRDUTIL!
    """
    tht = np.radians(azim)
    phi = np.radians(pola)
    bv = np.r_[
        np.sin(phi)*np.cos(tht),
        np.cos(phi),
        np.sin(phi)*np.sin(tht)]
    return -bv


def calc_angles_from_beam_vec(bvec):
    """
    Return the azimuth and polar angle from a beam
    vector
    """
    bvec = np.atleast_1d(bvec).flatten()
    nvec = unitRowVector(-bvec)
    azim = float(
        np.degrees(np.arctan2(nvec[2], nvec[0]))
    )
    pola = float(np.degrees(np.arccos(nvec[1])))
    return azim, pola


def migrate_instrument_config(instrument_config):
    """utility function to generate old instrument config dictionary"""
    cfg_list = []
    for detector_id in instrument_config['detectors']:
        cfg_list.append(
            dict(
                detector=instrument_config['detectors'][detector_id],
                oscillation_stage=instrument_config['oscillation_stage'],
            )
        )
    return cfg_list


def angle_in_range(angle, ranges, ccw=True, units='degrees'):
    """
    Return the index of the first wedge the angle is found in

    WARNING: always clockwise; assumes wedges are not overlapping
    """
    tau = 360.
    if units.lower() == 'radians':
        tau = 2*np.pi
    w = np.nan
    for i, wedge in enumerate(ranges):
        amin = wedge[0]
        amax = wedge[1]
        check = amin + np.mod(angle - amin, tau)
        if check < amax:
            w = i
            break
    return w


# ???: move to gridutil?
def centers_of_edge_vec(edges):
    assert np.r_[edges].ndim == 1, "edges must be 1-d"
    return np.average(np.vstack([edges[:-1], edges[1:]]), axis=0)


def max_tth(instr):
    """
    Return the maximum Bragg angle (in radians) subtended by the instrument.

    Parameters
    ----------
    instr : hexrd.instrument.HEDMInstrument instance
        the instrument class to evalutate.

    Returns
    -------
    tth_max : float
        The maximum observable Bragg angle by the instrument in radians.
    """
    tth_max = 0.
    for det in instr.detectors.values():
        ptth, peta = det.pixel_angles()
        tth_max = max(np.max(ptth), tth_max)
    return tth_max


def _gaussian_dist(x, cen, fwhm):
    sigm = fwhm/(2*np.sqrt(2*np.log(2)))
    return np.exp(-0.5*(x - cen)**2/sigm**2)


def _sigma_to_fwhm(sigm):
    return sigm*ct.sigma_to_fwhm


def _fwhm_to_sigma(fwhm):
    return fwhm/ct.sigma_to_fwhm


# =============================================================================
# CLASSES
# =============================================================================


class HEDMInstrument(object):
    """
    Abstraction of XRD instrument.

    * Distortion needs to be moved to a class with registry; tuple unworkable
    * where should reference eta be defined? currently set to default config
    """

    def __init__(self, instrument_config=None,
                 image_series=None, eta_vector=None,
                 instrument_name=None, tilt_calibration_mapping=None):
        self._id = instrument_name_DFLT

        if eta_vector is None:
            self._eta_vector = eta_vec_DFLT
        else:
            self._eta_vector = eta_vector

        if instrument_config is None:
            if instrument_name is not None:
                self._id = instrument_name
            self._num_panels = 1
            self._beam_energy = beam_energy_DFLT
            self._beam_vector = beam_vec_DFLT

            self._detectors = dict(
                panel_id_DFLT=PlanarDetector(
                    rows=nrows_DFLT, cols=ncols_DFLT,
                    pixel_size=pixel_size_DFLT,
                    tvec=t_vec_d_DFLT,
                    tilt=tilt_params_DFLT,
                    bvec=self._beam_vector,
                    evec=self._eta_vector,
                    distortion=None),
                )

            self._tvec = t_vec_s_DFLT
            self._chi = chi_DFLT
        else:
            if instrument_name is None:
                if 'id' in instrument_config:
                    self._id = instrument_config['id']
            else:
                self._id = instrument_name
            self._num_panels = len(instrument_config['detectors'])
            self._beam_energy = instrument_config['beam']['energy']  # keV
            self._beam_vector = calc_beam_vec(
                instrument_config['beam']['vector']['azimuth'],
                instrument_config['beam']['vector']['polar_angle'],
                )

            # now build detector dict
            detectors_config = instrument_config['detectors']
            det_dict = dict.fromkeys(detectors_config)
            for det_id, det_info in detectors_config.items():
                pixel_info = det_info['pixels']
                affine_info = det_info['transform']
                try:
                    saturation_level = det_info['saturation_level']
                except(KeyError):
                    saturation_level = 2**16
                shape = (pixel_info['rows'], pixel_info['columns'])

                panel_buffer = None
                if buffer_key in det_info:
                    det_buffer = det_info[buffer_key]
                    if det_buffer is not None:
                        if isinstance(det_buffer, np.ndarray):
                            assert det_buffer.shape == shape, \
                                "buffer shape must match detector"
                        elif isinstance(det_buffer, list):
                            panel_buffer = np.asarray(det_buffer)
                        elif np.isscalar(det_buffer):
                            panel_buffer = det_buffer*np.ones(2)
                        else:
                            raise RuntimeError(
                                "panel buffer spec invalid for %s" % det_id
                            )

                # FIXME: must promote this to a class w/ registry
                distortion = None
                if distortion_key in det_info:
                    distortion = det_info[distortion_key]
                    if det_info[distortion_key] is not None:
                        # !!! hard-coded GE distortion
                        distortion = [GE_41RT, distortion['parameters']]

                det_dict[det_id] = PlanarDetector(
                        name=det_id,
                        rows=pixel_info['rows'],
                        cols=pixel_info['columns'],
                        pixel_size=pixel_info['size'],
                        panel_buffer=panel_buffer,
                        saturation_level=saturation_level,
                        tvec=affine_info['translation'],
                        tilt=affine_info['tilt'],
                        bvec=self._beam_vector,
                        evec=self._eta_vector,
                        distortion=distortion)

            self._detectors = det_dict

            self._tvec = np.r_[
                instrument_config['oscillation_stage']['translation']
            ]
            self._chi = instrument_config['oscillation_stage']['chi']

        #
        # set up calibration parameter list and refinement flags
        #
        # first, grab the mapping function for tilt parameters if specified
        if tilt_calibration_mapping is not None:
            if not isinstance(tilt_calibration_mapping, RotMatEuler):
                raise RuntimeError(
                    "tilt mapping must be a 'RotMatEuler' instance"
                )
        self._tilt_calibration_mapping = tilt_calibration_mapping

        # grab angles from beam vec
        # !!! these are in DEGREES!
        azim, pola = calc_angles_from_beam_vec(self._beam_vector)

        # stack instrument level parameters
        # units: keV, degrees, mm
        self._calibration_parameters = [
            self._beam_energy,
            azim,
            pola,
            np.degrees(self._chi),
            *self._tvec,
        ]
        self._calibration_flags = instr_calibration_flags_DFLT

        # collect info from panels and append
        det_params = []
        det_flags = []
        for detector in self._detectors.values():
            this_det_params = detector.calibration_parameters
            if self._tilt_calibration_mapping is not None:
                rmat = makeRotMatOfExpMap(detector.tilt)
                self._tilt_calibration_mapping.rmat = rmat
                tilt = np.degrees(self._tilt_calibration_mapping.angles)
                this_det_params[:3] = tilt
            det_params.append(this_det_params)
            det_flags.append(detector.calibration_flags)
        det_params = np.hstack(det_params)
        det_flags = np.hstack(det_flags)

        # !!! hstack here assumes that calib params will be float and
        # !!! flags will all be bool
        self._calibration_parameters = np.hstack(
            [self._calibration_parameters,
             det_params]
        ).flatten()
        self._calibration_flags = np.hstack(
            [self._calibration_flags,
             det_flags]
        )
        return

    # properties for physical size of rectangular detector
    @property
    def id(self):
        return self._id

    @property
    def num_panels(self):
        return self._num_panels

    @property
    def detectors(self):
        return self._detectors

    @property
    def detector_parameters(self):
        pdict = {}
        for key, panel in self.detectors.items():
            pdict[key] = panel.config_dict(
                self.chi, self.tvec,
                beam_energy=self.beam_energy,
                beam_vector=self.beam_vector
            )
        return pdict

    @property
    def tvec(self):
        return self._tvec

    @tvec.setter
    def tvec(self, x):
        x = np.array(x).flatten()
        assert len(x) == 3, 'input must have length = 3'
        self._tvec = x

    @property
    def chi(self):
        return self._chi

    @chi.setter
    def chi(self, x):
        self._chi = float(x)

    @property
    def beam_energy(self):
        return self._beam_energy

    @beam_energy.setter
    def beam_energy(self, x):
        self._beam_energy = float(x)

    @property
    def beam_wavelength(self):
        return ct.keVToAngstrom(self.beam_energy)

    @property
    def beam_vector(self):
        return self._beam_vector

    @beam_vector.setter
    def beam_vector(self, x):
        x = np.array(x).flatten()
        if len(x) == 3:
            assert sum(x*x) > 1-ct.sqrt_epsf, \
                'input must have length = 3 and have unit magnitude'
            self._beam_vector = x
        elif len(x) == 2:
            self._beam_vector = calc_beam_vec(*x)
        else:
            raise RuntimeError("input must be a unit vector or angle pair")
        # ...maybe change dictionary item behavior for 3.x compatibility?
        for detector_id in self.detectors:
            panel = self.detectors[detector_id]
            panel.bvec = self._beam_vector

    @property
    def eta_vector(self):
        return self._eta_vector

    @eta_vector.setter
    def eta_vector(self, x):
        x = np.array(x).flatten()
        assert len(x) == 3 and sum(x*x) > 1-ct.sqrt_epsf, \
            'input must have length = 3 and have unit magnitude'
        self._eta_vector = x
        # ...maybe change dictionary item behavior for 3.x compatibility?
        for detector_id in self.detectors:
            panel = self.detectors[detector_id]
            panel.evec = self._eta_vector

    @property
    def tilt_calibration_mapping(self):
        return self._tilt_calibration_mapping

    @tilt_calibration_mapping.setter
    def tilt_calibration_mapping(self, x):
        if not isinstance(x, RotMatEuler) and x is not None:
            raise RuntimeError(
                    "tilt mapping must be None or a 'RotMatEuler' instance"
                )
        self._tilt_calibration_mapping = x

    @property
    def calibration_parameters(self):
        """
        Yields concatenated list of instrument parameters.

        Returns
        -------
        array_like
            concatenated list of instrument parameters.

        """
        # grab angles from beam vec
        # !!! these are in DEGREES!
        azim, pola = calc_angles_from_beam_vec(self.beam_vector)

        # stack instrument level parameters
        # units: keV, degrees, mm
        calibration_parameters = [
            self.beam_energy,
            azim,
            pola,
            np.degrees(self.chi),
            *self.tvec,
        ]

        # collect info from panels and append
        det_params = []
        det_flags = []
        for detector in self.detectors.values():
            this_det_params = detector.calibration_parameters
            if self.tilt_calibration_mapping is not None:
                rmat = makeRotMatOfExpMap(detector.tilt)
                self.tilt_calibration_mapping.rmat = rmat
                tilt = np.degrees(self.tilt_calibration_mapping.angles)
                this_det_params[:3] = tilt
            det_params.append(this_det_params)
            det_flags.append(detector.calibration_flags)
        det_params = np.hstack(det_params)
        det_flags = np.hstack(det_flags)

        # !!! hstack here assumes that calib params will be float and
        # !!! flags will all be bool
        calibration_parameters = np.hstack(
            [calibration_parameters,
             det_params]
        ).flatten()
        self._calibration_parameters = calibration_parameters
        return self._calibration_parameters

    @property
    def calibration_flags(self):
        return self._calibration_flags

    @calibration_flags.setter
    def calibration_flags(self, x):
        x = np.array(x, dtype=bool).flatten()
        if len(x) != len(self._calibration_flags):
            raise RuntimeError(
                "length of parameter list must be %d; you gave %d"
                % (len(self._calibration_flags), len(x))
            )
        ii = 7
        for panel in self.detectors.values():
            npp = 6
            if panel.distortion is not None:
                # FIXME: pending distortion update
                npp += len(panel.distortion[1])
            panel.calibration_flags = x[ii:ii + npp]
        self._calibration_flags = x

    # =========================================================================
    # METHODS
    # =========================================================================

    def write_config(self, filename=None, calibration_dict={}):
        """ WRITE OUT YAML FILE """
        # initialize output dictionary

        par_dict = {}

        par_dict['id'] = self.id

        azim, pola = calc_angles_from_beam_vec(self.beam_vector)
        beam = dict(
            energy=self.beam_energy,
            vector=dict(
                azimuth=azim,
                polar_angle=pola,
            )
        )
        par_dict['beam'] = beam

        if calibration_dict:
            par_dict['calibration_crystal'] = calibration_dict

        ostage = dict(
            chi=self.chi,
            translation=self.tvec.tolist()
        )
        par_dict['oscillation_stage'] = ostage

        det_dict = dict.fromkeys(self.detectors)
        for det_name, panel in self.detectors.items():
            pdict = panel.config_dict(self.chi, self.tvec)  # don't need beam
            det_dict[det_name] = pdict['detector']
        par_dict['detectors'] = det_dict
        if filename is not None:
            with open(filename, 'w') as f:
                yaml.dump(par_dict, stream=f, Dumper=NumPyIncludeDumper)
        return par_dict

    def update_from_parameter_list(self, p):
        """
        Update the instrument class from a parameter list.

        Utility function to update instrument parameters from a 1-d master
        parameter list (e.g. as used in calibration)

        !!! Note that angles are reported in DEGREES!
        """
        self.beam_energy = p[0]
        self.beam_vector = calc_beam_vec(p[1], p[2])
        self.chi = np.radians(p[3])
        self.tvec = np.r_[p[4:7]]

        ii = 7
        for det_name, detector in self.detectors.items():
            this_det_params = detector.calibration_parameters
            npd = len(this_det_params)  # total number of params
            dpnp = npd - 6  # number of distortion params

            # first do tilt
            tilt = np.r_[p[ii:ii + 3]]
            if self.tilt_calibration_mapping is not None:
                self.tilt_calibration_mapping.angles = np.radians(tilt)
                rmat = self.tilt_calibration_mapping.rmat
                phi, n = angleAxisOfRotMat(rmat)
                tilt = phi*n.flatten()
            detector.tilt = tilt

            # then do translation
            ii += 3
            detector.tvec = np.r_[p[ii:ii + 3]]

            # then do distortion (if necessart)
            # FIXME will need to update this with distortion fix
            ii += 3
            if dpnp > 0:
                if detector.distortion is None:
                    raise RuntimeError(
                        "distortion discrepancy for '%s'!"
                        % det_name
                    )
                else:
                    if len(detector.distortion[1]) != dpnp:
                        raise RuntimeError(
                            "length of dist params is incorrect"
                        )
                detector.distortion[1] = p[ii:ii + dpnp]
                ii += dpnp
        return

    def extract_polar_maps(self, plane_data, imgser_dict,
                           active_hkls=None, threshold=None,
                           tth_tol=None, eta_tol=0.25):
        """
        Extract eta-omega maps from an imageseries.

        Quick and dirty way to histogram angular patch data for make
        pole figures suitable for fiber generation

        TODO: streamline projection code
        TODO: normalization
        """
        if tth_tol is not None:
            plane_data.tThWidth = np.radians(tth_tol)
        else:
            tth_tol = np.degrees(plane_data.tThWidth)

        tth_ranges = plane_data.getTThRanges()
        if active_hkls is not None:
            assert hasattr(active_hkls, '__len__'), \
                "active_hkls must be an iterable with __len__"
            tth_ranges = tth_ranges[active_hkls]

        # # need this for making eta ranges
        # eta_tol_vec = 0.5*np.radians([-eta_tol, eta_tol])

        # make rings clipped to panel
        # !!! eta_idx has the same length as plane_data.exclusions
        #       each entry are the integer indices into the bins
        # !!! eta_edges is the list of eta bin EDGES
        # We can use the same eta_edge for all detectors, so calculate it once
        pow_angs, pow_xys, eta_idx, eta_edges = list(
                self.detectors.values()
            )[0].make_powder_rings(plane_data,
                                   merge_hkls=False, delta_eta=eta_tol,
                                   full_output=True)
        delta_eta = eta_edges[1] - eta_edges[0]
        ncols_eta = len(eta_edges) - 1

        ring_maps_panel = dict.fromkeys(self.detectors)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as tp:
            for i_d, det_key in enumerate(self.detectors):
                print("working on detector '%s'..." % det_key)

                # grab panel
                panel = self.detectors[det_key]
                # native_area = panel.pixel_area  # pixel ref area

                # pixel angular coords for the detector panel
                ptth, peta = panel.pixel_angles()

                # grab omegas from imageseries and squawk if missing
                try:
                    omegas = imgser_dict[det_key].metadata['omega']
                except(KeyError):
                    msg = "imageseries for '%s' has no omega info" % det_key
                    raise RuntimeError(msg)

                # initialize maps and assing by row (omega/frame)
                nrows_ome = len(omegas)

                ring_maps = []
                for i_r, tthr in enumerate(tth_ranges):
                    print("working on ring %d..." % i_r)

                    # init map with NaNs
                    this_map = np.nan*np.ones((nrows_ome, ncols_eta))

                    # mark pixels in the spec'd tth range
                    pixels_in_tthr = np.logical_and(
                        ptth >= tthr[0], ptth <= tthr[1]
                    )

                    # catch case where ring isn't on detector
                    if not np.any(pixels_in_tthr):
                        ring_maps.append(this_map)
                        continue

                    # ???: faster to index with bool or use np.where,
                    # or recode in numba?
                    rtth_idx = np.where(pixels_in_tthr)

                    # grab relevant eta coords using histogram
                    # !!!: This allows use to calculate arc length and
                    #      detect a branch cut.  The histogram idx var
                    #      is the left-hand edges...
                    retas = peta[rtth_idx]
                    if fast_histogram:
                        reta_hist = histogram1d(
                            retas,
                            len(eta_edges) - 1,
                            (eta_edges[0], eta_edges[-1])
                        )
                    else:
                        reta_hist, _ = histogram1d(retas, bins=eta_edges)
                    reta_idx = np.where(reta_hist)[0]
                    reta_bin_idx = np.hstack(
                        [reta_idx,
                         reta_idx[-1] + 1]
                    )

                    # ring arc lenght on panel
                    arc_length = angularDifference(
                        eta_edges[reta_bin_idx[0]],
                        eta_edges[reta_bin_idx[-1]]
                    )

                    # Munge eta bins
                    # !!! need to work with the subset to preserve
                    #     NaN values at panel extents!
                    #
                    # !!! MUST RE-MAP IF BRANCH CUT IS IN RANGE
                    #
                    # The logic below assumes that eta_edges span 2*pi to
                    # single precision
                    eta_bins = eta_edges[reta_bin_idx]
                    if arc_length < 1e-4:
                        # have branch cut in here
                        ring_gap = np.where(
                            reta_idx
                            - np.arange(len(reta_idx))
                        )[0]
                        if len(ring_gap) > 0:
                            # have incomplete ring
                            eta_stop_idx = ring_gap[0]
                            eta_stop = eta_edges[eta_stop_idx]
                            new_period = np.cumsum([eta_stop, 2*np.pi])
                            # remap
                            retas = mapAngle(retas, new_period)
                            tmp_bins = mapAngle(
                                eta_edges[reta_idx], new_period
                            )
                            tmp_idx = np.argsort(tmp_bins)
                            reta_idx = reta_idx[np.argsort(tmp_bins)]
                            eta_bins = np.hstack(
                                [tmp_bins[tmp_idx],
                                 tmp_bins[tmp_idx][-1] + delta_eta]
                            )
                            pass
                        pass
                    # histogram intensities over eta ranges
                    for i_row, image in enumerate(imgser_dict[det_key]):
                        if fast_histogram:
                            def _on_done(map, row, reta, future):
                                map[row, reta] = future.result()

                            f = tp.submit(
                                histogram1d,
                                retas,
                                len(eta_bins) - 1,
                                (eta_bins[0], eta_bins[-1]),
                                weights=image[rtth_idx]
                            )
                            f.add_done_callback(
                                functools.partial(
                                    _on_done, this_map, i_row, reta_idx
                                )
                            )
                        else:
                            def _on_done(map, row, reta, future):
                                map[row, reta], _ = future.result()

                            f = tp.submit(
                                histogram1d,
                                retas,
                                bins=eta_bins,
                                weights=image[rtth_idx]
                            )
                            f.add_done_callback(
                                functools.partial(
                                    _on_done, this_map, i_row, reta_idx
                                )
                            )
                        pass    # end loop on rows
                    ring_maps.append(this_map)
                    pass    # end loop on rings
                ring_maps_panel[det_key] = ring_maps

        return ring_maps_panel, eta_edges

    def extract_line_positions(self, plane_data, imgser_dict,
                               tth_tol=None, eta_tol=1., npdiv=2,
                               collapse_eta=True, collapse_tth=False,
                               do_interpolation=True):
        """
        Extract the line positions from powder diffraction images.

        Generates and processes 'caked' sector data over an instrument.

        Parameters
        ----------
        plane_data : TYPE
            DESCRIPTION.
        imgser_dict : TYPE
            DESCRIPTION.
        tth_tol : TYPE, optional
            DESCRIPTION. The default is None.
        eta_tol : TYPE, optional
            DESCRIPTION. The default is 1..
        npdiv : TYPE, optional
            DESCRIPTION. The default is 2.
        collapse_eta : TYPE, optional
            DESCRIPTION. The default is True.
        collapse_tth : TYPE, optional
            DESCRIPTION. The default is False.
        do_interpolation : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        panel_data : TYPE
            DESCRIPTION.
        """
        if not hasattr(plane_data, '__len__'):
            plane_data = plane_data.makeNew()  # make local copy to munge
            if tth_tol is not None:
                plane_data.tThWidth = np.radians(tth_tol)
            tth_ranges = np.degrees(plane_data.getMergedRanges()[1])
            tth_tols = np.vstack([i[1] - i[0] for i in tth_ranges])
        else:
            tth_tols = np.ones(len(plane_data))*tth_tol

        # =====================================================================
        # LOOP OVER DETECTORS
        # =====================================================================
        panel_data = dict.fromkeys(self.detectors)
        for i_det, detector_id in enumerate(self.detectors):
            print("working on detector '%s'..." % detector_id)
            # pbar.update(i_det + 1)
            # grab panel
            panel = self.detectors[detector_id]
            instr_cfg = panel.config_dict(
                chi=self.chi, tvec=self.tvec,
                beam_energy=self.beam_energy,
                beam_vector=self.beam_vector
            )
            native_area = panel.pixel_area  # pixel ref area
            images = imgser_dict[detector_id]
            if images.ndim == 2:
                n_images = 1
                images = np.tile(images, (1, 1, 1))
            elif images.ndim == 3:
                n_images = len(images)
            else:
                raise RuntimeError("images must be 2- or 3-d")

            # make rings
            pow_angs, pow_xys = panel.make_powder_rings(
                plane_data, merge_hkls=True,
                delta_tth=tth_tol, delta_eta=eta_tol)

            # =================================================================
            # LOOP OVER RING SETS
            # =================================================================
            ring_data = []
            for i_ring, these_data in enumerate(zip(pow_angs, pow_xys)):
                print("interpolating 2theta bin %d..." % i_ring)

                # points are already checked to fall on detector
                angs = these_data[0]
                xys = these_data[1]

                # make the tth,eta patches for interpolation
                patches = xrdutil.make_reflection_patches(
                    instr_cfg, angs, panel.angularPixelSize(xys),
                    tth_tol=tth_tols[i_ring], eta_tol=eta_tol,
                    npdiv=npdiv, quiet=True)

                # loop over patches
                # FIXME: fix initialization
                if collapse_tth:
                    patch_data = np.zeros((len(angs), n_images))
                else:
                    patch_data = []
                for i_p, patch in enumerate(patches):
                    # strip relevant objects out of current patch
                    vtx_angs, vtx_xys, conn, areas, xys_eval, ijs = patch

                    # need to reshape eval pts for interpolation
                    xy_eval = np.vstack([
                        xys_eval[0].flatten(),
                        xys_eval[1].flatten()]).T

                    _, on_panel = panel.clip_to_panel(xy_eval)

                    if np.any(~on_panel):
                        continue

                    if collapse_tth:
                        ang_data = (vtx_angs[0][0, [0, -1]],
                                    vtx_angs[1][[0, -1], 0])
                    else:
                        ang_data = (vtx_angs[0][0, :],
                                    angs[i_p][-1])

                    prows, pcols = areas.shape
                    area_fac = areas/float(native_area)

                    # interpolate
                    if not collapse_tth:
                        ims_data = []
                    for j_p in np.arange(len(images)):
                        # catch interpolation type
                        image = images[j_p]
                        if do_interpolation:
                            tmp = panel.interpolate_bilinear(
                                    xy_eval,
                                    image,
                                ).reshape(prows, pcols)*area_fac
                        else:
                            tmp = image[ijs[0], ijs[1]]*area_fac

                        # catch collapsing options
                        if collapse_tth:
                            patch_data[i_p, j_p] = np.sum(tmp)
                            # ims_data.append(np.sum(tmp))
                        else:
                            if collapse_eta:
                                ims_data.append(np.sum(tmp, axis=0))
                            else:
                                ims_data.append(tmp)
                        pass  # close image loop
                    if not collapse_tth:
                        patch_data.append((ang_data, ims_data))
                    pass  # close patch loop
                ring_data.append(patch_data)
                pass  # close ring loop
            panel_data[detector_id] = ring_data
            pass  # close panel loop
            # pbar.finish()
        return panel_data

    def simulate_powder_pattern(self, plane_data_list, fwhm=2., noise=None):
        """
        Generates simple powder diffraction images.

        FIXME: noise isn't connected
        TODO: add hooks to the Rietveld model

        Parameters
        ----------
        plane_data_list : list, (n,)
            List of n hexrd.crystallography.PlaneData objects.
        fwhm : scalar, optional
            The FWHM of the gaussian profile in degrees. The default is 2.0.

        Returns
        -------
        img_dict : dict
            Dictionary of simulated images for each detector.

        """
        img_dict = dict.fromkeys(self.detectors)
        for det_key, panel in self.detectors.items():
            ptth, peta = panel.pixel_angles(origin=self.tvec)
            gint = np.zeros_like(ptth)
            sigm = np.radians(_fwhm_to_sigma(fwhm))
            for plane_data in plane_data_list:
                if isinstance(plane_data, PlaneData):
                    tths = plane_data.getTTh()
                    weights = np.ones_like(tths)
                else:
                    tths = [i[0] for i in plane_data]
                    weights = [i[1] for i in plane_data]
                for pk in zip(tths, weights):
                    gint += pk[1]*_gaussian_dist(ptth, pk[0], sigm)
            img_dict[det_key] = gint
        return img_dict


    def simulate_laue_pattern(self, crystal_data,
                              minEnergy=5., maxEnergy=35.,
                              rmat_s=None, grain_params=None):
        """
        Simulate Laue diffraction over the instrument.

        Parameters
        ----------
        crystal_data : TYPE
            DESCRIPTION.
        minEnergy : TYPE, optional
            DESCRIPTION. The default is 5..
        maxEnergy : TYPE, optional
            DESCRIPTION. The default is 35..
        rmat_s : TYPE, optional
            DESCRIPTION. The default is None.
        grain_params : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        TODO: revisit output; dict, or concatenated list?
        """
        results = dict.fromkeys(self.detectors)
        for det_key, panel in self.detectors.items():
            results[det_key] = panel.simulate_laue_pattern(
                crystal_data,
                minEnergy=minEnergy, maxEnergy=maxEnergy,
                rmat_s=rmat_s, tvec_s=self.tvec,
                grain_params=grain_params,
                beam_vec=self.beam_vector)
        return results

    def simulate_rotation_series(self, plane_data, grain_param_list,
                                 eta_ranges=[(-np.pi, np.pi), ],
                                 ome_ranges=[(-np.pi, np.pi), ],
                                 ome_period=(-np.pi, np.pi),
                                 wavelength=None):
        """
        Simulate a monochromatic rotation series over the instrument.

        Parameters
        ----------
        plane_data : TYPE
            DESCRIPTION.
        grain_param_list : TYPE
            DESCRIPTION.
        eta_ranges : TYPE, optional
            DESCRIPTION. The default is [(-np.pi, np.pi), ].
        ome_ranges : TYPE, optional
            DESCRIPTION. The default is [(-np.pi, np.pi), ].
        ome_period : TYPE, optional
            DESCRIPTION. The default is (-np.pi, np.pi).
        wavelength : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        TODO: revisit output; dict, or concatenated list?
        """
        results = dict.fromkeys(self.detectors)
        for det_key, panel in self.detectors.items():
            results[det_key] = panel.simulate_rotation_series(
                plane_data, grain_param_list,
                eta_ranges=eta_ranges,
                ome_ranges=ome_ranges,
                ome_period=ome_period,
                chi=self.chi, tVec_s=self.tvec,
                wavelength=wavelength)
        return results

    def pull_spots(self, plane_data, grain_params,
                   imgser_dict,
                   tth_tol=0.25, eta_tol=1., ome_tol=1.,
                   npdiv=2, threshold=10,
                   eta_ranges=[(-np.pi, np.pi), ],
                   ome_period=(-np.pi, np.pi),
                   dirname='results', filename=None, output_format='text',
                   save_spot_list=False,
                   quiet=True, check_only=False,
                   interp='nearest'):
        """
        Exctract reflection info from a rotation series.

        Input must be encoded as an OmegaImageseries object.

        Parameters
        ----------
        plane_data : TYPE
            DESCRIPTION.
        grain_params : TYPE
            DESCRIPTION.
        imgser_dict : TYPE
            DESCRIPTION.
        tth_tol : TYPE, optional
            DESCRIPTION. The default is 0.25.
        eta_tol : TYPE, optional
            DESCRIPTION. The default is 1..
        ome_tol : TYPE, optional
            DESCRIPTION. The default is 1..
        npdiv : TYPE, optional
            DESCRIPTION. The default is 2.
        threshold : TYPE, optional
            DESCRIPTION. The default is 10.
        eta_ranges : TYPE, optional
            DESCRIPTION. The default is [(-np.pi, np.pi), ].
        ome_period : TYPE, optional
            DESCRIPTION. The default is (-np.pi, np.pi).
        dirname : TYPE, optional
            DESCRIPTION. The default is 'results'.
        filename : TYPE, optional
            DESCRIPTION. The default is None.
        output_format : TYPE, optional
            DESCRIPTION. The default is 'text'.
        save_spot_list : TYPE, optional
            DESCRIPTION. The default is False.
        quiet : TYPE, optional
            DESCRIPTION. The default is True.
        check_only : TYPE, optional
            DESCRIPTION. The default is False.
        interp : TYPE, optional
            DESCRIPTION. The default is 'nearest'.

        Returns
        -------
        compl : TYPE
            DESCRIPTION.
        output : TYPE
            DESCRIPTION.

        """
        # grain parameters
        rMat_c = makeRotMatOfExpMap(grain_params[:3])
        tVec_c = grain_params[3:6]

        # grab omega ranges from first imageseries
        #
        # WARNING: all imageseries AND all wedges within are assumed to have
        # the same omega values; put in a check that they are all the same???
        oims0 = next(iter(imgser_dict.values()))
        ome_ranges = [np.radians([i['ostart'], i['ostop']])
                      for i in oims0.omegawedges.wedges]

        # delta omega in DEGREES grabbed from first imageseries in the dict
        delta_ome = oims0.omega[0, 1] - oims0.omega[0, 0]

        # make omega grid for frame expansion around reference frame
        # in DEGREES
        ndiv_ome, ome_del = make_tolerance_grid(
            delta_ome, ome_tol, 1, adjust_window=True,
        )

        # generate structuring element for connected component labeling
        if ndiv_ome == 1:
            label_struct = ndimage.generate_binary_structure(2, 2)
        else:
            label_struct = ndimage.generate_binary_structure(3, 3)

        # simulate rotation series
        sim_results = self.simulate_rotation_series(
            plane_data, [grain_params, ],
            eta_ranges=eta_ranges,
            ome_ranges=ome_ranges,
            ome_period=ome_period)

        # patch vertex generator (global for instrument)
        tol_vec = 0.5*np.radians(
            [-tth_tol, -eta_tol,
             -tth_tol,  eta_tol,
             tth_tol,  eta_tol,
             tth_tol, -eta_tol])

        # prepare output if requested
        if filename is not None and output_format.lower() == 'hdf5':
            this_filename = os.path.join(dirname, filename)
            writer = GrainDataWriter_h5(
                os.path.join(dirname, filename),
                self.write_config(), grain_params)

        # =====================================================================
        # LOOP OVER PANELS
        # =====================================================================
        iRefl = 0
        compl = []
        output = dict.fromkeys(self.detectors)
        for detector_id in self.detectors:
            # initialize text-based output writer
            if filename is not None and output_format.lower() == 'text':
                output_dir = os.path.join(
                    dirname, detector_id
                    )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                this_filename = os.path.join(
                    output_dir, filename
                )
                writer = PatchDataWriter(this_filename)

            # grab panel
            panel = self.detectors[detector_id]
            instr_cfg = panel.config_dict(
                self.chi, self.tvec,
                beam_energy=self.beam_energy,
                beam_vector=self.beam_vector
            )
            native_area = panel.pixel_area  # pixel ref area

            # pull out the OmegaImageSeries for this panel from input dict
            ome_imgser = imgser_dict[detector_id]

            # extract simulation results
            sim_results_p = sim_results[detector_id]
            hkl_ids = sim_results_p[0][0]
            hkls_p = sim_results_p[1][0]
            ang_centers = sim_results_p[2][0]
            xy_centers = sim_results_p[3][0]
            ang_pixel_size = sim_results_p[4][0]

            # now verify that full patch falls on detector...
            # ???: strictly necessary?
            #
            # patch vertex array from sim
            nangs = len(ang_centers)
            patch_vertices = (
                np.tile(ang_centers[:, :2], (1, 4)) +
                np.tile(tol_vec, (nangs, 1))
            ).reshape(4*nangs, 2)
            ome_dupl = np.tile(
                ang_centers[:, 2], (4, 1)
            ).T.reshape(len(patch_vertices), 1)

            # find vertices that all fall on the panel
            det_xy, rmats_s, on_plane = xrdutil._project_on_detector_plane(
                np.hstack([patch_vertices, ome_dupl]),
                panel.rmat, rMat_c, self.chi,
                panel.tvec, tVec_c, self.tvec,
                panel.distortion)
            _, on_panel = panel.clip_to_panel(det_xy, buffer_edges=True)

            # all vertices must be on...
            patch_is_on = np.all(on_panel.reshape(nangs, 4), axis=1)
            patch_xys = det_xy.reshape(nangs, 4, 2)[patch_is_on]

            # re-filter...
            hkl_ids = hkl_ids[patch_is_on]
            hkls_p = hkls_p[patch_is_on, :]
            ang_centers = ang_centers[patch_is_on, :]
            xy_centers = xy_centers[patch_is_on, :]
            ang_pixel_size = ang_pixel_size[patch_is_on, :]

            # TODO: add polygon testing right here!
            # done <JVB 06/21/16>
            if check_only:
                patch_output = []
                for i_pt, angs in enumerate(ang_centers):
                    # the evaluation omegas;
                    # expand about the central value using tol vector
                    ome_eval = np.degrees(angs[2]) + ome_del

                    # ...vectorize the omega_to_frame function to avoid loop?
                    frame_indices = [
                        ome_imgser.omega_to_frame(ome)[0] for ome in ome_eval
                    ]
                    if -1 in frame_indices:
                        if not quiet:
                            msg = """
                            window for (%d%d%d) falls outside omega range
                            """ % tuple(hkls_p[i_pt, :])
                            print(msg)
                        continue
                    else:
                        these_vertices = patch_xys[i_pt]
                        ijs = panel.cartToPixel(these_vertices)
                        ii, jj = polygon(ijs[:, 0], ijs[:, 1])
                        contains_signal = False
                        for i_frame in frame_indices:
                            contains_signal = contains_signal or np.any(
                                ome_imgser[i_frame][ii, jj] > threshold
                            )
                        compl.append(contains_signal)
                        patch_output.append((ii, jj, frame_indices))
            else:
                # make the tth,eta patches for interpolation
                patches = xrdutil.make_reflection_patches(
                    instr_cfg,
                    ang_centers[:, :2], ang_pixel_size,
                    omega=ang_centers[:, 2],
                    tth_tol=tth_tol, eta_tol=eta_tol,
                    rmat_c=rMat_c, tvec_c=tVec_c,
                    npdiv=npdiv, quiet=True)

                # GRAND LOOP over reflections for this panel
                patch_output = []
                for i_pt, patch in enumerate(patches):

                    # strip relevant objects out of current patch
                    vtx_angs, vtx_xy, conn, areas, xy_eval, ijs = patch

                    prows, pcols = areas.shape
                    nrm_fac = areas/float(native_area)
                    nrm_fac = nrm_fac / np.min(nrm_fac)

                    # grab hkl info
                    hkl = hkls_p[i_pt, :]
                    hkl_id = hkl_ids[i_pt]

                    # edge arrays
                    tth_edges = vtx_angs[0][0, :]
                    delta_tth = tth_edges[1] - tth_edges[0]
                    eta_edges = vtx_angs[1][:, 0]
                    delta_eta = eta_edges[1] - eta_edges[0]

                    # need to reshape eval pts for interpolation
                    xy_eval = np.vstack([xy_eval[0].flatten(),
                                         xy_eval[1].flatten()]).T

                    # the evaluation omegas;
                    # expand about the central value using tol vector
                    ome_eval = np.degrees(ang_centers[i_pt, 2]) + ome_del

                    # ???: vectorize the omega_to_frame function to avoid loop?
                    frame_indices = [
                        ome_imgser.omega_to_frame(ome)[0] for ome in ome_eval
                    ]

                    if -1 in frame_indices:
                        if not quiet:
                            msg = """
                            window for (%d%d%d) falls outside omega range
                            """ % tuple(hkl)
                            print(msg)
                        continue
                    else:
                        # initialize spot data parameters
                        # !!! maybe change these to nan to not fuck up writer
                        peak_id = -999
                        sum_int = np.nan
                        max_int = np.nan
                        meas_angs = np.nan*np.ones(3)
                        meas_xy = np.nan*np.ones(2)

                        # quick check for intensity
                        contains_signal = False
                        patch_data_raw = []
                        for i_frame in frame_indices:
                            tmp = ome_imgser[i_frame][ijs[0], ijs[1]]
                            contains_signal = contains_signal or np.any(
                                tmp > threshold
                            )
                            patch_data_raw.append(tmp)
                            pass
                        patch_data_raw = np.stack(patch_data_raw, axis=0)
                        compl.append(contains_signal)

                        if contains_signal:
                            # initialize patch data array for intensities
                            if interp.lower() == 'bilinear':
                                patch_data = np.zeros(
                                    (len(frame_indices), prows, pcols))
                                for i, i_frame in enumerate(frame_indices):
                                    patch_data[i] = \
                                        panel.interpolate_bilinear(
                                            xy_eval,
                                            ome_imgser[i_frame],
                                            pad_with_nans=False
                                        ).reshape(prows, pcols)  # * nrm_fac
                            elif interp.lower() == 'nearest':
                                patch_data = patch_data_raw  # * nrm_fac
                            else:
                                msg = "interpolation option " + \
                                    "'%s' not understood"
                                raise(RuntimeError, msg % interp)

                            # now have interpolated patch data...
                            labels, num_peaks = ndimage.label(
                                patch_data > threshold, structure=label_struct
                            )
                            slabels = np.arange(1, num_peaks + 1)

                            if num_peaks > 0:
                                peak_id = iRefl
                                coms = np.array(
                                    ndimage.center_of_mass(
                                        patch_data,
                                        labels=labels,
                                        index=slabels
                                    )
                                )
                                if num_peaks > 1:
                                    center = np.r_[patch_data.shape]*0.5
                                    center_t = np.tile(center, (num_peaks, 1))
                                    com_diff = coms - center_t
                                    closest_peak_idx = np.argmin(
                                        np.sum(com_diff**2, axis=1)
                                    )
                                else:
                                    closest_peak_idx = 0
                                    pass  # end multipeak conditional
                                coms = coms[closest_peak_idx]
                                # meas_omes = \
                                #     ome_edges[0] + (0.5 + coms[0])*delta_ome
                                meas_omes = \
                                    ome_eval[0] + coms[0]*delta_ome
                                meas_angs = np.hstack(
                                    [tth_edges[0] + (0.5 + coms[2])*delta_tth,
                                     eta_edges[0] + (0.5 + coms[1])*delta_eta,
                                     mapAngle(
                                         np.radians(meas_omes), ome_period
                                         )
                                     ]
                                )

                                # intensities
                                #   - summed is 'integrated' over interpolated
                                #     data
                                #   - max is max of raw input data
                                sum_int = np.sum(
                                    patch_data[
                                        labels == slabels[closest_peak_idx]
                                    ]
                                )
                                max_int = np.max(
                                    patch_data_raw[
                                        labels == slabels[closest_peak_idx]
                                    ]
                                )
                                # ???: Should this only use labeled pixels?
                                # Those are segmented from interpolated data,
                                # not raw; likely ok in most cases.

                                # need MEASURED xy coords
                                gvec_c = anglesToGVec(
                                    meas_angs,
                                    chi=self.chi,
                                    rMat_c=rMat_c,
                                    bHat_l=self.beam_vector)
                                rMat_s = makeOscillRotMat(
                                    [self.chi, meas_angs[2]]
                                )
                                meas_xy = gvecToDetectorXY(
                                    gvec_c,
                                    panel.rmat, rMat_s, rMat_c,
                                    panel.tvec, self.tvec, tVec_c,
                                    beamVec=self.beam_vector)
                                if panel.distortion is not None:
                                    # FIXME: distortion handling
                                    meas_xy = panel.distortion[0](
                                        np.atleast_2d(meas_xy),
                                        panel.distortion[1],
                                        invert=True).flatten()
                                    pass
                                # FIXME: why is this suddenly necessary???
                                meas_xy = meas_xy.squeeze()
                                pass  # end num_peaks > 0
                        else:
                            patch_data = patch_data_raw
                            pass  # end contains_signal
                        # write output
                        if filename is not None:
                            if output_format.lower() == 'text':
                                writer.dump_patch(
                                    peak_id, hkl_id, hkl, sum_int, max_int,
                                    ang_centers[i_pt], meas_angs,
                                    xy_centers[i_pt], meas_xy)
                            elif output_format.lower() == 'hdf5':
                                xyc_arr = xy_eval.reshape(
                                    prows, pcols, 2
                                ).transpose(2, 0, 1)
                                writer.dump_patch(
                                    detector_id, iRefl, peak_id, hkl_id, hkl,
                                    tth_edges, eta_edges, np.radians(ome_eval),
                                    xyc_arr, ijs, frame_indices, patch_data,
                                    ang_centers[i_pt], xy_centers[i_pt],
                                    meas_angs, meas_xy)
                            pass  # end conditional on write output
                        pass  # end conditional on check only
                        patch_output.append([
                                peak_id, hkl_id, hkl, sum_int, max_int,
                                ang_centers[i_pt], meas_angs, meas_xy,
                                ])
                        iRefl += 1
                    pass  # end patch conditional
                pass  # end patch loop
            output[detector_id] = patch_output
            if filename is not None and output_format.lower() == 'text':
                writer.close()
            pass  # end detector loop
        if filename is not None and output_format.lower() == 'hdf5':
            writer.close()
        return compl, output

    """def fit_grain(self, grain_params, data_dir='results'):"""

    pass  # end class: HEDMInstrument


class PlanarDetector(object):
    """Base class for 2D planar, rectangular row-column detector"""

    __pixelPitchUnit = 'mm'

    def __init__(self,
                 rows=2048, cols=2048,
                 pixel_size=(0.2, 0.2),
                 tvec=np.r_[0., 0., -1000.],
                 tilt=ct.zeros_3,
                 name='default',
                 bvec=ct.beam_vec,
                 evec=ct.eta_vec,
                 saturation_level=None,
                 panel_buffer=None,
                 roi=None,
                 distortion=None):
        """
        Instantiate a PlanarDetector object.

        Parameters
        ----------
        rows : TYPE, optional
            DESCRIPTION. The default is 2048.
        cols : TYPE, optional
            DESCRIPTION. The default is 2048.
        pixel_size : TYPE, optional
            DESCRIPTION. The default is (0.2, 0.2).
        tvec : TYPE, optional
            DESCRIPTION. The default is np.r_[0., 0., -1000.].
        tilt : TYPE, optional
            DESCRIPTION. The default is ct.zeros_3.
        name : TYPE, optional
            DESCRIPTION. The default is 'default'.
        bvec : TYPE, optional
            DESCRIPTION. The default is ct.beam_vec.
        evec : TYPE, optional
            DESCRIPTION. The default is ct.eta_vec.
        saturation_level : TYPE, optional
            DESCRIPTION. The default is None.
        panel_buffer : TYPE, optional
            If a scalar or len(2) array_like, the interpretation is a border
            in mm. If an array with shape (nrows, ncols), interpretation is a
            boolean with True marking valid pixels.  The default is None.
        roi : TYPE, optional
            DESCRIPTION. The default is None.
        distortion : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._name = name

        self._rows = rows
        self._cols = cols

        self._pixel_size_row = pixel_size[0]
        self._pixel_size_col = pixel_size[1]

        self._saturation_level = saturation_level

        self._panel_buffer = panel_buffer

        self._roi = roi

        self._tvec = np.array(tvec).flatten()
        self._tilt = np.array(tilt).flatten()

        self._bvec = np.array(bvec).flatten()
        self._evec = np.array(evec).flatten()

        self._distortion = distortion

        #
        # set up calibration parameter list and refinement flags
        #
        # order for a single detector will be
        #
        #     [tilt, translation, <distortion>]
        dparams = []
        if self._distortion is not None:
            # need dparams
            # FIXME: must update when we fix distortion
            dparams.append(np.atleast_1d(self._distortion[1]).flatten())
        dparams = np.array(dparams).flatten()
        self._calibration_parameters = np.hstack(
                [self._tilt, self._tvec, dparams]
            )
        self._calibration_flags = np.hstack(
                [panel_calibration_flags_DFLT,
                 np.zeros(len(dparams), dtype=bool)]
            )
        return

    # detector ID
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, s):
        assert isinstance(s, str), "requires string input"
        self._name = s

    # properties for physical size of rectangular detector
    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, x):
        assert isinstance(x, int)
        self._rows = x

    @property
    def cols(self):
        return self._cols

    @cols.setter
    def cols(self, x):
        assert isinstance(x, int)
        self._cols = x

    @property
    def pixel_size_row(self):
        return self._pixel_size_row

    @pixel_size_row.setter
    def pixel_size_row(self, x):
        self._pixel_size_row = float(x)

    @property
    def pixel_size_col(self):
        return self._pixel_size_col

    @pixel_size_col.setter
    def pixel_size_col(self, x):
        self._pixel_size_col = float(x)

    @property
    def pixel_area(self):
        return self.pixel_size_row * self.pixel_size_col

    @property
    def saturation_level(self):
        return self._saturation_level

    @saturation_level.setter
    def saturation_level(self, x):
        if x is not None:
            assert np.isreal(x)
        self._saturation_level = x

    @property
    def panel_buffer(self):
        return self._panel_buffer

    @panel_buffer.setter
    def panel_buffer(self, x):
        """if not None, a buffer in mm (x, y)"""
        if x is not None:
            assert len(x) == 2 or x.ndim == 2
        self._panel_buffer = x

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, vertex_array):
        """
        vertex array must be

        [[r0, c0], [r1, c1], ..., [rn, cn]]

        and have len >= 3

        does NOT need to repeat start vertex for closure
        """
        if vertex_array is not None:
            assert len(vertex_array) >= 3
        self._roi = vertex_array

    @property
    def row_dim(self):
        return self.rows * self.pixel_size_row

    @property
    def col_dim(self):
        return self.cols * self.pixel_size_col

    @property
    def row_pixel_vec(self):
        return self.pixel_size_row*(0.5*(self.rows-1)-np.arange(self.rows))

    @property
    def row_edge_vec(self):
        return self.pixel_size_row*(0.5*self.rows-np.arange(self.rows+1))

    @property
    def col_pixel_vec(self):
        return self.pixel_size_col*(np.arange(self.cols)-0.5*(self.cols-1))

    @property
    def col_edge_vec(self):
        return self.pixel_size_col*(np.arange(self.cols+1)-0.5*self.cols)

    @property
    def corner_ul(self):
        return np.r_[-0.5 * self.col_dim,  0.5 * self.row_dim]

    @property
    def corner_ll(self):
        return np.r_[-0.5 * self.col_dim, -0.5 * self.row_dim]

    @property
    def corner_lr(self):
        return np.r_[0.5 * self.col_dim, -0.5 * self.row_dim]

    @property
    def corner_ur(self):
        return np.r_[0.5 * self.col_dim,  0.5 * self.row_dim]

    @property
    def tvec(self):
        return self._tvec

    @tvec.setter
    def tvec(self, x):
        x = np.array(x).flatten()
        assert len(x) == 3, 'input must have length = 3'
        self._tvec = x

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, x):
        assert len(x) == 3, 'input must have length = 3'
        self._tilt = np.array(x).squeeze()

    @property
    def bvec(self):
        return self._bvec

    @bvec.setter
    def bvec(self, x):
        x = np.array(x).flatten()
        assert len(x) == 3 and sum(x*x) > 1-ct.sqrt_epsf, \
            'input must have length = 3 and have unit magnitude'
        self._bvec = x

    @property
    def evec(self):
        return self._evec

    @evec.setter
    def evec(self, x):
        x = np.array(x).flatten()
        assert len(x) == 3 and sum(x*x) > 1-ct.sqrt_epsf, \
            'input must have length = 3 and have unit magnitude'
        self._evec = x

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, x):
        """
        Probably should make distortion a class...
        ***FIX THIS***
        """
        assert len(x) == 2 and hasattr(x[0], '__call__'), \
            'distortion must be a tuple: (<func>, params)'
        self._distortion = x

    @property
    def rmat(self):
        return makeRotMatOfExpMap(self.tilt)

    @property
    def normal(self):
        return self.rmat[:, 2]

    @property
    def beam_position(self):
        """
        returns the coordinates of the beam in the cartesian detector
        frame {Xd, Yd, Zd}.  NaNs if no intersection.
        """
        output = np.nan * np.ones(2)
        b_dot_n = np.dot(self.bvec, self.normal)
        if np.logical_and(
            abs(b_dot_n) > ct.sqrt_epsf,
            np.sign(b_dot_n) == -1
        ):
            u = np.dot(self.normal, self.tvec) / b_dot_n
            p2_l = u*self.bvec
            p2_d = np.dot(self.rmat.T, p2_l - self.tvec)
            output = p2_d[:2]
        return output

    # ...memoize???
    @property
    def pixel_coords(self):
        pix_i, pix_j = np.meshgrid(
            self.row_pixel_vec, self.col_pixel_vec,
            indexing='ij')
        return pix_i, pix_j

    @property
    def calibration_parameters(self):
        #
        # set up calibration parameter list and refinement flags
        #
        # order for a single detector will be
        #
        #     [tilt, translation, <distortion>]
        dparams = []
        if self.distortion is not None:
            # need dparams
            # FIXME: must update when we fix distortion
            dparams.append(np.atleast_1d(self.distortion[1]).flatten())
        dparams = np.array(dparams).flatten()
        self._calibration_parameters = np.hstack(
                [self.tilt, self.tvec, dparams]
            )
        return self._calibration_parameters

    @property
    def calibration_flags(self):
        return self._calibration_flags

    @calibration_flags.setter
    def calibration_flags(self, x):
        x = np.array(x, dtype=bool).flatten()
        if len(x) != len(self._calibration_flags):
            raise RuntimeError(
                "length of parameter list must be %d; you gave %d"
                % (len(self._calibration_flags), len(x))
            )
        self._calibration_flags = x

    # =========================================================================
    # METHODS
    # =========================================================================

    def config_dict(self, chi=0, tvec=ct.zeros_3,
                    beam_energy=beam_energy_DFLT, beam_vector=ct.beam_vec,
                    sat_level=None, panel_buffer=None):
        """
        Return a dictionary of detector parameters.

        Optional instrument level parameters.  This is a convenience function
        to work with the APIs in several functions in xrdutil.

        Parameters
        ----------
        chi : float, optional
            DESCRIPTION. The default is 0.
        tvec : array_like (3,), optional
            DESCRIPTION. The default is ct.zeros_3.
        beam_energy : float, optional
            DESCRIPTION. The default is beam_energy_DFLT.
        beam_vector : aray_like (3,), optional
            DESCRIPTION. The default is ct.beam_vec.
        sat_level : scalar, optional
            DESCRIPTION. The default is None.
        panel_buffer : scalar, array_like (2,), optional
            DESCRIPTION. The default is None.

        Returns
        -------
        config_dict : dict
            DESCRIPTION.

        """
        config_dict = {}

        # =====================================================================
        # DETECTOR PARAMETERS
        # =====================================================================
        if sat_level is None:
            sat_level = self.saturation_level

        if panel_buffer is None:
            # FIXME: won't work right if it is an array
            panel_buffer = self.panel_buffer
        if isinstance(panel_buffer, np.ndarray):
            panel_buffer = panel_buffer.flatten().tolist()

        det_dict = dict(
                transform=dict(
                    tilt=self.tilt.tolist(),
                    translation=self.tvec.tolist(),
                ),
                pixels=dict(
                    rows=self.rows,
                    columns=self.cols,
                    size=[self.pixel_size_row, self.pixel_size_col],
                )
            )

        # saturation level
        det_dict['saturation_level'] = sat_level

        # panel buffer
        # FIXME if it is an array, the write will be a mess
        det_dict['buffer'] = panel_buffer

        if self.distortion is not None:
            # FIXME: HARD CODED DISTORTION!
            dist_d = dict(
                function_name='GE_41RT',
                parameters=np.r_[self.distortion[1]].tolist()
            )
            det_dict['distortion'] = dist_d

        # =====================================================================
        # SAMPLE STAGE PARAMETERS
        # =====================================================================
        stage_dict = dict(
            chi=chi,
            translation=tvec.tolist()
        )

        # =====================================================================
        # BEAM PARAMETERS
        # =====================================================================
        beam_dict = dict(
            energy=beam_energy,
            vector=beam_vector
        )

        config_dict['detector'] = det_dict
        config_dict['oscillation_stage'] = stage_dict
        config_dict['beam'] = beam_dict

        return config_dict

    def pixel_angles(self, origin=ct.zeros_3):
        assert len(origin) == 3, "origin must have 3 elemnts"
        pix_i, pix_j = self.pixel_coords
        xy = np.ascontiguousarray(
            np.vstack([
                pix_j.flatten(), pix_i.flatten()
                ]).T
            )
        if self.distortion is not None:
            # FIXME: old-style distortion
            xy = self.distortion[0](xy, self.distortion[1], invert=False)
        angs, g_vec = detectorXYToGvec(
            xy, self.rmat, ct.identity_3x3,
            self.tvec, ct.zeros_3, origin,
            beamVec=self.bvec, etaVec=self.evec)
        del(g_vec)
        tth = angs[0].reshape(self.rows, self.cols)
        eta = angs[1].reshape(self.rows, self.cols)
        return tth, eta

    def cartToPixel(self, xy_det, pixels=False):
        """
        Convert vstacked array or list of [x,y] points in the center-based
        cartesian frame {Xd, Yd, Zd} to (i, j) edge-based indices

        i is the row index, measured from the upper-left corner
        j is the col index, measured from the upper-left corner

        if pixels=True, then (i,j) are integer pixel indices.
        else (i,j) are continuous coords
        """
        xy_det = np.atleast_2d(xy_det)

        npts = len(xy_det)

        tmp_ji = xy_det - np.tile(self.corner_ul, (npts, 1))
        i_pix = -tmp_ji[:, 1] / self.pixel_size_row - 0.5
        j_pix = tmp_ji[:, 0] / self.pixel_size_col - 0.5

        ij_det = np.vstack([i_pix, j_pix]).T
        if pixels:
            ij_det = np.array(np.round(ij_det), dtype=int)
        return ij_det

    def pixelToCart(self, ij_det):
        """
        Convert vstacked array or list of [i,j] pixel indices
        (or UL corner-based points) and convert to (x,y) in the
        cartesian frame {Xd, Yd, Zd}
        """
        ij_det = np.atleast_2d(ij_det)

        x = (ij_det[:, 1] + 0.5)*self.pixel_size_col\
            + self.corner_ll[0]
        y = (self.rows - ij_det[:, 0] - 0.5)*self.pixel_size_row\
            + self.corner_ll[1]
        return np.vstack([x, y]).T

    def angularPixelSize(self, xy, rMat_s=None, tVec_s=None, tVec_c=None):
        """
        Wraps xrdutil.angularPixelSize
        """
        # munge kwargs
        if rMat_s is None:
            rMat_s = ct.identity_3x3
        if tVec_s is None:
            tVec_s = ct.zeros_3x1
        if tVec_c is None:
            tVec_c = ct.zeros_3x1

        # call function
        ang_ps = xrdutil.angularPixelSize(
            xy, (self.pixel_size_row, self.pixel_size_col),
            self.rmat, rMat_s,
            self.tvec, tVec_s, tVec_c,
            distortion=self.distortion,
            beamVec=self.bvec, etaVec=self.evec)
        return ang_ps

    def clip_to_panel(self, xy, buffer_edges=True):
        """
        if self.roi is not None, uses it by default

        TODO: check if need shape kwarg
        TODO: optimize ROI search better than list comprehension below
        TODO: panel_buffer can be a 2-d boolean mask, but needs testing

        """
        xy = np.atleast_2d(xy)

        if self.roi is not None:
            ij_crds = self.cartToPixel(xy, pixels=True)
            ii, jj = polygon(self.roi[:, 0], self.roi[:, 1],
                             shape=(self.rows, self.cols))
            on_panel_rows = [i in ii for i in ij_crds[:, 0]]
            on_panel_cols = [j in jj for j in ij_crds[:, 1]]
            on_panel = np.logical_and(on_panel_rows, on_panel_cols)
        else:
            xlim = 0.5*self.col_dim
            ylim = 0.5*self.row_dim
            if buffer_edges and self.panel_buffer is not None:
                if self.panel_buffer.ndim == 2:
                    pix = self.cartToPixel(xy, pixels=True)

                    roff = np.logical_or(pix[:, 0] < 0, pix[:, 0] >= self.rows)
                    coff = np.logical_or(pix[:, 1] < 0, pix[:, 1] >= self.cols)

                    idx = np.logical_or(roff, coff)

                    pix[idx, :] = 0

                    on_panel = self.panel_buffer[pix[:, 0], pix[:, 1]]
                    on_panel[idx] = False
                else:
                    xlim -= self.panel_buffer[0]
                    ylim -= self.panel_buffer[1]
                    on_panel_x = np.logical_and(
                        xy[:, 0] >= -xlim, xy[:, 0] <= xlim
                    )
                    on_panel_y = np.logical_and(
                        xy[:, 1] >= -ylim, xy[:, 1] <= ylim
                    )
                    on_panel = np.logical_and(on_panel_x, on_panel_y)
            elif not buffer_edges or self.panel_buffer is None:
                on_panel_x = np.logical_and(
                    xy[:, 0] >= -xlim, xy[:, 0] <= xlim
                )
                on_panel_y = np.logical_and(
                    xy[:, 1] >= -ylim, xy[:, 1] <= ylim
                )
                on_panel = np.logical_and(on_panel_x, on_panel_y)
        return xy[on_panel, :], on_panel

    def cart_to_angles(self, xy_data):
        """
        TODO: distortion
        """
        rmat_s = ct.identity_3x3
        tvec_s = ct.zeros_3
        tvec_c = ct.zeros_3
        angs, g_vec = detectorXYToGvec(
            xy_data, self.rmat, rmat_s,
            self.tvec, tvec_s, tvec_c,
            beamVec=self.bvec, etaVec=self.evec)
        tth_eta = np.vstack([angs[0], angs[1]]).T
        return tth_eta, g_vec

    def angles_to_cart(self, tth_eta, rmat_c=None, tvec_c=None):
        """
        TODO: distortion
        """
        if rmat_c is None:
            rmat_c = ct.identity_3x3
        if tvec_c is None:
            tvec_c = ct.zeros_3

        rmat_s = ct.identity_3x3
        tvec_s = ct.zeros_3

        angs = np.hstack([tth_eta, np.zeros((len(tth_eta), 1))])

        xy_det = gvecToDetectorXY(
            anglesToGVec(angs, bHat_l=self.bvec, eHat_l=self.evec),
            self.rmat, rmat_s, rmat_c,
            self.tvec, tvec_s, tvec_c,
            beamVec=self.bvec)
        return xy_det

    def interpolate_nearest(self, xy, img, pad_with_nans=True):
        """
        TODO: revisit normalization in here?

        """
        is_2d = img.ndim == 2
        right_shape = img.shape[0] == self.rows and img.shape[1] == self.cols
        assert is_2d and right_shape,\
            "input image must be 2-d with shape (%d, %d)"\
            % (self.rows, self.cols)

        # initialize output with nans
        if pad_with_nans:
            int_xy = np.nan*np.ones(len(xy))
        else:
            int_xy = np.zeros(len(xy))

        # clip away points too close to or off the edges of the detector
        xy_clip, on_panel = self.clip_to_panel(xy, buffer_edges=True)

        # get pixel indices of clipped points
        i_src = cellIndices(self.row_pixel_vec, xy_clip[:, 1])
        j_src = cellIndices(self.col_pixel_vec, xy_clip[:, 0])

        # next interpolate across cols
        int_vals = img[i_src, j_src]
        int_xy[on_panel] = int_vals
        return int_xy

    def interpolate_bilinear(self, xy, img, pad_with_nans=True):
        """
        Interpolate an image array at the specified cartesian points.

        Parameters
        ----------
        xy : array_like, (n, 2)
            Array of cartesian coordinates in the image plane at which
            to evaluate intensity.
        img : array_like
            2-dimensional image array.
        pad_with_nans : bool, optional
            Toggle for assigning NaN to points that fall off the detector.
            The default is True.

        Returns
        -------
        int_xy : array_like, (n,)
            The array of interpolated intensities at each of the n input
            coordinates.

        Notes
        -----
        TODO: revisit normalization in here?
        """

        is_2d = img.ndim == 2
        right_shape = img.shape[0] == self.rows and img.shape[1] == self.cols
        assert is_2d and right_shape,\
            "input image must be 2-d with shape (%d, %d)"\
            % (self.rows, self.cols)

        # initialize output with nans
        if pad_with_nans:
            int_xy = np.nan*np.ones(len(xy))
        else:
            int_xy = np.zeros(len(xy))

        # clip away points too close to or off the edges of the detector
        xy_clip, on_panel = self.clip_to_panel(xy, buffer_edges=True)

        # grab fractional pixel indices of clipped points
        ij_frac = self.cartToPixel(xy_clip)

        # get floors/ceils from array of pixel _centers_
        # and fix indices running off the pixel centers
        # !!! notice we already clipped points to the panel!
        i_floor = cellIndices(self.row_pixel_vec, xy_clip[:, 1])
        i_floor_img = _fix_indices(i_floor, 0, self.rows - 1)

        j_floor = cellIndices(self.col_pixel_vec, xy_clip[:, 0])
        j_floor_img = _fix_indices(j_floor, 0, self.cols - 1)

        # ceilings from floors
        i_ceil = i_floor + 1
        i_ceil_img = _fix_indices(i_ceil, 0, self.rows - 1)

        j_ceil = j_floor + 1
        j_ceil_img = _fix_indices(j_ceil, 0, self.cols - 1)

        # first interpolate at top/bottom rows
        row_floor_int = \
            (j_ceil - ij_frac[:, 1])*img[i_floor_img, j_floor_img] \
            + (ij_frac[:, 1] - j_floor)*img[i_floor_img, j_ceil_img]
        row_ceil_int = \
            (j_ceil - ij_frac[:, 1])*img[i_ceil_img, j_floor_img] \
            + (ij_frac[:, 1] - j_floor)*img[i_ceil_img, j_ceil_img]

        # next interpolate across cols
        int_vals = \
            (i_ceil - ij_frac[:, 0])*row_floor_int \
            + (ij_frac[:, 0] - i_floor)*row_ceil_int
        int_xy[on_panel] = int_vals
        return int_xy

    def make_powder_rings(
            self, pd, merge_hkls=False, delta_tth=None,
            delta_eta=10., eta_period=None,
            rmat_s=ct.identity_3x3,  tvec_s=ct.zeros_3,
            tvec_c=ct.zeros_3, full_output=False):
        """
        Generate points on Debye_Scherrer rings over the detector.

        !!! it is assuming that rmat_s is built from (chi, ome) as it the case
            for HEDM!

        Parameters
        ----------
        pd : TYPE
            DESCRIPTION.
        merge_hkls : TYPE, optional
            DESCRIPTION. The default is False.
        delta_tth : TYPE, optional
            DESCRIPTION. The default is None.
        delta_eta : TYPE, optional
            DESCRIPTION. The default is 10..
        eta_period : TYPE, optional
            DESCRIPTION. The default is None.
        rmat_s : TYPE, optional
            DESCRIPTION. The default is ct.identity_3x3.
        tvec_s : TYPE, optional
            DESCRIPTION. The default is ct.zeros_3.
        tvec_c : TYPE, optional
            DESCRIPTION. The default is ct.zeros_3.
        full_output : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # in case you want to give it tth angles directly
        if hasattr(pd, '__len__'):
            tth = np.array(pd).flatten()
            if delta_tth is None:
                raise RuntimeError(
                    "If supplying a 2theta list as first arg, "
                    + "must supply a delta_tth")
            sector_vertices = np.tile(
                0.5*np.radians([-delta_tth, -delta_eta,
                                -delta_tth, delta_eta,
                                delta_tth, delta_eta,
                                delta_tth, -delta_eta,
                                0.0, 0.0]), (len(tth), 1)
                )
            # Convert to radians as is done below
            del_eta = np.radians(delta_eta)
        else:
            # Okay, we have a PlaneData object
            try:
                pd = PlaneData.makeNew(pd)    # make a copy to munge
            except(TypeError):
                # !!! have some other object here, likely a dummy plane data
                # object of some sort...
                pass

            if delta_tth is not None:
                pd.tThWidth = np.radians(delta_tth)
            else:
                delta_tth = np.degrees(pd.tThWidth)

            # conversions, meh...
            del_eta = np.radians(delta_eta)

            # do merging if asked
            if merge_hkls:
                _, tth_ranges = pd.getMergedRanges()
                tth = np.array([0.5*sum(i) for i in tth_ranges])
            else:
                tth_ranges = pd.getTThRanges()
                tth = pd.getTTh()
            tth_pm = tth_ranges - np.tile(tth, (2, 1)).T
            sector_vertices = np.vstack(
                [[i[0], -del_eta,
                  i[0], del_eta,
                  i[1], del_eta,
                  i[1], -del_eta,
                  0.0, 0.0]
                 for i in tth_pm])

        # for generating rings, make eta vector in correct period
        if eta_period is None:
            eta_period = (-np.pi, np.pi)
        neta = int(360./float(delta_eta))

        # this is the vector of ETA EDGES
        eta_edges = mapAngle(
            np.radians(
                delta_eta*np.linspace(0., neta, num=neta + 1)
            ) + eta_period[0],
            eta_period
        )

        # get eta bin centers from edges
        """
        # !!! this way is probably overkill, since we have delta eta
        eta_centers = np.average(
            np.vstack([eta[:-1], eta[1:]),
            axis=0)
        """
        # !!! should be safe as eta_edges are monotonic
        eta_centers = eta_edges[:-1] + 0.5*del_eta

        # get chi and ome from rmat_s
        # ??? not needed chi = np.arctan2(rmat_s[2, 1], rmat_s[1, 1])
        ome = np.arctan2(rmat_s[0, 2], rmat_s[0, 0])

        # make list of angle tuples
        angs = [
            np.vstack(
                [i*np.ones(neta), eta_centers, ome*np.ones(neta)]
            ) for i in tth
        ]

        # need xy coords and pixel sizes
        valid_ang = []
        valid_xy = []
        map_indices = []
        npp = 5  # [ll, ul, ur, lr, center]
        for i_ring in range(len(angs)):
            # expand angles to patch vertices
            these_angs = angs[i_ring].T
            patch_vertices = (
                np.tile(these_angs[:, :2], (1, npp))
                + np.tile(sector_vertices[i_ring], (neta, 1))
            ).reshape(npp*neta, 2)

            # duplicate ome array
            ome_dupl = np.tile(
                these_angs[:, 2], (npp, 1)
            ).T.reshape(npp*neta, 1)

            # find vertices that all fall on the panel
            gVec_ring_l = anglesToGVec(
                np.hstack([patch_vertices, ome_dupl]),
                bHat_l=self.bvec)
            all_xy = gvecToDetectorXY(
                gVec_ring_l,
                self.rmat, rmat_s, ct.identity_3x3,
                self.tvec, tvec_s, tvec_c,
                beamVec=self.bvec)
            if self.distortion is not None:
                # FIXME
                all_xy = self.distortion[0](
                    all_xy,
                    self.distortion[1],
                    invert=True)
            _, on_panel = self.clip_to_panel(all_xy)

            # all vertices must be on...
            patch_is_on = np.all(on_panel.reshape(neta, npp), axis=1)
            patch_xys = all_xy.reshape(neta, 5, 2)[patch_is_on]

            # the surving indices
            idx = np.where(patch_is_on)[0]

            # form output arrays
            valid_ang.append(these_angs[patch_is_on, :2])
            valid_xy.append(patch_xys[:, -1, :].squeeze())
            map_indices.append(idx)
            pass
        # ??? is this option necessary?
        if full_output:
            return valid_ang, valid_xy, map_indices, eta_edges
        else:
            return valid_ang, valid_xy

    def map_to_plane(self, pts, rmat, tvec):
        """
        Map detctor points to specified plane.

        Parameters
        ----------
        pts : TYPE
            DESCRIPTION.
        rmat : TYPE
            DESCRIPTION.
        tvec : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        Notes
        -----
        by convention:

        n * (u*pts_l - tvec) = 0

        [pts]_l = rmat*[pts]_m + tvec

        """
        # arg munging
        pts = np.atleast_2d(pts)
        npts = len(pts)

        # map plane normal & translation vector, LAB FRAME
        nvec_map_lab = rmat[:, 2].reshape(3, 1)
        tvec_map_lab = np.atleast_2d(tvec).reshape(3, 1)
        tvec_d_lab = np.atleast_2d(self.tvec).reshape(3, 1)

        # put pts as 3-d in panel CS and transform to 3-d lab coords
        pts_det = np.hstack([pts, np.zeros((npts, 1))])
        pts_lab = np.dot(self.rmat, pts_det.T) + tvec_d_lab

        # scaling along pts vectors to hit map plane
        u = np.dot(nvec_map_lab.T, tvec_map_lab) \
            / np.dot(nvec_map_lab.T, pts_lab)

        # pts on map plane, in LAB FRAME
        pts_map_lab = np.tile(u, (3, 1)) * pts_lab

        return np.dot(rmat.T, pts_map_lab - tvec_map_lab)[:2, :].T

    def simulate_rotation_series(self, plane_data, grain_param_list,
                                 eta_ranges=[(-np.pi, np.pi), ],
                                 ome_ranges=[(-np.pi, np.pi), ],
                                 ome_period=(-np.pi, np.pi),
                                 chi=0., tVec_s=ct.zeros_3,
                                 wavelength=None):
        """
        Simulate a monochromatic rotation series for a list of grains.

        Parameters
        ----------
        plane_data : TYPE
            DESCRIPTION.
        grain_param_list : TYPE
            DESCRIPTION.
        eta_ranges : TYPE, optional
            DESCRIPTION. The default is [(-np.pi, np.pi), ].
        ome_ranges : TYPE, optional
            DESCRIPTION. The default is [(-np.pi, np.pi), ].
        ome_period : TYPE, optional
            DESCRIPTION. The default is (-np.pi, np.pi).
        chi : TYPE, optional
            DESCRIPTION. The default is 0..
        tVec_s : TYPE, optional
            DESCRIPTION. The default is ct.zeros_3.
        wavelength : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        valid_ids : TYPE
            DESCRIPTION.
        valid_hkls : TYPE
            DESCRIPTION.
        valid_angs : TYPE
            DESCRIPTION.
        valid_xys : TYPE
            DESCRIPTION.
        ang_pixel_size : TYPE
            DESCRIPTION.

        """
        # grab B-matrix from plane data
        bMat = plane_data.latVecOps['B']

        # reconcile wavelength
        #   * added sanity check on exclusions here; possible to
        #   * make some reflections invalid (NaN)
        if wavelength is None:
            wavelength = plane_data.wavelength
        else:
            if plane_data.wavelength != wavelength:
                plane_data.wavelength = ct.keVToAngstrom(wavelength)
        assert not np.any(np.isnan(plane_data.getTTh())),\
            "plane data exclusions incompatible with wavelength"

        # vstacked G-vector id, h, k, l
        full_hkls = xrdutil._fetch_hkls_from_planedata(plane_data)

        """ LOOP OVER GRAINS """
        valid_ids = []
        valid_hkls = []
        valid_angs = []
        valid_xys = []
        ang_pixel_size = []
        for gparm in grain_param_list:

            # make useful parameters
            rMat_c = makeRotMatOfExpMap(gparm[:3])
            tVec_c = gparm[3:6]
            vInv_s = gparm[6:]

            # All possible bragg conditions as vstacked [tth, eta, ome]
            # for each omega solution
            angList = np.vstack(
                oscillAnglesOfHKLs(
                    full_hkls[:, 1:], chi,
                    rMat_c, bMat, wavelength,
                    vInv=vInv_s,
                    )
                )

            # filter by eta and omega ranges
            # ??? get eta range from detector?
            allAngs, allHKLs = xrdutil._filter_hkls_eta_ome(
                full_hkls, angList, eta_ranges, ome_ranges
                )
            allAngs[:, 2] = mapAngle(allAngs[:, 2], ome_period)

            # find points that fall on the panel
            det_xy, rMat_s, on_plane = xrdutil._project_on_detector_plane(
                allAngs,
                self.rmat, rMat_c, chi,
                self.tvec, tVec_c, tVec_s,
                self.distortion)
            xys_p, on_panel = self.clip_to_panel(det_xy)
            valid_xys.append(xys_p)

            # filter angs and hkls that are on the detector plane
            # !!! check this -- seems unnecessary but the results of
            #     _project_on_detector_plane() can have len < the input.
            #     the output of _project_on_detector_plane has been modified to
            #     hand back the index array to remedy this JVB 2020-05-27
            filtered_angs = np.atleast_2d(allAngs[on_plane, :])
            filtered_hkls = np.atleast_2d(allHKLs[on_plane, :])

            # grab hkls and gvec ids for this panel
            valid_hkls.append(filtered_hkls[on_panel, 1:])
            valid_ids.append(filtered_hkls[on_panel, 0])

            # reflection angles (voxel centers) and pixel size in (tth, eta)
            valid_angs.append(filtered_angs[on_panel, :])
            ang_pixel_size.append(self.angularPixelSize(xys_p))
        return valid_ids, valid_hkls, valid_angs, valid_xys, ang_pixel_size

    def simulate_laue_pattern(self, crystal_data,
                              minEnergy=5., maxEnergy=35.,
                              rmat_s=None, tvec_s=None,
                              grain_params=None,
                              beam_vec=None):
        """
        """
        if isinstance(crystal_data, PlaneData):

            plane_data = crystal_data

            # grab the expanded list of hkls from plane_data
            hkls = np.hstack(plane_data.getSymHKLs())

            # and the unit plane normals (G-vectors) in CRYSTAL FRAME
            gvec_c = np.dot(plane_data.latVecOps['B'], hkls)
        elif len(crystal_data) == 2:
            # !!! should clean this up
            hkls = np.array(crystal_data[0])
            bmat = crystal_data[1]
            gvec_c = np.dot(bmat, hkls)
        else:
            raise(RuntimeError, 'argument list not understood')
        nhkls_tot = hkls.shape[1]

        # parse energy ranges
        # TODO: allow for spectrum parsing
        multipleEnergyRanges = False
        if hasattr(maxEnergy, '__len__'):
            assert len(maxEnergy) == len(minEnergy), \
                'energy cutoff ranges must have the same length'
            multipleEnergyRanges = True
            lmin = []
            lmax = []
            for i in range(len(maxEnergy)):
                lmin.append(ct.keVToAngstrom(maxEnergy[i]))
                lmax.append(ct.keVToAngstrom(minEnergy[i]))
        else:
            lmin = ct.keVToAngstrom(maxEnergy)
            lmax = ct.keVToAngstrom(minEnergy)

        # parse grain parameters kwarg
        if grain_params is None:
            grain_params = np.atleast_2d(
                np.hstack([np.zeros(6), ct.identity_6x1])
            )
        n_grains = len(grain_params)

        # sample rotation
        if rmat_s is None:
            rmat_s = ct.identity_3x3

        # dummy translation vector... make input
        if tvec_s is None:
            tvec_s = ct.zeros_3

        # beam vector
        if beam_vec is None:
            beam_vec = ct.beam_vec

        # =========================================================================
        # LOOP OVER GRAINS
        # =========================================================================

        # pre-allocate output arrays
        xy_det = np.nan*np.ones((n_grains, nhkls_tot, 2))
        hkls_in = np.nan*np.ones((n_grains, 3, nhkls_tot))
        angles = np.nan*np.ones((n_grains, nhkls_tot, 2))
        dspacing = np.nan*np.ones((n_grains, nhkls_tot))
        energy = np.nan*np.ones((n_grains, nhkls_tot))
        for iG, gp in enumerate(grain_params):
            rmat_c = makeRotMatOfExpMap(gp[:3])
            tvec_c = gp[3:6].reshape(3, 1)
            vInv_s = mutil.vecMVToSymm(gp[6:].reshape(6, 1))

            # stretch them: V^(-1) * R * Gc
            gvec_s_str = np.dot(vInv_s, np.dot(rmat_c, gvec_c))
            ghat_c_str = mutil.unitVector(np.dot(rmat_c.T, gvec_s_str))

            # project
            dpts = gvecToDetectorXY(ghat_c_str.T,
                                    self.rmat, rmat_s, rmat_c,
                                    self.tvec, tvec_s, tvec_c,
                                    beamVec=beam_vec)

            # check intersections with detector plane
            canIntersect = ~np.isnan(dpts[:, 0])
            npts_in = sum(canIntersect)

            if np.any(canIntersect):
                dpts = dpts[canIntersect, :].reshape(npts_in, 2)
                dhkl = hkls[:, canIntersect].reshape(3, npts_in)

                # back to angles
                tth_eta, gvec_l = detectorXYToGvec(
                    dpts,
                    self.rmat, rmat_s,
                    self.tvec, tvec_s, tvec_c,
                    beamVec=beam_vec)
                tth_eta = np.vstack(tth_eta).T

                # warp measured points
                if self.distortion is not None:
                    if len(self.distortion) == 2:
                        dpts = self.distortion[0](
                            dpts, self.distortion[1],
                            invert=True)
                    else:
                        raise(RuntimeError,
                              "something is wrong with the distortion")

                # plane spacings and energies
                dsp = 1. / rowNorm(gvec_s_str[:, canIntersect].T)
                wlen = 2*dsp*np.sin(0.5*tth_eta[:, 0])

                # clip to detector panel
                _, on_panel = self.clip_to_panel(dpts, buffer_edges=True)

                if multipleEnergyRanges:
                    validEnergy = np.zeros(len(wlen), dtype=bool)
                    for i in range(len(lmin)):
                        in_energy_range = np.logical_and(
                                wlen >= lmin[i],
                                wlen <= lmax[i])
                        validEnergy = validEnergy | in_energy_range
                        pass
                else:
                    validEnergy = np.logical_and(wlen >= lmin, wlen <= lmax)
                    pass

                # index for valid reflections
                keepers = np.where(np.logical_and(on_panel, validEnergy))[0]

                # assign output arrays
                xy_det[iG][keepers, :] = dpts[keepers, :]
                hkls_in[iG][:, keepers] = dhkl[:, keepers]
                angles[iG][keepers, :] = tth_eta[keepers, :]
                dspacing[iG, keepers] = dsp[keepers]
                energy[iG, keepers] = ct.keVToAngstrom(wlen[keepers])
                pass    # close conditional on valids
            pass    # close loop on grains
        return xy_det, hkls_in, angles, dspacing, energy


# =============================================================================
# UTILITIES
# =============================================================================


class PatchDataWriter(object):
    """Class for dumping Bragg reflection data."""

    def __init__(self, filename):
        self._delim = '  '
        header_items = (
            '# ID', 'PID',
            'H', 'K', 'L',
            'sum(int)', 'max(int)',
            'pred tth', 'pred eta', 'pred ome',
            'meas tth', 'meas eta', 'meas ome',
            'pred X', 'pred Y',
            'meas X', 'meas Y'
        )
        self._header = self._delim.join([
            self._delim.join(np.tile('{:<6}', 5)).format(*header_items[:5]),
            self._delim.join(np.tile('{:<12}', 2)).format(*header_items[5:7]),
            self._delim.join(np.tile('{:<23}', 10)).format(*header_items[7:17])
        ])
        if isinstance(filename, IOBase):
            self.fid = filename
        else:
            self.fid = open(filename, 'w')
        print(self._header, file=self.fid)

    def __del__(self):
        self.close()

    def close(self):
        self.fid.close()

    def dump_patch(self, peak_id, hkl_id,
                   hkl, spot_int, max_int,
                   pangs, mangs, pxy, mxy):
        """
        !!! maybe need to check that last four inputs are arrays
        """
        if mangs is None:
            spot_int = np.nan
            max_int = np.nan
            mangs = np.nan*np.ones(3)
            mxy = np.nan*np.ones(2)

        res = [int(peak_id), int(hkl_id)] \
            + np.array(hkl, dtype=int).tolist() \
            + [spot_int, max_int] \
            + pangs.tolist() \
            + mangs.tolist() \
            + pxy.tolist() \
            + mxy.tolist()

        output_str = self._delim.join(
            [self._delim.join(np.tile('{:<6d}', 5)).format(*res[:5]),
             self._delim.join(np.tile('{:<12e}', 2)).format(*res[5:7]),
             self._delim.join(np.tile('{:<23.16e}', 10)).format(*res[7:])]
        )
        print(output_str, file=self.fid)
        return output_str


class GrainDataWriter(object):
    """Class for dumping grain data."""

    def __init__(self, filename=None, array=None):
        """Writes to either file or np array

        Array must be initialized with number of rows to be written.
        """
        if filename is None and array is None:
            raise RuntimeError(
                'GrainDataWriter must be specified with filename or array')

        self.array = None
        self.fid = None

        # array supersedes filename
        if array is not None:
            assert array.shape[1] == 21, \
                f'grain data table must have 21 columns not {array.shape[21]}'
            self.array = array
            self._array_row = 0
            return

        self._delim = '  '
        header_items = (
            '# grain ID', 'completeness', 'chi^2',
            'exp_map_c[0]', 'exp_map_c[1]', 'exp_map_c[2]',
            't_vec_c[0]', 't_vec_c[1]', 't_vec_c[2]',
            'inv(V_s)[0,0]', 'inv(V_s)[1,1]', 'inv(V_s)[2,2]',
            'inv(V_s)[1,2]*sqrt(2)',
            'inv(V_s)[0,2]*sqrt(2)',
            'inv(V_s)[0,2]*sqrt(2)',
            'ln(V_s)[0,0]', 'ln(V_s)[1,1]', 'ln(V_s)[2,2]',
            'ln(V_s)[1,2]', 'ln(V_s)[0,2]', 'ln(V_s)[0,1]'
        )
        self._header = self._delim.join(
            [self._delim.join(
                np.tile('{:<12}', 3)
                ).format(*header_items[:3]),
             self._delim.join(
                np.tile('{:<23}', len(header_items) - 3)
                ).format(*header_items[3:])]
        )
        if isinstance(filename, IOBase):
            self.fid = filename
        else:
            self.fid = open(filename, 'w')
        print(self._header, file=self.fid)

    def __del__(self):
        self.close()

    def close(self):
        if self.fid is not None:
            self.fid.close()

    def dump_grain(self, grain_id, completeness, chisq,
                   grain_params):
        assert len(grain_params) == 12, \
            "len(grain_params) must be 12, not %d" % len(grain_params)

        # extract strain
        emat = logm(np.linalg.inv(mutil.vecMVToSymm(grain_params[6:])))
        evec = mutil.symmToVecMV(emat, scale=False)

        res = [int(grain_id), completeness, chisq] \
            + grain_params.tolist() \
            + evec.tolist()

        if self.array is not None:
            row = self._array_row
            assert row < self.array.shape[0], \
                f'invalid row {row} in array table'
            self.array[row] = res
            self._array_row += 1
            return res

        # (else) format and write to file
        output_str = self._delim.join(
            [self._delim.join(
                ['{:<12d}', '{:<12f}', '{:<12e}']
             ).format(*res[:3]),
             self._delim.join(
                np.tile('{:<23.16e}', len(res) - 3)
             ).format(*res[3:])]
        )
        print(output_str, file=self.fid)
        return output_str


class GrainDataWriter_h5(object):
    """Class for dumping grain results to an HDF5 archive.

    TODO: add material spec
    """

    def __init__(self, filename, instr_cfg, grain_params, use_attr=False):
        if isinstance(filename, h5py.File):
            self.fid = filename
        else:
            self.fid = h5py.File(filename + ".hdf5", "w")
        icfg = dict(instr_cfg)

        # add instrument groups and attributes
        self.instr_grp = self.fid.create_group('instrument')
        unwrap_dict_to_h5(self.instr_grp, icfg, asattr=use_attr)

        # add grain group
        self.grain_grp = self.fid.create_group('grain')
        rmat_c = makeRotMatOfExpMap(grain_params[:3])
        tvec_c = np.array(grain_params[3:6]).flatten()
        vinv_s = np.array(grain_params[6:]).flatten()
        vmat_s = np.linalg.inv(mutil.vecMVToSymm(vinv_s))

        if use_attr:    # attribute version
            self.grain_grp.attrs.create('rmat_c', rmat_c)
            self.grain_grp.attrs.create('tvec_c', tvec_c)
            self.grain_grp.attrs.create('inv(V)_s', vinv_s)
            self.grain_grp.attrs.create('vmat_s', vmat_s)
        else:    # dataset version
            self.grain_grp.create_dataset('rmat_c', data=rmat_c)
            self.grain_grp.create_dataset('tvec_c', data=tvec_c)
            self.grain_grp.create_dataset('inv(V)_s', data=vinv_s)
            self.grain_grp.create_dataset('vmat_s', data=vmat_s)

        data_key = 'reflection_data'
        self.data_grp = self.fid.create_group(data_key)

        for det_key in self.instr_grp['detectors'].keys():
            self.data_grp.create_group(det_key)

    # FIXME: throws exception when called after close method
    # def __del__(self):
    #    self.close()

    def close(self):
        self.fid.close()

    def dump_patch(self, panel_id,
                   i_refl, peak_id, hkl_id, hkl,
                   tth_edges, eta_edges, ome_centers,
                   xy_centers, ijs, frame_indices,
                   spot_data, pangs, pxy, mangs, mxy, gzip=1):
        """
        to be called inside loop over patches

        default GZIP level for data arrays is 1
        """
        fi = np.array(frame_indices, dtype=int)

        panel_grp = self.data_grp[panel_id]
        spot_grp = panel_grp.create_group("spot_%05d" % i_refl)
        spot_grp.attrs.create('peak_id', int(peak_id))
        spot_grp.attrs.create('hkl_id', int(hkl_id))
        spot_grp.attrs.create('hkl', np.array(hkl, dtype=int))
        spot_grp.attrs.create('predicted_angles', pangs)
        spot_grp.attrs.create('predicted_xy', pxy)
        if mangs is None:
            mangs = np.nan*np.ones(3)
        spot_grp.attrs.create('measured_angles', mangs)
        if mxy is None:
            mxy = np.nan*np.ones(3)
        spot_grp.attrs.create('measured_xy', mxy)

        # get centers crds from edge arrays
        # FIXME: export full coordinate arrays, or just center vectors???
        #
        # ome_crd, eta_crd, tth_crd = np.meshgrid(
        #     ome_centers,
        #     centers_of_edge_vec(eta_edges),
        #     centers_of_edge_vec(tth_edges),
        #     indexing='ij')
        #
        # ome_dim, eta_dim, tth_dim = spot_data.shape

        # !!! for now just exporting center vectors for spot_data
        tth_crd = centers_of_edge_vec(tth_edges)
        eta_crd = centers_of_edge_vec(eta_edges)

        shuffle_data = True  # reduces size by 20%
        spot_grp.create_dataset('tth_crd', data=tth_crd,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('eta_crd', data=eta_crd,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('ome_crd', data=ome_centers,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('xy_centers', data=xy_centers,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('ij_centers', data=ijs,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('frame_indices', data=fi,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('intensities', data=spot_data,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        return


def unwrap_dict_to_h5(grp, d, asattr=True):
    while len(d) > 0:
        key, item = d.popitem()
        if isinstance(item, dict):
            subgrp = grp.create_group(key)
            unwrap_dict_to_h5(subgrp, item)
        else:
            if asattr:
                grp.attrs.create(key, item)
            else:
                grp.create_dataset(key, data=np.atleast_1d(item))


class GenerateEtaOmeMaps(object):
    """
    eta-ome map class derived from new image_series and YAML config

    ...for now...

    must provide:

    self.dataStore
    self.planeData
    self.iHKLList
    self.etaEdges # IN RADIANS
    self.omeEdges # IN RADIANS
    self.etas     # IN RADIANS
    self.omegas   # IN RADIANS

    """

    def __init__(self, image_series_dict, instrument, plane_data,
                 active_hkls=None, eta_step=0.25, threshold=None,
                 ome_period=(0, 360)):
        """
        image_series must be OmegaImageSeries class
        instrument_params must be a dict (loaded from yaml spec)
        active_hkls must be a list (required for now)
        """

        self._planeData = plane_data

        # ???: change name of iHKLList?
        # ???: can we change the behavior of iHKLList?
        if active_hkls is None:
            n_rings = len(plane_data.getTTh())
            self._iHKLList = range(n_rings)
        else:
            self._iHKLList = active_hkls
            n_rings = len(active_hkls)

        # ???: need to pass a threshold?
        eta_mapping, etas = instrument.extract_polar_maps(
            plane_data, image_series_dict,
            active_hkls=active_hkls, threshold=threshold,
            tth_tol=None, eta_tol=eta_step)

        # grab a det key
        # WARNING: this process assumes that the imageseries for all panels
        # have the same length and omegas
        det_key = list(eta_mapping.keys())[0]
        data_store = []
        for i_ring in range(n_rings):
            full_map = np.zeros_like(eta_mapping[det_key][i_ring])
            nan_mask_full = np.zeros(
                (len(eta_mapping), full_map.shape[0], full_map.shape[1])
            )
            i_p = 0
            for det_key, eta_map in eta_mapping.items():
                nan_mask = ~np.isnan(eta_map[i_ring])
                nan_mask_full[i_p] = nan_mask
                full_map[nan_mask] += eta_map[i_ring][nan_mask]
                i_p += 1
            re_nan_these = np.sum(nan_mask_full, axis=0) == 0
            full_map[re_nan_these] = np.nan
            data_store.append(full_map)
        self._dataStore = data_store

        # handle omegas
        omegas_array = image_series_dict[det_key].metadata['omega']
        self._omegas = mapAngle(
            np.radians(np.average(omegas_array, axis=1)),
            np.radians(ome_period)
        )
        self._omeEdges = mapAngle(
            np.radians(np.r_[omegas_array[:, 0], omegas_array[-1, 1]]),
            np.radians(ome_period)
        )

        # !!! must avoid the case where omeEdges[0] = omeEdges[-1] for the
        # indexer to work properly
        if abs(self._omeEdges[0] - self._omeEdges[-1]) <= ct.sqrt_epsf:
            # !!! SIGNED delta ome
            del_ome = np.radians(omegas_array[0, 1] - omegas_array[0, 0])
            self._omeEdges[-1] = self._omeEdges[-2] + del_ome

        # handle etas
        # WARNING: unlinke the omegas in imageseries metadata,
        # these are in RADIANS and represent bin centers
        self._etaEdges = etas
        self._etas = self._etaEdges[:-1] + 0.5*np.radians(eta_step)

    @property
    def dataStore(self):
        return self._dataStore

    @property
    def planeData(self):
        return self._planeData

    @property
    def iHKLList(self):
        return np.atleast_1d(self._iHKLList).flatten()

    @property
    def etaEdges(self):
        return self._etaEdges

    @property
    def omeEdges(self):
        return self._omeEdges

    @property
    def etas(self):
        return self._etas

    @property
    def omegas(self):
        return self._omegas

    def save(self, filename):
        xrdutil.EtaOmeMaps.save_eta_ome_maps(self, filename)
    pass  # end of class: GenerateEtaOmeMaps
