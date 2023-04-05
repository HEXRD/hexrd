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
import copy
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

from tqdm import tqdm

import yaml

import h5py

import numpy as np

from io import IOBase

from scipy import ndimage
from scipy.linalg import logm
from skimage.measure import regionprops

from hexrd import constants
from hexrd.imageseries import ImageSeries
from hexrd.imageseries.process import ProcessedImageSeries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd.fitting.utils import fit_ring
from hexrd.gridutil import make_tolerance_grid
from hexrd import matrixutil as mutil
from hexrd.transforms.xfcapi import (
    anglesToGVec,
    angularDifference,
    gvecToDetectorXY,
    makeOscillRotMat,
    makeRotMatOfExpMap,
    mapAngle,
    unitRowVector,
)
from hexrd import xrdutil
from hexrd.crystallography import PlaneData
from hexrd import constants as ct
from hexrd.rotations import angleAxisOfRotMat, RotMatEuler
from hexrd import distortion as distortion_pkg
from hexrd.utils.compatibility import h5py_read_string
from hexrd.utils.concurrent import distribute_tasks
from hexrd.valunits import valWUnit
from hexrd.wppf import LeBail

from .cylindrical_detector import CylindricalDetector
from .detector import (
    beam_energy_DFLT,
    max_workers_DFLT,
)
from .planar_detector import PlanarDetector

from skimage.draw import polygon
from skimage.util import random_noise
from hexrd.wppf import wppfsupport

try:
    from fast_histogram import histogram1d
    fast_histogram = True
except ImportError:
    from numpy import histogram as histogram1d
    fast_histogram = False

logger = logging.getLogger()
logger.setLevel('INFO')

# =============================================================================
# PARAMETERS
# =============================================================================

instrument_name_DFLT = 'instrument'

beam_vec_DFLT = ct.beam_vec
source_distance_DFLT = np.inf

eta_vec_DFLT = ct.eta_vec

panel_id_DFLT = 'generic'
nrows_DFLT = 2048
ncols_DFLT = 2048
pixel_size_DFLT = (0.2, 0.2)

tilt_params_DFLT = np.zeros(3)
t_vec_d_DFLT = np.r_[0., 0., -1000.]

chi_DFLT = 0.
t_vec_s_DFLT = np.zeros(3)

multi_ims_key = ct.shared_ims_key
ims_classes = (ImageSeries, ProcessedImageSeries, OmegaImageSeries)

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

buffer_key = 'buffer'
distortion_key = 'distortion'

# =============================================================================
# UTILITY METHODS
# =============================================================================


def generate_chunks(nrows, ncols, base_nrows, base_ncols,
                    row_gap=0, col_gap=0):
    """
    Generate chunking data for regularly tiled composite detectors.

    Parameters
    ----------
    nrows : int
        DESCRIPTION.
    ncols : int
        DESCRIPTION.
    base_nrows : int
        DESCRIPTION.
    base_ncols : int
        DESCRIPTION.
    row_gap : int, optional
        DESCRIPTION. The default is 0.
    col_gap : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    rects : array_like
        The (nrows*ncols, ) list of ROI specs (see Notes).
    labels : array_like
        The (nrows*ncols, ) list of ROI (i, j) matrix indexing labels 'i_j'.

    Notes
    -----
    ProcessedImageSeries needs a (2, 2) array for the 'rect' kwarg:
        [[row_start, row_stop],
         [col_start, col_stop]]
    """
    row_starts = np.array([i*(base_nrows + row_gap) for i in range(nrows)])
    col_starts = np.array([i*(base_ncols + col_gap) for i in range(ncols)])
    rr = np.vstack([row_starts, row_starts + base_nrows])
    cc = np.vstack([col_starts, col_starts + base_ncols])
    rects = []
    labels = []
    for i in range(nrows):
        for j in range(ncols):
            this_rect = np.array(
                [[rr[0, i], rr[1, i]],
                 [cc[0, j], cc[1, j]]]
            )
            rects.append(this_rect)
            labels.append('%d_%d' % (i, j))
    return rects, labels


def chunk_instrument(instr, rects, labels, use_roi=False):
    """
    Generate chunked config fro regularly tiled composite detectors.

    Parameters
    ----------
    instr : TYPE
        DESCRIPTION.
    rects : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    new_icfg_dict : TYPE
        DESCRIPTION.

    """
    icfg_dict = instr.write_config()
    new_icfg_dict = dict(beam=icfg_dict['beam'],
                         oscillation_stage=icfg_dict['oscillation_stage'],
                         detectors={})
    for panel_id, panel in instr.detectors.items():
        pcfg_dict = panel.config_dict(instr.chi, instr.tvec)['detector']

        for pnum, pdata in enumerate(zip(rects, labels)):
            rect, label = pdata
            panel_name = f'{panel_id}_{label}'

            row_col_dim = np.diff(rect)  # (2, 1)
            shape = tuple(row_col_dim.flatten())
            center = (rect[:, 0].reshape(2, 1) + 0.5*row_col_dim)

            sp_tvec = np.concatenate(
                [panel.pixelToCart(center.T).flatten(), np.zeros(1)]
            )

            tvec = np.dot(panel.rmat, sp_tvec) + panel.tvec

            # new config dict
            tmp_cfg = copy.deepcopy(pcfg_dict)

            # fix sizes
            tmp_cfg['pixels']['rows'] = shape[0]
            tmp_cfg['pixels']['columns'] = shape[1]
            if use_roi:
                tmp_cfg['pixels']['roi'] = (rect[0][0], rect[1][0])

            # update tvec
            tmp_cfg['transform']['translation'] = tvec.tolist()

            new_icfg_dict['detectors'][panel_name] = copy.deepcopy(tmp_cfg)

            if panel.panel_buffer.ndim == 2:  # have a mask array!
                submask = panel.panel_buffer[
                    rect[0, 0]:rect[0, 1], rect[1, 0]:rect[1, 1]
                ]
                '''
                submask[:m, :] = False
                submask[-m:, :] = False
                submask[:, :m] = False
                submask[:, -m:] = False
                '''
                new_icfg_dict['detectors'][panel_name]['buffer'] = submask
    return new_icfg_dict


def _parse_imgser_dict(imgser_dict, det_key, roi=None):
    """
    Associates a dict of imageseries to the target panel(s).

    Parameters
    ----------
    imgser_dict : dict
        The input dict of imageseries.  Either `det_key` is in imgser_dict, or
        the shared key is.  Entries can be an ImageSeries object or a 2- or 3-d
        ndarray of images.
    det_key : str
        The target detector key.
    roi : tuple or None, optional
        The roi of the target images.  Format is
            ((row_start, row_stop), (col_start, col_stop))
        The stops are used in the normal sense of a slice. The default is None.

    Raises
    ------
    RuntimeError
        If niether `det_key` nor the shared key is in the input imgser_dict;
        Also, if the shared key is specified but the roi is None.

    Returns
    -------
    ims : hexrd.imageseries
        The desired imageseries object.

    """
    # grab imageseries for this detector
    try:
        ims = imgser_dict[det_key]
    except KeyError:
        matched_det_keys = [det_key in k for k in imgser_dict]
        if multi_ims_key in imgser_dict:
            images_in = imgser_dict[multi_ims_key]
        elif np.any(matched_det_keys):
            if sum(matched_det_keys) != 1:
                raise RuntimeError(
                    f"multiple entries found for '{det_key}'"
                )
            # use boolean array to index the proper key
            # !!! these should be in the same order
            img_keys = img_keys = np.asarray(list(imgser_dict.keys()))
            matched_det_key = img_keys[matched_det_keys][0]  # !!! only one
            images_in = imgser_dict[matched_det_key]
        else:
            raise RuntimeError(
                f"neither '{det_key}' nor '{multi_ims_key}' found"
                + 'in imageseries input'
            )

        # have images now
        if roi is None:
            raise RuntimeError(
                "roi must be specified to use shared imageseries"
            )

        if isinstance(images_in, ims_classes):
            # input is an imageseries of some kind
            ims = ProcessedImageSeries(images_in, [('rectangle', roi), ])
            if isinstance(images_in, OmegaImageSeries):
                # if it was an OmegaImageSeries, must re-cast
                ims = OmegaImageSeries(ims)
        elif isinstance(images_in, np.ndarray):
            # 2- or 3-d array of images
            ndim = images_in.ndim
            if ndim == 2:
                ims = images_in[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            elif ndim == 3:
                nrows = roi[0][1] - roi[0][0]
                ncols = roi[1][1] - roi[1][0]
                n_images = len(images_in)
                ims = np.empty((n_images, nrows, ncols),
                               dtype=images_in.dtype)
                for i, image in images_in:
                    ims[i, :, :] = \
                        images_in[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
            else:
                raise RuntimeError(
                    f"image input dim must be 2 or 3; you gave {ndim}"
                )
    return ims


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
    assert np.asarray(edges).ndim == 1, "edges must be 1-d"
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


def pixel_resolution(instr):
    """
    Return the minimum, median, and maximum angular
    resolution of the instrument.

    Parameters
    ----------
    instr : HEDMInstrument instance
        An instrument.

    Returns
    -------
    tth_stats : float
        min/median/max tth resolution in radians.
    eta_stats : TYPE
        min/median/max eta resolution in radians.

    """
    max_tth = np.inf
    max_eta = np.inf
    min_tth = -np.inf
    min_eta = -np.inf
    ang_ps_full = []
    for panel in instr.detectors.values():
        angps = panel.angularPixelSize(
            np.stack(
                panel.pixel_coords,
                axis=0
            ).reshape(2, np.cumprod(panel.shape)[-1]).T
        )
        ang_ps_full.append(angps)
        max_tth = min(max_tth, np.min(angps[:, 0]))
        max_eta = min(max_eta, np.min(angps[:, 1]))
        min_tth = max(min_tth, np.max(angps[:, 0]))
        min_eta = max(min_eta, np.max(angps[:, 1]))
        pass
    med_tth, med_eta = np.median(np.vstack(ang_ps_full), axis=0).flatten()
    return (min_tth, med_tth, max_tth), (min_eta, med_eta, max_eta)


def max_resolution(instr):
    """
    Return the maximum angular resolution of the instrument.

    Parameters
    ----------
    instr : HEDMInstrument instance
        An instrument.

    Returns
    -------
    max_tth : float
        Maximum tth resolution in radians.
    max_eta : TYPE
        maximum eta resolution in radians.

    """
    max_tth = np.inf
    max_eta = np.inf
    for panel in instr.detectors.values():
        angps = panel.angularPixelSize(
            np.stack(
                panel.pixel_coords,
                axis=0
            ).reshape(2, np.cumprod(panel.shape)[-1]).T
        )
        max_tth = min(max_tth, np.min(angps[:, 0]))
        max_eta = min(max_eta, np.min(angps[:, 1]))
    return max_tth, max_eta


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
                 instrument_name=None, tilt_calibration_mapping=None,
                 max_workers=max_workers_DFLT):
        self._id = instrument_name_DFLT

        self._source_distance = source_distance_DFLT

        if eta_vector is None:
            self._eta_vector = eta_vec_DFLT
        else:
            self._eta_vector = eta_vector

        self.max_workers = max_workers

        if instrument_config is None:
            # Default instrument
            if instrument_name is not None:
                self._id = instrument_name
            self._num_panels = 1
            self._beam_energy = beam_energy_DFLT
            self._beam_vector = beam_vec_DFLT

            # FIXME: must add cylindrical
            self._detectors = dict(
                panel_id_DFLT=PlanarDetector(
                    rows=nrows_DFLT, cols=ncols_DFLT,
                    pixel_size=pixel_size_DFLT,
                    tvec=t_vec_d_DFLT,
                    tilt=tilt_params_DFLT,
                    bvec=self._beam_vector,
                    xrs_dist=self._source_distance,
                    evec=self._eta_vector,
                    distortion=None,
                    roi=None,
                    max_workers=self.max_workers),
                )

            self._tvec = t_vec_s_DFLT
            self._chi = chi_DFLT
        else:
            if isinstance(instrument_config, h5py.File):
                tmp = {}
                unwrap_h5_to_dict(instrument_config, tmp)
                instrument_config = tmp['instrument']
            elif not isinstance(instrument_config, dict):
                raise RuntimeError(
                    "instrument_config must be either an HDF5 file object"
                    + "or a dictionary.  You gave a %s"
                    % type(instrument_config)
                )
            if instrument_name is None:
                if 'id' in instrument_config:
                    self._id = instrument_config['id']
            else:
                self._id = instrument_name

            self._num_panels = len(instrument_config['detectors'])

            xrs_config = instrument_config['beam']
            self._beam_energy = xrs_config['energy']  # keV
            self._beam_vector = calc_beam_vec(
                xrs_config['vector']['azimuth'],
                xrs_config['vector']['polar_angle'],
                )

            if 'source_distance' in xrs_config:
                xrsd = xrs_config['source_distance']
                assert np.isscalar(xrsd), \
                    "'source_distance' must be a scalar"
                self._source_distance = xrsd

            # now build detector dict
            detectors_config = instrument_config['detectors']
            det_dict = dict.fromkeys(detectors_config)
            for det_id, det_info in detectors_config.items():
                pixel_info = det_info['pixels']
                affine_info = det_info['transform']
                detector_type = det_info.get('detector_type', 'planar')
                try:
                    saturation_level = det_info['saturation_level']
                except KeyError:
                    saturation_level = 2**16
                shape = (pixel_info['rows'], pixel_info['columns'])

                panel_buffer = None
                if buffer_key in det_info:
                    det_buffer = det_info[buffer_key]
                    if det_buffer is not None:
                        if isinstance(det_buffer, np.ndarray):
                            if det_buffer.ndim == 2:
                                assert det_buffer.shape == shape, \
                                    "buffer shape must match detector"
                            else:
                                assert len(det_buffer) == 2
                            panel_buffer = det_buffer
                        elif isinstance(det_buffer, list):
                            panel_buffer = np.asarray(det_buffer)
                        elif np.isscalar(det_buffer):
                            panel_buffer = det_buffer*np.ones(2)
                        else:
                            raise RuntimeError(
                                "panel buffer spec invalid for %s" % det_id
                            )

                # optional roi
                roi = pixel_info.get('roi')

                # handle distortion
                distortion = None
                if distortion_key in det_info:
                    distortion_cfg = det_info[distortion_key]
                    if distortion_cfg is not None:
                        try:
                            func_name = distortion_cfg['function_name']
                            dparams = distortion_cfg['parameters']
                            distortion = distortion_pkg.get_mapping(
                                func_name, dparams
                            )
                        except KeyError:
                            raise RuntimeError(
                                "problem with distortion specification"
                            )
                if detector_type.lower() not in DETECTOR_TYPES:
                    msg = f'Unknown detector type: {detector_type}'
                    raise NotImplementedError(msg)

                DetectorClass = DETECTOR_TYPES[detector_type.lower()]

                det_dict[det_id] = DetectorClass(
                        name=det_id,
                        rows=pixel_info['rows'],
                        cols=pixel_info['columns'],
                        pixel_size=pixel_info['size'],
                        panel_buffer=panel_buffer,
                        saturation_level=saturation_level,
                        tvec=affine_info['translation'],
                        tilt=affine_info['tilt'],
                        bvec=self._beam_vector,
                        xrs_dist=self._source_distance,
                        evec=self._eta_vector,
                        distortion=distortion,
                        roi=roi,
                        max_workers=self.max_workers)

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

        self.update_memoization_sizes()

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
                beam_vector=self.beam_vector,
                style='hdf5'
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

        # reset on all detectors
        for panel in self.detectors.values():
            panel.bvec = self._beam_vector

    @property
    def source_distance(self):
        return self._source_distance

    @source_distance.setter
    def source_distance(self, x):
        assert np.isscalar(x), \
            f"'source_distance' must be a scalar; you input '{x}'"
        self._source_distance = x

        # reset on all detectors
        for panel in self.detectors.values():
            panel.xrs_dist = self._source_distance

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
                npp += len(panel.distortion.params)
            panel.calibration_flags = x[ii:ii + npp]
        self._calibration_flags = x

    # =========================================================================
    # METHODS
    # =========================================================================

    def write_config(self, file=None, style='yaml', calibration_dict={}):
        """ WRITE OUT YAML FILE """
        # initialize output dictionary
        assert style.lower() in ['yaml', 'hdf5'], \
            "style must be either 'yaml', or 'hdf5'; you gave '%s'" % style

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
        if self.source_distance is not None:
            beam['source_distance'] = self.source_distance

        par_dict['beam'] = beam

        if calibration_dict:
            par_dict['calibration_crystal'] = calibration_dict

        ostage = dict(
            chi=self.chi,
            translation=self.tvec.tolist()
        )
        par_dict['oscillation_stage'] = ostage

        det_dict = dict.fromkeys(self.detectors)
        for det_name, detector in self.detectors.items():
            # grab panel config
            # !!! don't need beam or tvec
            # !!! have vetted style
            pdict = detector.config_dict(chi=self.chi, tvec=self.tvec,
                                         beam_energy=self.beam_energy,
                                         beam_vector=self.beam_vector,
                                         style=style)
            det_dict[det_name] = pdict['detector']
        par_dict['detectors'] = det_dict

        # handle output file if requested
        if file is not None:
            if style.lower() == 'yaml':
                with open(file, 'w') as f:
                    yaml.dump(par_dict, stream=f)
            else:
                def _write_group(file):
                    instr_grp = file.create_group('instrument')
                    unwrap_dict_to_h5(instr_grp, par_dict, asattr=False)

                # hdf5
                if isinstance(file, str):
                    with h5py.File(file, 'w') as f:
                        _write_group(f)
                elif isinstance(file, h5py.File):
                    _write_group(file)
                else:
                    raise TypeError("Unexpected file type.")

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
                    try:
                        detector.distortion.params = p[ii:ii + dpnp]
                    except AssertionError:
                        raise RuntimeError(
                            "distortion for '%s' " % det_name
                            + "expects %d params but got %d"
                            % (len(detector.distortion.params), dpnp)
                        )
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
        !!!: images must be non-negative!
        !!!: plane_data is NOT a copy!
        """
        if tth_tol is not None:
            plane_data.tThWidth = np.radians(tth_tol)
        else:
            tth_tol = np.degrees(plane_data.tThWidth)

        # make rings clipped to panel
        # !!! eta_idx has the same length as plane_data.exclusions
        #     each entry are the integer indices into the bins
        # !!! eta_edges is the list of eta bin EDGES; same for all
        #     detectors, so calculate it once
        # !!! grab first panel
        panel = next(iter(self.detectors.values()))
        pow_angs, pow_xys, tth_ranges, eta_idx, eta_edges = \
            panel.make_powder_rings(
                plane_data, merge_hkls=False,
                delta_eta=eta_tol, full_output=True
            )

        if active_hkls is not None:
            assert hasattr(active_hkls, '__len__'), \
                "active_hkls must be an iterable with __len__"

            # need to re-cast for element-wise operations
            active_hkls = np.array(active_hkls)

            # these are all active reflection unique hklIDs
            active_hklIDs = plane_data.getHKLID(
                plane_data.hkls, master=True
            )

            # find indices
            idx = np.zeros_like(active_hkls, dtype=int)
            for i, input_hklID in enumerate(active_hkls):
                try:
                    idx[i] = np.where(active_hklIDs == input_hklID)[0]
                except ValueError:
                    raise RuntimeError(f"hklID '{input_hklID}' is invalid")
            tth_ranges = tth_ranges[idx]
            pass  # end of active_hkls handling

        delta_eta = eta_edges[1] - eta_edges[0]
        ncols_eta = len(eta_edges) - 1

        ring_maps_panel = dict.fromkeys(self.detectors)
        for i_d, det_key in enumerate(self.detectors):
            print("working on detector '%s'..." % det_key)

            # grab panel
            panel = self.detectors[det_key]
            # native_area = panel.pixel_area  # pixel ref area

            # pixel angular coords for the detector panel
            ptth, peta = panel.pixel_angles()

            # grab imageseries for this detector
            ims = _parse_imgser_dict(imgser_dict, det_key, roi=panel.roi)

            # grab omegas from imageseries and squawk if missing
            try:
                omegas = ims.metadata['omega']
            except KeyError:
                raise RuntimeError(
                    f"imageseries for '{det_key}' has no omega info"
                )

            # initialize maps and assing by row (omega/frame)
            nrows_ome = len(omegas)

            # init map with NaNs
            shape = (len(tth_ranges), nrows_ome, ncols_eta)
            ring_maps = np.full(shape, np.nan)

            # Generate ring parameters once, and re-use them for each image
            ring_params = []
            for tthr in tth_ranges:
                kwargs = {
                    'tthr': tthr,
                    'ptth': ptth,
                    'peta': peta,
                    'eta_edges': eta_edges,
                    'delta_eta': delta_eta,
                }
                ring_params.append(_generate_ring_params(**kwargs))

            # Divide up the images among processes
            tasks = distribute_tasks(len(ims), self.max_workers)
            func = partial(_run_histograms, ims=ims, tth_ranges=tth_ranges,
                           ring_maps=ring_maps, ring_params=ring_params,
                           threshold=threshold)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(func, tasks)

            ring_maps_panel[det_key] = ring_maps

        return ring_maps_panel, eta_edges

    def extract_line_positions(self, plane_data, imgser_dict,
                               tth_tol=None, eta_tol=1., npdiv=2,
                               eta_centers=None,
                               collapse_eta=True, collapse_tth=False,
                               do_interpolation=True, do_fitting=False,
                               tth_distortion=None, fitting_kwargs=None):
        """
        Perform annular interpolation on diffraction images.

        Provides data for extracting the line positions from powder diffraction
        images, pole figure patches from imageseries, or Bragg peaks from
        Laue diffraction images.

        Parameters
        ----------
        plane_data : hexrd.crystallography.PlaneData object or array_like
            Object determining the 2theta positions for the integration
            sectors.  If PlaneData, this will be all non-excluded reflections,
            subject to merging within PlaneData.tThWidth.  If array_like,
            interpreted as a list of 2theta angles IN DEGREES.
        imgser_dict : dict
            Dictionary of powder diffraction images, one for each detector.
        tth_tol : scalar, optional
            The radial (i.e. 2theta) width of the integration sectors
            IN DEGREES.  This arg is required if plane_data is array_like.
            The default is None.
        eta_tol : scalar, optional
            The azimuthal (i.e. eta) width of the integration sectors
            IN DEGREES. The default is 1.
        npdiv : int, optional
            The number of oversampling pixel subdivision (see notes).
            The default is 2.
        eta_centers : array_like, optional
            The desired azimuthal sector centers.  The default is None.  If
            None, then bins are distrubted sequentially from (-180, 180).
        collapse_eta : bool, optional
            Flag for summing sectors in eta. The default is True.
        collapse_tth : bool, optional
            Flag for summing sectors in 2theta. The default is False.
        do_interpolation : bool, optional
            If True, perform bilinear interpolation. The default is True.
        do_fitting : bool, optional
            If True, then perform spectrum fitting, and append the results
            to the returned data. collapse_eta must also be True for this
            to have any effect. The default is False.
        tth_distortion : special class, optional
            for special case of pinhole camera distortions.  See
            hexrd.xrdutil.phutil.SampleLayerDistortion (only type supported)
        fitting_kwargs : dict, optional
            kwargs passed to hexrd.fitting.utils.fit_ring if do_fitting is True

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        panel_data : dict
            Dictionary over the detctors with the following structure:
                [list over (merged) 2theta ranges]
                  [list over valid eta sectors]
                    [angle data <input dependent>,
                     bin intensities <input dependent>,
                     fitting results <input dependent>]

        Notes
        -----
        TODO: May change the array_like input units to degrees.
        TODO: rename function.

        """

        if fitting_kwargs is None:
            fitting_kwargs = {}

        # =====================================================================
        # LOOP OVER DETECTORS
        # =====================================================================
        logger.info("Interpolating ring data")
        pbar_dets = partial(tqdm, total=self.num_panels, desc="Detector",
                            position=self.num_panels)

        # Split up the workers among the detectors
        max_workers_per_detector = max(1, self.max_workers // self.num_panels)

        kwargs = {
            'plane_data': plane_data,
            'tth_tol': tth_tol,
            'eta_tol': eta_tol,
            'eta_centers': eta_centers,
            'npdiv': npdiv,
            'collapse_tth': collapse_tth,
            'collapse_eta': collapse_eta,
            'do_interpolation': do_interpolation,
            'do_fitting': do_fitting,
            'fitting_kwargs': fitting_kwargs,
            'tth_distortion': tth_distortion,
            'max_workers': max_workers_per_detector,
        }
        func = partial(_extract_detector_line_positions, **kwargs)

        def make_instr_cfg(panel):
            return panel.config_dict(
                chi=self.chi, tvec=self.tvec,
                beam_energy=self.beam_energy,
                beam_vector=self.beam_vector,
                style='hdf5'
            )

        images = []
        for detector_id, panel in self.detectors.items():
            images.append(_parse_imgser_dict(imgser_dict, detector_id,
                                             roi=panel.roi))

        panels = [self.detectors[k] for k in self.detectors]
        instr_cfgs = [make_instr_cfg(x) for x in panels]
        pbp_array = np.arange(self.num_panels)
        iter_args = zip(panels, instr_cfgs, images, pbp_array)
        with ProcessPoolExecutor(mp_context=constants.mp_context,
                                 max_workers=self.num_panels) as executor:
            results = list(pbar_dets(executor.map(func, iter_args)))

        panel_data = {}
        for det, res in zip(self.detectors, results):
            panel_data[det] = res

        return panel_data

    def simulate_powder_pattern(self,
                                mat_list,
                                params=None,
                                bkgmethod=None,
                                origin=None,
                                noise=None):
        """
        Generate powder diffraction iamges from specified materials.

        Parameters
        ----------
        mat_list : array_like (n, )
            List of Material classes.
        params : dict, optional
            Dictionary of LeBail parameters (see Notes). The default is None.
        bkgmethod : dict, optional
            Background function specification. The default is None.
        origin : array_like (3,), optional
            Vector describing the origin of the diffrction volume.
            The default is None, wiich is equivalent to [0, 0, 0].
        noise : str, optional
            Flag describing type of noise to be applied. The default is None.

        Returns
        -------
        img_dict : dict
            Dictionary of diffraciton images over the detectors.

        Notes
        -----
        TODO: add more controls for noise function.
        TODO: modify hooks to LeBail parameters.
        TODO: add optional volume fraction weights for phases in mat_list
        """
        """
        >> @AUTHOR:     Saransh Singh, Lanwrence Livermore National Lab,
                        saransh1@llnl.gov
        >> @DATE:       01/22/2021 SS 1.0 original
        >> @DETAILS:    adding hook to WPPF class. this changes the input list
                        significantly
        """
        if origin is None:
            origin = self.tvec
        origin = np.asarray(origin).squeeze()
        assert len(origin) == 3, \
            "origin must be a 3-element sequence"

        '''
        if params is none, fill in some sane default values
        only the first value is used. the rest of the values are
        the upper, lower bounds and vary flag for refinement which
        are not used but required for interfacing with WPPF

        zero_error : zero shift error
        U, V, W : Cagliotti parameters
        P, X, Y : Lorentzian parameters
        eta1, eta2, eta3 : Mixing parameters
        '''
        if params is None:
            # params = {'zero_error': [0.0, -1., 1., True],
            #           'U': [2e-1, -1., 1., True],
            #           'V': [2e-2, -1., 1., True],
            #           'W': [2e-2, -1., 1., True],
            #           'X': [2e-1, -1., 1., True],
            #           'Y': [2e-1, -1., 1., True]
            #           }
            params = wppfsupport._generate_default_parameters_LeBail(
                mat_list,
                1)
        '''
        use the material list to obtain the dictionary of initial intensities
        we need to make sure that the intensities are properly scaled by the
        lorentz polarization factor. since the calculation is done in the
        LeBail class, all that means is the initial intensity needs that factor
        in there
        '''
        img_dict = dict.fromkeys(self.detectors)

        # find min and max tth over all panels
        tth_mi = np.inf
        tth_ma = 0.
        ptth_dict = dict.fromkeys(self.detectors)
        for det_key, panel in self.detectors.items():
            ptth, peta = panel.pixel_angles(origin=origin)
            tth_mi = min(tth_mi, ptth.min())
            tth_ma = max(tth_ma, ptth.max())
            ptth_dict[det_key] = ptth

        '''
        now make a list of two theta and dummy ones for the experimental
        spectrum this is never really used so any values should be okay. We
        could also pas the integrated detector image if we would like to
        simulate some realistic background. But thats for another day.
        '''
        # convert angles to degrees because thats what the WPPF expects
        tth_mi = np.degrees(tth_mi)
        tth_ma = np.degrees(tth_ma)

        # get tth angular resolution for instrument
        ang_res = max_resolution(self)

        # !!! calc nsteps by oversampling
        nsteps = int(np.ceil(2*(tth_ma - tth_mi)/np.degrees(ang_res[0])))

        # evaulation vector for LeBail
        tth = np.linspace(tth_mi, tth_ma, nsteps)

        expt = np.vstack([tth, np.ones_like(tth)]).T

        wavelength = [
            valWUnit('lp', 'length', self.beam_wavelength, 'angstrom'),
            1.
        ]

        '''
        now go through the material list and get the intensity dictionary
        '''
        intensity = {}
        for mat in mat_list:

            multiplicity = mat.planeData.getMultiplicity()

            tth = mat.planeData.getTTh()

            LP = (1 + np.cos(tth)**2) / \
                np.cos(0.5*tth)/np.sin(0.5*tth)**2

            intensity[mat.name] = {}
            intensity[mat.name]['synchrotron'] = \
                mat.planeData.get_structFact() * LP * multiplicity

        kwargs = {
            'expt_spectrum': expt,
            'params': params,
            'phases': mat_list,
            'wavelength': {
                'synchrotron': wavelength
            },
            'bkgmethod': bkgmethod,
            'intensity_init': intensity,
            'peakshape': 'pvtch'
        }

        self.WPPFclass = LeBail(**kwargs)

        self.simulated_spectrum = self.WPPFclass.spectrum_sim
        self.background = self.WPPFclass.background

        '''
        now that we have the simulated intensities, its time to get the
        two theta for the detector pixels and interpolate what the intensity
        for each pixel should be
        '''

        img_dict = dict.fromkeys(self.detectors)
        for det_key, panel in self.detectors.items():
            ptth = ptth_dict[det_key]

            img = np.interp(np.degrees(ptth),
                            self.simulated_spectrum.x,
                            self.simulated_spectrum.y + self.background.y)

            if noise is None:
                img_dict[det_key] = img

            else:
                if noise.lower() == 'poisson':
                    im_noise = random_noise(img,
                                            mode='poisson',
                                            clip=True)
                    mi = im_noise.min()
                    ma = im_noise.max()
                    if ma > mi:
                        im_noise = (im_noise - mi)/(ma - mi)

                    img_dict[det_key] = im_noise

                elif noise.lower() == 'gaussian':
                    img_dict[det_key] = random_noise(img,
                                                     mode='gaussian',
                                                     clip=True)

                elif noise.lower() == 'salt':
                    img_dict[det_key] = random_noise(img, mode='salt')

                elif noise.lower() == 'pepper':
                    img_dict[det_key] = random_noise(img, mode='pepper')

                elif noise.lower() == 's&p':
                    img_dict[det_key] = random_noise(img, mode='s&p')

                elif noise.lower() == 'speckle':
                    img_dict[det_key] = random_noise(img,
                                                     mode='speckle',
                                                     clip=True)

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

        xy_det, hkls_in, angles, dspacing, energy

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
                   ome_period=None,
                   dirname='results', filename=None, output_format='text',
                   return_spot_list=False,
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
        return_spot_list : TYPE, optional
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
        if ome_period is None:
            ims = next(iter(imgser_dict.values()))
            ostart = ims.omega[0, 0]
            ome_period = np.radians(ostart + np.r_[0., 360.])

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
        for detector_id, panel in self.detectors.items():
            # initialize text-based output writer
            if filename is not None and output_format.lower() == 'text':
                output_dir = os.path.join(
                    dirname, detector_id
                    )
                os.makedirs(output_dir, exist_ok=True)
                this_filename = os.path.join(
                    output_dir, filename
                )
                writer = PatchDataWriter(this_filename)

            # grab panel
            instr_cfg = panel.config_dict(
                self.chi, self.tvec,
                beam_energy=self.beam_energy,
                beam_vector=self.beam_vector,
                style='hdf5'
            )
            native_area = panel.pixel_area  # pixel ref area

            # pull out the OmegaImageSeries for this panel from input dict
            ome_imgser = _parse_imgser_dict(imgser_dict,
                                            detector_id,
                                            roi=panel.roi)

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
                            window for (%d  %d  %d) falls outside omega range
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
                                raise RuntimeError(msg % interp)

                            # now have interpolated patch data...
                            labels, num_peaks = ndimage.label(
                                patch_data > threshold, structure=label_struct
                            )
                            slabels = np.arange(1, num_peaks + 1)

                            if num_peaks > 0:
                                peak_id = iRefl
                                props = regionprops(labels, patch_data)
                                coms = np.vstack(
                                    [x.weighted_centroid for x in props])
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
                                # FIXME: overload angles_to_cart?
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
                                    meas_xy = panel.distortion.apply_inverse(
                                        np.atleast_2d(meas_xy)
                                    ).flatten()
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

                        if return_spot_list:
                            # Full output
                            xyc_arr = xy_eval.reshape(
                                prows, pcols, 2
                            ).transpose(2, 0, 1)
                            _patch_output = [
                                detector_id, iRefl, peak_id, hkl_id, hkl,
                                tth_edges, eta_edges, np.radians(ome_eval),
                                xyc_arr, ijs, frame_indices, patch_data,
                                ang_centers[i_pt], xy_centers[i_pt],
                                meas_angs, meas_xy
                            ]
                        else:
                            # Trimmed output
                            _patch_output = [
                                peak_id, hkl_id, hkl, sum_int, max_int,
                                ang_centers[i_pt], meas_angs, meas_xy
                            ]
                        patch_output.append(_patch_output)
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

    def update_memoization_sizes(self):
        # Resize all known memoization functions to have a cache at least
        # the size of the number of detectors.
        all_panels = list(self.detectors.values())
        PlanarDetector.update_memoization_sizes(all_panels)
        CylindricalDetector.update_memoization_sizes(all_panels)


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
            'inv(V_s)[0,1]*sqrt(2)',
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


def unwrap_dict_to_h5(grp, d, asattr=False):
    """
    Unwraps a dictionary to an HDF5 file of the same structure.

    Parameters
    ----------
    grp : HDF5 group object
        The HDF5 group to recursively unwrap the dict into.
    d : dict
        Input dict (of dicts).
    asattr : bool, optional
        Flag to write end member in dictionary tree to an attribute. If False,
        if writes the object to a dataset using numpy.  The default is False.

    Returns
    -------
    None.

    """
    while len(d) > 0:
        key, item = d.popitem()
        if isinstance(item, dict):
            subgrp = grp.create_group(key)
            unwrap_dict_to_h5(subgrp, item, asattr=asattr)
        else:
            if asattr:
                try:
                    grp.attrs.create(key, item)
                except TypeError:
                    if item is None:
                        continue
                    else:
                        raise
            else:
                try:
                    grp.create_dataset(key, data=np.atleast_1d(item))
                except TypeError:
                    if item is None:
                        continue
                    else:
                        # probably a string badness
                        grp.create_dataset(key, data=item)


def unwrap_h5_to_dict(f, d):
    """
    Unwraps a simple HDF5 file to a dictionary of the same structure.

    Parameters
    ----------
    f : HDF5 file (mode r)
        The input HDF5 file object.
    d : dict
        dictionary object to update.

    Returns
    -------
    None.

    Notes
    -----
    As written, ignores attributes and uses numpy to cast HDF5 datasets to
    dict entries.  Checks for 'O' type arrays and casts to strings; also
    converts single-element arrays to scalars.
    """
    for key, val in f.items():
        try:
            d[key] = {}
            unwrap_h5_to_dict(val, d[key])
        except AttributeError:
            # reached a dataset
            if np.dtype(val) == 'O':
                d[key] = h5py_read_string(val)
            else:
                tmp = np.array(val)
                if tmp.ndim == 1 and len(tmp) == 1:
                    d[key] = tmp[0]
                else:
                    d[key] = tmp


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

        FIXME: get rid of omega period; should get it from imageseries
        """

        self._planeData = plane_data

        # ???: change name of iHKLList?
        # ???: can we change the behavior of iHKLList?
        if active_hkls is None:
            self._iHKLList = plane_data.getHKLID(
                plane_data.hkls, master=True
            )
            n_rings = len(self._iHKLList)
        else:
            assert hasattr(active_hkls, '__len__'), \
                "active_hkls must be an iterable with __len__"
            self._iHKLList = active_hkls
            n_rings = len(active_hkls)

        # grab a det key and corresponding imageseries (first will do)
        # !!! assuming that the imageseries for all panels
        #     have the same length and omegas
        det_key, this_det_ims = next(iter(image_series_dict.items()))

        # handle omegas
        # !!! for multi wedge, enforncing monotonicity
        # !!! wedges also cannot overlap or span more than 360
        omegas_array = this_det_ims.metadata['omega']  # !!! DEGREES
        delta_ome = omegas_array[0][-1] - omegas_array[0][0]
        frame_mask = None
        ome_period = omegas_array[0, 0] + np.r_[0., 360.]  # !!! be careful
        if this_det_ims.omegawedges.nwedges > 1:
            delta_omes = [(i['ostop'] - i['ostart'])/i['nsteps']
                          for i in this_det_ims.omegawedges.wedges]
            check_wedges = mutil.uniqueVectors(np.atleast_2d(delta_omes),
                                               tol=1e-6).squeeze()
            assert check_wedges.size == 1, \
                "all wedges must have the same delta omega to 1e-6"
            # grab representative delta ome
            # !!! assuming positive delta consistent with OmegaImageSeries
            delta_ome = delta_omes[0]

            # grab full-range start/stop
            # !!! be sure to map to the same period to enable arithmatic
            # ??? safer to do this way rather than just pulling from
            #     the omegas attribute?
            owedges = this_det_ims.omegawedges.wedges
            ostart = owedges[0]['ostart']  # !!! DEGREES
            ostop = float(
                mapAngle(owedges[-1]['ostop'], ome_period, units='degrees')
            )
            # compute total nsteps
            # FIXME: need check for roundoff badness
            nsteps = int((ostop - ostart)/delta_ome)
            ome_edges_full = np.linspace(
                ostart, ostop, num=nsteps+1, endpoint=True
            )
            omegas_array = np.vstack(
                [ome_edges_full[:-1], ome_edges_full[1:]]
            ).T
            ome_centers = np.average(omegas_array, axis=1)

            # use OmegaImageSeries method to determine which bins have data
            # !!! this array has -1 outside a wedge
            # !!! again assuming the valid frame order increases monotonically
            frame_mask = np.array(
                [this_det_ims.omega_to_frame(ome)[0] != -1
                 for ome in ome_centers]
            )
            pass  # end multi-wedge case

        # ???: need to pass a threshold?
        eta_mapping, etas = instrument.extract_polar_maps(
            plane_data, image_series_dict,
            active_hkls=active_hkls, threshold=threshold,
            tth_tol=None, eta_tol=eta_step)

        # for convenience grab map shape from first
        map_shape = next(iter(eta_mapping.values())).shape[1:]

        # pack all detectors with masking
        # FIXME: add omega masking
        data_store = []
        for i_ring in range(n_rings):
            # first handle etas
            full_map = np.zeros(map_shape, dtype=float)
            nan_mask_full = np.zeros(
                (len(eta_mapping), map_shape[0], map_shape[1])
            )
            i_p = 0
            for det_key, eta_map in eta_mapping.items():
                nan_mask = ~np.isnan(eta_map[i_ring])
                nan_mask_full[i_p] = nan_mask
                full_map[nan_mask] += eta_map[i_ring][nan_mask]
                i_p += 1
            re_nan_these = np.sum(nan_mask_full, axis=0) == 0
            full_map[re_nan_these] = np.nan

            # now omegas
            if frame_mask is not None:
                # !!! must expand row dimension to include
                #     skipped omegas
                tmp = np.ones((len(frame_mask), map_shape[1]))*np.nan
                tmp[frame_mask, :] = full_map
                full_map = tmp
            data_store.append(full_map)
        self._dataStore = data_store

        # set required attributes
        self._omegas = mapAngle(
            np.radians(np.average(omegas_array, axis=1)),
            np.radians(ome_period)
        )
        self._omeEdges = mapAngle(
            np.radians(np.r_[omegas_array[:, 0], omegas_array[-1, 1]]),
            np.radians(ome_period)
        )

        # !!! must avoid the case where omeEdges[0] = omeEdges[-1] for the
        #     indexer to work properly
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


def _generate_ring_params(tthr, ptth, peta, eta_edges, delta_eta):
    # mark pixels in the spec'd tth range
    pixels_in_tthr = np.logical_and(
        ptth >= tthr[0], ptth <= tthr[1]
    )

    # catch case where ring isn't on detector
    if not np.any(pixels_in_tthr):
        return None

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

    return retas, eta_bins, rtth_idx, reta_idx


def _run_histograms(rows, ims, tth_ranges, ring_maps, ring_params, threshold):
    for i_row in range(*rows):
        image = ims[i_row]

        # handle threshold if specified
        if threshold is not None:
            # !!! NaNs get preserved
            image = np.array(image)
            image[image < threshold] = 0.

        for i_r, tthr in enumerate(tth_ranges):
            this_map = ring_maps[i_r]
            params = ring_params[i_r]
            if not params:
                # We are supposed to skip this ring...
                continue

            # Unpack the params
            retas, eta_bins, rtth_idx, reta_idx = params

            if fast_histogram:
                result = histogram1d(retas, len(eta_bins) - 1,
                                     (eta_bins[0], eta_bins[-1]),
                                     weights=image[rtth_idx])
            else:
                result, _ = histogram1d(retas, bins=eta_bins,
                                        weights=image[rtth_idx])

            this_map[i_row, reta_idx] = result


def _extract_detector_line_positions(iter_args, plane_data, tth_tol,
                                     eta_tol, eta_centers, npdiv,
                                     collapse_tth, collapse_eta,
                                     do_interpolation, do_fitting,
                                     fitting_kwargs, tth_distortion,
                                     max_workers):
    panel, instr_cfg, images, pbp = iter_args

    if images.ndim == 2:
        images = np.tile(images, (1, 1, 1))
    elif images.ndim != 3:
        raise RuntimeError("images must be 2- or 3-d")

    # make rings
    # !!! adding tth_distortion pass-through; comes in as dict over panels
    tth_distr_cls = None
    if tth_distortion is not None:
        tth_distr_cls = tth_distortion[panel.name]

    pow_angs, pow_xys, tth_ranges = panel.make_powder_rings(
        plane_data, merge_hkls=True,
        delta_tth=tth_tol, delta_eta=eta_tol,
        eta_list=eta_centers, tth_distortion=tth_distr_cls)

    tth_tols = np.degrees(np.hstack([i[1] - i[0] for i in tth_ranges]))

    # !!! this is only needed if doing fitting
    if isinstance(plane_data, PlaneData):
        tth_idx, tth_ranges = plane_data.getMergedRanges(cullDupl=True)
        tth_ref = plane_data.getTTh()
        tth0 = [np.degrees(tth_ref[i]) for i in tth_idx]
    else:
        tth0 = plane_data

    # =================================================================
    # LOOP OVER RING SETS
    # =================================================================
    pbar_rings = partial(tqdm, total=len(pow_angs), desc="Ringset",
                         position=pbp)

    kwargs = {
        'instr_cfg': instr_cfg,
        'panel': panel,
        'eta_tol': eta_tol,
        'npdiv': npdiv,
        'collapse_tth': collapse_tth,
        'collapse_eta': collapse_eta,
        'images': images,
        'do_interpolation': do_interpolation,
        'do_fitting': do_fitting,
        'fitting_kwargs': fitting_kwargs,
        'tth_distortion': tth_distr_cls,
    }
    func = partial(_extract_ring_line_positions, **kwargs)
    iter_arg = zip(pow_angs, pow_xys, tth_tols, tth0)
    with ProcessPoolExecutor(mp_context=constants.mp_context,
                             max_workers=max_workers) as executor:
        return list(pbar_rings(executor.map(func, iter_arg)))


def _extract_ring_line_positions(iter_args, instr_cfg, panel, eta_tol, npdiv,
                                 collapse_tth, collapse_eta, images,
                                 do_interpolation, do_fitting, fitting_kwargs,
                                 tth_distortion):
    """
    Extracts data for a single Debye-Scherrer ring <private>.

    Parameters
    ----------
    iter_args : tuple
        (angs [radians],
         xys [mm],
         tth_tol [deg],
         this_tth0 [deg])
    instr_cfg : TYPE
        DESCRIPTION.
    panel : TYPE
        DESCRIPTION.
    eta_tol : TYPE
        DESCRIPTION.
    npdiv : TYPE
        DESCRIPTION.
    collapse_tth : TYPE
        DESCRIPTION.
    collapse_eta : TYPE
        DESCRIPTION.
    images : TYPE
        DESCRIPTION.
    do_interpolation : TYPE
        DESCRIPTION.
    do_fitting : TYPE
        DESCRIPTION.
    fitting_kwargs : TYPE
        DESCRIPTION.
    tth_distortion : TYPE
        DESCRIPTION.

    Yields
    ------
    patch_data : TYPE
        DESCRIPTION.

    """
    # points are already checked to fall on detector
    angs, xys, tth_tol, this_tth0 = iter_args

    n_images = len(images)
    native_area = panel.pixel_area

    # make the tth,eta patches for interpolation
    patches = xrdutil.make_reflection_patches(
        instr_cfg, angs, panel.angularPixelSize(xys),
        tth_tol=tth_tol, eta_tol=eta_tol, npdiv=npdiv, quiet=True)

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
        elif collapse_eta:
            # !!! yield the tth bin centers
            tth_centers = np.average(
                np.vstack(
                    [vtx_angs[0][0, :-1], vtx_angs[0][0, 1:]]
                ),
                axis=0
            )
            ang_data = (tth_centers,
                        angs[i_p][-1])
            if do_fitting:
                fit_data = []
        else:
            ang_data = vtx_angs

        prows, pcols = areas.shape
        area_fac = areas/float(native_area)

        # interpolate
        if not collapse_tth:
            ims_data = []
        for j_p in np.arange(len(images)):
            # catch interpolation type
            image = images[j_p]
            if do_interpolation:
                p_img = panel.interpolate_bilinear(
                        xy_eval,
                        image,
                    ).reshape(prows, pcols)*area_fac
            else:
                p_img = image[ijs[0], ijs[1]]*area_fac

            # catch flat spectrum data, which will cause
            # fitting to fail.
            # ???: best here, or make fitting handle it?
            mxval = np.max(p_img)
            mnval = np.min(p_img)
            if mxval == 0 or (1. - mnval/mxval) < 0.01:
                continue

            # catch collapsing options
            if collapse_tth:
                patch_data[i_p, j_p] = np.average(p_img)
                # ims_data.append(np.sum(p_img))
            else:
                if collapse_eta:
                    lineout = np.average(p_img, axis=0)
                    ims_data.append(lineout)
                    if do_fitting:
                        if tth_distortion is not None:
                            # must correct tth0
                            tmp = tth_distortion.apply(
                                panel.angles_to_cart(
                                    np.vstack(
                                        [np.radians(this_tth0),
                                         np.tile(ang_data[-1], len(this_tth0))]
                                    ).T
                                ),
                                return_nominal=True)
                            pk_centers = np.degrees(tmp[:, 0])
                        else:
                            pk_centers = this_tth0
                        kwargs = {
                            'tth_centers': np.degrees(tth_centers),
                            'lineout': lineout,
                            'tth_pred': pk_centers,
                            **fitting_kwargs,
                        }
                        result = fit_ring(**kwargs)
                        fit_data.append(result)
                else:
                    ims_data.append(p_img)
            pass  # close image loop
        if not collapse_tth:
            output = [ang_data, ims_data]
            if do_fitting:
                output.append(fit_data)
            patch_data.append(output)

    return patch_data


DETECTOR_TYPES = {
    'planar': PlanarDetector,
    'cylindrical': CylindricalDetector,
}
