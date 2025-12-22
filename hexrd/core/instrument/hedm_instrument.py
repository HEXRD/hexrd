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
from contextlib import contextmanager
import copy
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Literal, Optional, Union

from tqdm import tqdm

import yaml

import h5py

import numpy as np

from io import IOBase

from scipy import ndimage
from scipy.linalg import logm
from skimage.measure import regionprops

from hexrd.core import constants
from hexrd.core.imageseries import ImageSeries
from hexrd.core.imageseries.process import ProcessedImageSeries
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.core.fitting.utils import fit_ring
from hexrd.core.gridutil import make_tolerance_grid
from hexrd.core import matrixutil as mutil
from hexrd.core.transforms.xfcapi import (
    angles_to_gvec,
    gvec_to_xy,
    make_sample_rmat,
    make_rmat_of_expmap,
    unit_vector,
)

# TODO: Resolve extra-core-dependency
from hexrd.hedm import xrdutil
from hexrd.hed.xrdutil import _project_on_detector_plane
from hexrd.core.material.crystallography import PlaneData
from hexrd.core import constants as ct
from hexrd.core.rotations import mapAngle
from hexrd.core import distortion as distortion_pkg
from hexrd.core.utils.concurrent import distribute_tasks
from hexrd.core.utils.hdf5 import unwrap_dict_to_h5, unwrap_h5_to_dict
from hexrd.core.utils.yaml import NumpyToNativeDumper
from hexrd.core.valunits import valWUnit
from hexrd.powder.wppf import LeBail

from .cylindrical_detector import CylindricalDetector
from .detector import beam_energy_DFLT, Detector, max_workers_DFLT
from .planar_detector import PlanarDetector

from skimage.draw import polygon
from skimage.util import random_noise
from hexrd.powder.wppf import wppfsupport

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
t_vec_d_DFLT = np.r_[0.0, 0.0, -1000.0]

chi_DFLT = 0.0
t_vec_s_DFLT = np.zeros(3)

multi_ims_key = ct.shared_ims_key
ims_classes = (ImageSeries, ProcessedImageSeries, OmegaImageSeries)

buffer_key = 'buffer'
distortion_key = 'distortion'

# =============================================================================
# UTILITY METHODS
# =============================================================================


def generate_chunks(
    nrows, ncols, base_nrows, base_ncols, row_gap=0, col_gap=0
):
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
    row_starts = np.array([i * (base_nrows + row_gap) for i in range(nrows)])
    col_starts = np.array([i * (base_ncols + col_gap) for i in range(ncols)])
    rr = np.vstack([row_starts, row_starts + base_nrows])
    cc = np.vstack([col_starts, col_starts + base_ncols])
    rects = []
    labels = []
    for i in range(nrows):
        for j in range(ncols):
            this_rect = np.array([[rr[0, i], rr[1, i]], [cc[0, j], cc[1, j]]])
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
    new_icfg_dict = dict(
        beam=icfg_dict['beam'],
        oscillation_stage=icfg_dict['oscillation_stage'],
        detectors={},
    )
    for panel_id, panel in instr.detectors.items():
        pcfg_dict = panel.config_dict(instr.chi, instr.tvec)['detector']

        for pnum, pdata in enumerate(zip(rects, labels)):
            rect, label = pdata
            panel_name = f'{panel_id}_{label}'

            row_col_dim = np.diff(rect)  # (2, 1)
            shape = tuple(row_col_dim.flatten())
            center = rect[:, 0].reshape(2, 1) + 0.5 * row_col_dim

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

            if panel.panel_buffer is not None:
                if panel.panel_buffer.ndim == 2:  # have a mask array!
                    submask = panel.panel_buffer[
                        rect[0, 0] : rect[0, 1], rect[1, 0] : rect[1, 1]
                    ]
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
    ims : hexrd.core.imageseries
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
                raise RuntimeError(f"multiple entries found for '{det_key}'")
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
            ims = ProcessedImageSeries(
                images_in,
                [
                    ('rectangle', roi),
                ],
            )
            if isinstance(images_in, OmegaImageSeries):
                # if it was an OmegaImageSeries, must re-cast
                ims = OmegaImageSeries(ims)
        elif isinstance(images_in, np.ndarray):
            # 2- or 3-d array of images
            ndim = images_in.ndim
            if ndim == 2:
                ims = images_in[roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]]
            elif ndim == 3:
                nrows = roi[0][1] - roi[0][0]
                ncols = roi[1][1] - roi[1][0]
                n_images = len(images_in)
                ims = np.empty((n_images, nrows, ncols), dtype=images_in.dtype)
                for i, image in images_in:
                    ims[i, :, :] = images_in[
                        roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]
                    ]
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
        np.sin(phi) * np.cos(tht), np.cos(phi), np.sin(phi) * np.sin(tht)
    ]
    return -bv


def calc_angles_from_beam_vec(bvec):
    """
    Return the azimuth and polar angle from a beam
    vector
    """
    bvec = np.atleast_1d(bvec).flatten()
    nvec = unit_vector(-bvec)
    azim = float(np.degrees(np.arctan2(nvec[2], nvec[0])))
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
    tau = 360.0
    if units.lower() == 'radians':
        tau = 2 * np.pi
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
    if np.asarray(edges).size < 2:
        raise ValueError(
            "edges must be an array-like with at least 2 elements"
        )
    return np.average(np.vstack([edges[:-1], edges[1:]]), axis=0)


def max_tth(instr):
    """
    Return the maximum Bragg angle (in radians) subtended by the instrument.

    Parameters
    ----------
    instr : hexrd.hedm.instrument.HEDMInstrument instance
        the instrument class to evalutate.

    Returns
    -------
    tth_max : float
        The maximum observable Bragg angle by the instrument in radians.
    """
    tth_max = 0.0
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
            np.stack(panel.pixel_coords, axis=0)
            .reshape(2, np.cumprod(panel.shape)[-1])
            .T
        )
        ang_ps_full.append(angps)
        max_tth = min(max_tth, np.min(angps[:, 0]))
        max_eta = min(max_eta, np.min(angps[:, 1]))
        min_tth = max(min_tth, np.max(angps[:, 0]))
        min_eta = max(min_eta, np.max(angps[:, 1]))
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
            np.stack(panel.pixel_coords, axis=0)
            .reshape(2, np.cumprod(panel.shape)[-1])
            .T
        )
        mask = ~np.logical_or(
            np.isclose(angps[:, 0], 0), np.isclose(angps[:, 1], 0)
        )
        max_tth = min(max_tth, np.min(angps[mask, 0]))
        max_eta = min(max_eta, np.min(angps[mask, 1]))
    return max_tth, max_eta


def _gaussian_dist(x, cen, fwhm):
    sigm = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * (x - cen) ** 2 / sigm**2)


def _sigma_to_fwhm(sigm):
    return sigm * ct.sigma_to_fwhm


def _fwhm_to_sigma(fwhm):
    return fwhm / ct.sigma_to_fwhm


# =============================================================================
# CLASSES
# =============================================================================


class HEDMInstrument(object):
    """
    Abstraction of XRD instrument.

    * Distortion needs to be moved to a class with registry; tuple unworkable
    * where should reference eta be defined? currently set to default config
    """

    def __init__(
        self,
        instrument_config=None,
        image_series=None,
        eta_vector=None,
        instrument_name=None,
        tilt_calibration_mapping=None,
        max_workers=max_workers_DFLT,
        physics_package=None,
        active_beam_name: Optional[str] = None,
    ):
        self._id = instrument_name_DFLT

        self._active_beam_name = active_beam_name
        self._beam_dict = {}

        if eta_vector is None:
            self._eta_vector = eta_vec_DFLT
        else:
            self._eta_vector = eta_vector

        self.max_workers = max_workers

        self.physics_package = physics_package

        if instrument_config is None:
            # Default instrument
            if instrument_name is not None:
                self._id = instrument_name
            self._num_panels = 1
            self._create_default_beam()

            # FIXME: must add cylindrical
            self._detectors = dict(
                panel_id_DFLT=PlanarDetector(
                    rows=nrows_DFLT,
                    cols=ncols_DFLT,
                    pixel_size=pixel_size_DFLT,
                    tvec=t_vec_d_DFLT,
                    tilt=tilt_params_DFLT,
                    bvec=self.beam_vector,
                    xrs_dist=self.source_distance,
                    evec=self._eta_vector,
                    distortion=None,
                    roi=None,
                    group=None,
                    max_workers=self.max_workers,
                ),
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

            if instrument_config.get('physics_package', None) is not None:
                self.physics_package = instrument_config['physics_package']

            xrs_config = instrument_config['beam']
            is_single_beam = 'energy' in xrs_config and 'vector' in xrs_config
            if is_single_beam:
                # Assume single beam. Load the same way as multibeam
                self._create_default_beam()
                xrs_config = {self.active_beam_name: xrs_config}

            # Multi beam load
            for beam_name, beam in xrs_config.items():
                self._beam_dict[beam_name] = {
                    'energy': beam['energy'],
                    'vector': calc_beam_vec(
                        beam['vector']['azimuth'],
                        beam['vector']['polar_angle'],
                    ),
                    'distance': beam.get('source_distance', np.inf),
                    'energy_correction': beam.get('energy_correction', None),
                }

            # Set the active beam name if not set already
            if self._active_beam_name is None:
                self._active_beam_name = next(iter(self._beam_dict))

            # now build detector dict
            detectors_config = instrument_config['detectors']
            det_dict = dict.fromkeys(detectors_config)
            for det_id, det_info in detectors_config.items():
                det_group = det_info.get('group')  # optional detector group
                pixel_info = det_info['pixels']
                affine_info = det_info['transform']
                detector_type = det_info.get('detector_type', 'planar')
                filter = det_info.get('filter', None)
                coating = det_info.get('coating', None)
                phosphor = det_info.get('phosphor', None)
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
                                if det_buffer.shape != shape:
                                    msg = (
                                        f'Buffer shape for {det_id} '
                                        f'({det_buffer.shape}) does not match '
                                        f'detector shape ({shape})'
                                    )
                                    raise BufferShapeMismatchError(msg)
                            else:
                                if len(det_buffer) != 2:
                                    raise ValueError(
                                        f"Buffer length for {det_id} must be 2"
                                    )
                            panel_buffer = det_buffer
                        elif isinstance(det_buffer, list):
                            panel_buffer = np.asarray(det_buffer)
                        elif np.isscalar(det_buffer):
                            panel_buffer = det_buffer * np.ones(2)
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
                kwargs = dict(
                    name=det_id,
                    rows=pixel_info['rows'],
                    cols=pixel_info['columns'],
                    pixel_size=pixel_info['size'],
                    panel_buffer=panel_buffer,
                    saturation_level=saturation_level,
                    tvec=affine_info['translation'],
                    tilt=affine_info['tilt'],
                    bvec=self.beam_vector,
                    xrs_dist=self.source_distance,
                    evec=self._eta_vector,
                    distortion=distortion,
                    roi=roi,
                    group=det_group,
                    max_workers=self.max_workers,
                    detector_filter=filter,
                    detector_coating=coating,
                    phosphor=phosphor,
                )

                if DetectorClass is CylindricalDetector:
                    # Add cylindrical detector kwargs
                    kwargs['radius'] = det_info.get('radius', 49.51)

                det_dict[det_id] = DetectorClass(**kwargs)

            self._detectors = det_dict

            self._tvec = np.r_[
                instrument_config['oscillation_stage']['translation']
            ]
            self._chi = instrument_config['oscillation_stage']['chi']

        # grab angles from beam vec
        # !!! these are in DEGREES!
        azim, pola = calc_angles_from_beam_vec(self.beam_vector)

        self.update_memoization_sizes()

    @property
    def mean_detector_center(self) -> np.ndarray:
        """Return the mean center for all detectors"""
        centers = np.array([panel.tvec for panel in self.detectors.values()])
        return centers.sum(axis=0) / len(centers)

    def mean_group_center(self, group: str) -> np.ndarray:
        """Return the mean center for detectors belonging to a group"""
        centers = np.array(
            [x.tvec for x in self.detectors_in_group(group).values()]
        )
        return centers.sum(axis=0) / len(centers)

    @property
    def detector_groups(self) -> list[str]:
        groups = []
        for panel in self.detectors.values():
            group = panel.group
            if group is not None and group not in groups:
                groups.append(group)

        return groups

    def detectors_in_group(self, group: str) -> dict[str, Detector]:
        return {k: v for k, v in self.detectors.items() if v.group == group}

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
                self.chi,
                self.tvec,
                beam_energy=self.beam_energy,
                beam_vector=self.beam_vector,
                style='hdf5',
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
    def beam_energy(self) -> float:
        return self.active_beam['energy']

    @beam_energy.setter
    def beam_energy(self, x: float):
        self.active_beam['energy'] = float(x)
        self.beam_dict_modified()

    @property
    def beam_wavelength(self):
        return ct.keVToAngstrom(self.beam_energy)

    @property
    def has_multi_beam(self) -> bool:
        return len(self.beam_dict) > 1

    @property
    def beam_dict(self) -> dict:
        return self._beam_dict

    def _create_default_beam(self):
        name = 'XRS1'
        self._beam_dict[name] = {
            'energy': beam_energy_DFLT,
            'vector': beam_vec_DFLT.copy(),
            'distance': np.inf,
            'energy_correction': None,
        }

        if self._active_beam_name is None:
            self._active_beam_name = name

    @property
    def beam_names(self) -> list[str]:
        return list(self.beam_dict)

    def xrs_beam_energy(self, beam_name: Optional[str]) -> float:
        if beam_name is None:
            beam_name = self.active_beam_name

        return self.beam_dict[beam_name]['energy']

    @property
    def active_beam_name(self) -> str:
        return self._active_beam_name

    @active_beam_name.setter
    def active_beam_name(self, name: str):
        if name not in self.beam_dict:
            raise ValueError(f'"{name}" is not present in "{self.beam_names}"')

        self._active_beam_name = name

        # Update anything beam related where we need to
        self._update_panel_beams()

    def beam_dict_modified(self):
        # A function to call to indicate that the beam dict was modified.
        # Update anything beam related where we need to
        self._update_panel_beams()

    @property
    def active_beam(self) -> dict:
        return self.beam_dict[self.active_beam_name]

    def _update_panel_beams(self):
        # FIXME: maybe we shouldn't store these on the panels?
        # Might be hard to fix, though...
        for panel in self.detectors.values():
            panel.bvec = self.beam_vector
            panel.xrs_dist = self.source_distance

    @property
    def beam_vector(self) -> np.ndarray:
        return self.active_beam['vector']

    @beam_vector.setter
    def beam_vector(self, x: np.ndarray):
        """Accepts either a 3-element unit vector, or a 2-element
        (azimuth, polar angle) pair in degrees to set the beam vector."""
        x = np.array(x).flatten()
        if len(x) not in (2, 3):
            raise ValueError("beam_vector must be a 2 or 3-element array-like")

        if len(x) == 3:
            if np.abs(np.linalg.norm(x) - 1) > np.finfo(float).eps:
                raise ValueError("beam_vector must be a unit vector")
            bvec = x
        elif len(x) == 2:
            bvec = calc_beam_vec(*x)

        # Modify the beam vector for the active beam dict
        self.active_beam['vector'] = bvec
        self.beam_dict_modified()

    @property
    def source_distance(self):
        return self.active_beam['distance']

    @source_distance.setter
    def source_distance(self, x):
        assert np.isscalar(
            x
        ), f"'source_distance' must be a scalar; you input '{x}'"
        self.active_beam['distance'] = x
        self.beam_dict_modified()

    @property
    def energy_correction(self) -> Union[dict, None]:
        """Energy correction dict appears as follows:

        {
            # The beam energy gradient center, along the specified
            # axis, in millimeters.
            'intercept': 0.0,

            # The slope of the beam energy gradient along the
            # specified axis, in eV/mm.
            'slope': 0.0,

            # The specified axis for the beam energy gradient,
            # either 'x' or 'y'.
            'axis': 'y',
        }
        """
        return self.active_beam['energy_correction']

    @energy_correction.setter
    def energy_correction(self, v: Union[dict, None]):
        if v is not None:
            # First validate
            keys = sorted(list(v))
            default_keys = sorted(
                list(self.create_default_energy_correction())
            )
            if keys != default_keys:
                raise ValueError(
                    f'energy_correction keys do not match required keys.\nGot: {keys}\nExpected: {default_keys}'
                )

        self.active_beam['energy_correction'] = v

    @staticmethod
    def create_default_energy_correction() -> dict[str, float]:
        return {
            'intercept': 0.0,  # in mm
            'slope': 0.0,  # eV/mm
            'axis': 'y',
        }

    @property
    def eta_vector(self):
        return self._eta_vector

    @eta_vector.setter
    def eta_vector(self, x):
        x = np.array(x).flatten()
        if len(x) != 3:
            raise ValueError("eta_vector must be a 3-element array-like")
        elif np.abs(np.linalg.norm(x) - 1) > np.finfo(float).eps:
            raise ValueError("eta_vector must be a unit vector")
        self._eta_vector = x
        # ...maybe change dictionary item behavior for 3.x compatibility?
        for detector_id in self.detectors:
            panel = self.detectors[detector_id]
            panel.evec = self._eta_vector

    # =========================================================================
    # METHODS
    # =========================================================================

    def write_config(self, file=None, style='yaml', calibration_dict={}):
        """WRITE OUT YAML FILE"""
        # initialize output dictionary
        if style.lower() not in ['yaml', 'hdf5']:
            raise ValueError(
                f"style must be 'yaml' or 'hdf5' but is '{style}'"
            )

        par_dict = {}

        par_dict['id'] = self.id

        # Multi beam writer
        beam_dict = {}
        for beam_name, beam in self.beam_dict.items():
            azim, polar = calc_angles_from_beam_vec(beam['vector'])
            beam_dict[beam_name] = {
                'energy': beam['energy'],
                'vector': {
                    'azimuth': azim,
                    'polar_angle': polar,
                },
            }
            if beam.get('distance') != np.inf:
                beam_dict[beam_name]['source_distance'] = beam['distance']

            if beam.get('energy_correction') is not None:
                beam_dict[beam_name]['energy_correction'] = beam[
                    'energy_correction'
                ]

        if len(beam_dict) == 1:
            # Just write it out a single beam (classical way)
            beam_dict = next(iter(beam_dict.values()))

        par_dict['beam'] = beam_dict

        if calibration_dict:
            par_dict['calibration_crystal'] = calibration_dict

        ostage = dict(chi=self.chi, translation=self.tvec.tolist())
        par_dict['oscillation_stage'] = ostage

        det_dict = dict.fromkeys(self.detectors)
        for det_name, detector in self.detectors.items():
            # grab panel config
            # !!! don't need beam or tvec
            # !!! have vetted style
            pdict = detector.config_dict(
                chi=self.chi,
                tvec=self.tvec,
                beam_energy=self.beam_energy,
                beam_vector=self.beam_vector,
                style=style,
            )
            det_dict[det_name] = pdict['detector']
        par_dict['detectors'] = det_dict

        # handle output file if requested
        if file is not None:
            if style.lower() == 'yaml':
                with open(file, 'w') as f:
                    yaml.dump(par_dict, stream=f, Dumper=NumpyToNativeDumper)
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

    def extract_polar_maps(
        self,
        plane_data,
        imgser_dict,
        active_hkls=None,
        threshold=None,
        tth_tol=None,
        eta_tol=0.25,
    ):
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
        pow_angs, pow_xys, tth_ranges, eta_idx, eta_edges = (
            panel.make_powder_rings(
                plane_data,
                merge_hkls=False,
                delta_eta=eta_tol,
                full_output=True,
            )
        )

        if active_hkls is not None:
            assert hasattr(
                active_hkls, '__len__'
            ), "active_hkls must be an iterable with __len__"

            # need to re-cast for element-wise operations
            active_hkls = np.array(active_hkls)

            # these are all active reflection unique hklIDs
            active_hklIDs = plane_data.getHKLID(plane_data.hkls, master=True)

            # find indices
            idx = np.zeros_like(active_hkls, dtype=int)
            for i, input_hklID in enumerate(active_hkls):
                try:
                    idx[i] = np.where(active_hklIDs == input_hklID)[0]
                except ValueError:
                    raise RuntimeError(f"hklID '{input_hklID}' is invalid")
            tth_ranges = tth_ranges[idx]

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
            func = partial(
                _run_histograms,
                ims=ims,
                tth_ranges=tth_ranges,
                ring_maps=ring_maps,
                ring_params=ring_params,
                threshold=threshold,
            )

            max_workers = self.max_workers
            if max_workers == 1 or len(tasks) == 1:
                # Just execute it serially.
                for task in tasks:
                    func(task)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Evaluate the results via `list()`, so that if an
                    # exception is raised in a thread, it will be re-raised
                    # and visible to the user.
                    list(executor.map(func, tasks))

            ring_maps_panel[det_key] = ring_maps

        return ring_maps_panel, eta_edges

    def extract_line_positions(
        self,
        plane_data,
        imgser_dict,
        tth_tol=None,
        eta_tol=1.0,
        npdiv=2,
        eta_centers=None,
        collapse_eta=True,
        collapse_tth=False,
        do_interpolation=True,
        do_fitting=False,
        tth_distortion=None,
        fitting_kwargs=None,
    ):
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
            for special case of pinhole camera distortions.
            See classes in hexrd.xrdutil.phutil
        fitting_kwargs : dict, optional
            kwargs passed to hexrd.core.fitting.utils.fit_ring if do_fitting is True

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
        pbar_dets = partial(
            tqdm,
            total=self.num_panels,
            desc="Detector",
            position=self.num_panels,
        )

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
                chi=self.chi,
                tvec=self.tvec,
                beam_energy=self.beam_energy,
                beam_vector=self.beam_vector,
                style='hdf5',
            )

        images = []
        for detector_id, panel in self.detectors.items():
            images.append(
                _parse_imgser_dict(imgser_dict, detector_id, roi=panel.roi)
            )

        panels = [self.detectors[k] for k in self.detectors]
        instr_cfgs = [make_instr_cfg(x) for x in panels]
        pbp_array = np.arange(self.num_panels)
        iter_args = zip(panels, instr_cfgs, images, pbp_array)
        with ProcessPoolExecutor(
            mp_context=constants.mp_context, max_workers=self.num_panels
        ) as executor:
            results = list(pbar_dets(executor.map(func, iter_args)))

        panel_data = {}
        for det, res in zip(self.detectors, results):
            panel_data[det] = res

        return panel_data

    def simulate_powder_pattern(
        self, mat_list, params=None, bkgmethod=None, origin=None, noise=None
    ):
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
        assert len(origin) == 3, "origin must be a 3-element sequence"

        if bkgmethod is None:
            bkgmethod = {'chebyshev': 3}

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
                1,
                bkgmethod,
            )
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
        tth_ma = 0.0
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
        nsteps = int(np.ceil(2 * (tth_ma - tth_mi) / np.degrees(ang_res[0])))

        # evaulation vector for LeBail
        tth = np.linspace(tth_mi, tth_ma, nsteps)

        expt = np.vstack([tth, np.ones_like(tth)]).T

        wavelength = [
            valWUnit('lp', 'length', self.beam_wavelength, 'angstrom'),
            1.0,
        ]

        '''
        now go through the material list and get the intensity dictionary
        '''
        intensity = {}
        for mat in mat_list:

            multiplicity = mat.planeData.getMultiplicity()

            tth = mat.planeData.getTTh()

            LP = (
                (1 + np.cos(tth) ** 2)
                / np.cos(0.5 * tth)
                / np.sin(0.5 * tth) ** 2
            )

            intensity[mat.name] = {}
            intensity[mat.name]['synchrotron'] = (
                mat.planeData.structFact * LP * multiplicity
            )

        kwargs = {
            'expt_spectrum': expt,
            'params': params,
            'phases': mat_list,
            'wavelength': {'synchrotron': wavelength},
            'bkgmethod': bkgmethod,
            'intensity_init': intensity,
            'peakshape': 'pvtch',
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

            img = np.interp(
                np.degrees(ptth),
                self.simulated_spectrum.x,
                self.simulated_spectrum.y + self.background.y,
            )

            if noise is None:
                img_dict[det_key] = img

            else:
                # Rescale to be between 0 and 1 so random_noise() will work
                prev_max = img.max()
                img /= prev_max

                if noise.lower() == 'poisson':
                    im_noise = random_noise(img, mode='poisson', clip=True)
                    mi = im_noise.min()
                    ma = im_noise.max()
                    if ma > mi:
                        im_noise = (im_noise - mi) / (ma - mi)

                elif noise.lower() == 'gaussian':
                    im_noise = random_noise(img, mode='gaussian', clip=True)

                elif noise.lower() == 'salt':
                    im_noise = random_noise(img, mode='salt')

                elif noise.lower() == 'pepper':
                    im_noise = random_noise(img, mode='pepper')

                elif noise.lower() == 's&p':
                    im_noise = random_noise(img, mode='s&p')

                elif noise.lower() == 'speckle':
                    im_noise = random_noise(img, mode='speckle', clip=True)

                # Now scale back up
                img_dict[det_key] = im_noise * prev_max

        return img_dict

    def simulate_laue_pattern(
        self,
        crystal_data,
        minEnergy=5.0,
        maxEnergy=35.0,
        rmat_s=None,
        grain_params=None,
    ):
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
                minEnergy=minEnergy,
                maxEnergy=maxEnergy,
                rmat_s=rmat_s,
                tvec_s=self.tvec,
                grain_params=grain_params,
                beam_vec=self.beam_vector,
            )
        return results

    def simulate_rotation_series(
        self,
        plane_data,
        grain_param_list,
        eta_ranges=[
            (-np.pi, np.pi),
        ],
        ome_ranges=[
            (-np.pi, np.pi),
        ],
        ome_period=(-np.pi, np.pi),
        wavelength=None,
    ):
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
                plane_data,
                grain_param_list,
                eta_ranges=eta_ranges,
                ome_ranges=ome_ranges,
                ome_period=ome_period,
                chi=self.chi,
                tVec_s=self.tvec,
                wavelength=wavelength,
                energy_correction=self.energy_correction,
            )
        return results

    def pull_spots(
        self,
        plane_data: PlaneData,
        grain_params: tuple | np.ndarray,
        imgser_dict: dict,
        tth_tol: Optional[float] = 0.25,
        eta_tol: Optional[float] = 1.0,
        ome_tol: Optional[float] = 1.0,
        npdiv: Optional[int] = 2,
        threshold: Optional[int] = 10,
        eta_ranges: Optional[tuple[tuple]] = [(-np.pi, np.pi)],
        ome_period: Optional[tuple] = None,
        dirname: Optional[str] = 'results',
        filename: Optional[os.PathLike] = None,
        output_format: Literal['text', 'hdf5'] = 'text',
        return_spot_list: Optional[bool] = False,
        check_only: Optional[bool] = False,
        interp: Literal['nearest', 'bilinear'] = 'nearest',
        quiet: Optional[Any] = None,
    ) -> tuple[tuple, dict]:
        """Extract reflection info from a rotation series.

        Input must be encoded as an OmegaImageseries object.

        plane_data : PlaneData
            Object containing crystallographic plane data
        grain_params : list or np.ndarray
            A 6-element array defining the grain's orientation and position.
            The first 3 elements are orientation parameters (exponential map),
            and the last 3 are the translation vector in the sample frame.
        imgser_dict : dict
            Map for detector IDs to OmegaImageSeries objects.
        tth_tol : float, optional
            Tolerance in 2-theta (radial direction) in degrees. Default 0.25.
        eta_tol : float, optional
            Tolerance in eta (azimuthal direction) in degrees. Default 1.0.
        ome_tol : float, optional
            Tolerance in omega (rotation) in degrees. Default 1.0.
        npdiv : int, optional
            Number of sub-pixel divisions for patch interpolation. Default 2.
        threshold : int, optional
            Intensity threshold for peak detection. Default 10.
        eta_ranges : list of tuple, optional
            List of (min, max) tuples defining valid eta ranges in radians.
            Default [(-np.pi, np.pi)].
        ome_period : tuple, optional
            (min, max) tuple defining the periodicity of omega in radians.
            If None, it is inferred from the image series. Default None.
        dirname : str, optional
            Directory path for saving output files. Default 'results'.
        filename : str, optional
            Base filename for output. If None, no files are written. Default None.
        output_format : str, optional
            Format for output files, either 'text' or 'hdf5'. Default 'text'.
        return_spot_list : bool, optional
            If True, the returned dictionary contains full patch data (images, coordinates)
            rather than just summary statistics. Default False.
        check_only : bool, optional
            If True, performs a geometric check to see if spots fall on the detector
            without performing full interpolation or integration. Default False.
        interp : str, optional
            Interpolation method for patch extraction. Options are 'bilinear' or 'nearest'.
            Default 'nearest'.

        Returns
        -------
        patch_has_signal : list[bool]
            A list indicating whether signal was found for each predicted reflection.
        output : dict[str, list]
            A dictionary keyed by detector ID. The values are lists containing extracted
            spot data. If `return_spot_list`, these lists contain full patch
            arrays and coordinates; otherwise, they contain summary data.
        """
        if quiet is not None:
            logging.warning(
                "'quiet' argument is deprecated and is marked for removal.",
                DeprecationWarning,
            )

        if check_only:
            return self._pull_spots_check_only(
                plane_data,
                grain_params,
                imgser_dict,
                tth_tol=tth_tol,
                eta_tol=eta_tol,
                ome_tol=ome_tol,
                threshold=threshold,
                eta_ranges=eta_ranges,
                ome_period=ome_period,
                filename=filename,
            )

        if filename is not None and not isinstance(filename, str):
            raise TypeError("filename must be a string type")

        if not isinstance(output_format, str):
            raise TypeError("output_format must be a string type")
        output_format = output_format.lower()
        interp = interp.lower()

        write_hdf5 = filename is not None and output_format == 'hdf5'
        write_text = filename is not None and output_format == 'text'

        rMat_c = make_rmat_of_expmap(grain_params[:3])
        tVec_c = grain_params[3:6]

        oims0 = next(iter(imgser_dict.values()))
        omega_ranges = [
            np.radians([i['ostart'], i['ostop']])
            for i in oims0.omegawedges.wedges
        ]
        if ome_period is None:
            ims = next(iter(imgser_dict.values()))
            ome_period = np.radians(ims.omega[0, 0] + np.array([0.0, 360.0]))

        # delta omega in DEGREES grabbed from first imageseries in the dict
        delta_omega = oims0.omega[0, 1] - oims0.omega[0, 0]

        # make omega grid for frame expansion around reference frame in DEGREES
        ndiv_ome, ome_grid = make_tolerance_grid(
            delta_omega, ome_tol, 1, adjust_window=True
        )

        # generate structuring element for connected component labeling
        if ndiv_ome == 1:
            label_struct = ndimage.generate_binary_structure(2, 2)
        else:
            label_struct = ndimage.generate_binary_structure(3, 3)

        sim_results = self.simulate_rotation_series(
            plane_data,
            [grain_params],
            eta_ranges=eta_ranges,
            ome_ranges=omega_ranges,
            ome_period=ome_period,
        )

        if write_hdf5:
            writer = GrainDataWriter_h5(
                os.path.join(dirname, filename),
                self.write_config(),
                grain_params,
            )

        # =====================================================================
        # LOOP OVER PANELS
        # =====================================================================
        next_invalid_peak_id = -100
        patch_has_signal = []
        output = defaultdict(list)
        for detector_id, panel in self.detectors.items():
            hkl_ids, hkls_p, ang_centers, xy_centers, ang_pixel_size = [
                item[0] for item in sim_results[detector_id]
            ]

            _, mask = self._get_panel_mask(
                tth_tol, eta_tol, ang_centers, panel, rMat_c, tVec_c
            )

            hkls_p = hkls_p[mask]
            ang_centers = ang_centers[mask]
            hkl_ids = hkl_ids[mask]
            xy_centers = xy_centers[mask]

            patches = self._get_panel_patches(
                panel,
                ang_centers,
                ang_pixel_size[mask],
                tth_tol,
                eta_tol,
                rMat_c,
                tVec_c,
                npdiv,
            )
            omega_image_series = _parse_imgser_dict(
                imgser_dict, detector_id, roi=panel.roi
            )

            if write_text:
                output_dir = os.path.join(dirname, detector_id)
                os.makedirs(output_dir, exist_ok=True)
                writer = PatchDataWriter(os.path.join(output_dir, filename))

            for patch_id, (vtx_angs, _, _, areas, xy_eval, ijs) in enumerate(
                patches
            ):
                prows, pcols = areas.shape
                hkl_id, hkl = hkl_ids[patch_id], hkls_p[patch_id, :]

                tth_edges = vtx_angs[0][0, :]
                eta_edges = vtx_angs[1][:, 0]
                delta_ttheta = tth_edges[1] - tth_edges[0]
                delta_eta = eta_edges[1] - eta_edges[0]

                omega_eval = np.degrees(ang_centers[patch_id, 2]) + ome_grid

                frame_indices = [
                    omega_image_series.omega_to_frame(omega)[0]
                    for omega in omega_eval
                ]
                if -1 in frame_indices:
                    logging.info(f"window for {hkl} falls outside omega range")
                    continue

                omega_edges = omega_image_series.omega[frame_indices[0]][0]
                patch_data_raw = np.stack(
                    [
                        omega_image_series[i, ijs[0], ijs[1]]
                        for i in frame_indices
                    ],
                    axis=0,
                )
                contains_signal = np.any(patch_data_raw > threshold)
                patch_has_signal.append(contains_signal)

                # initialize spot data parameters
                peak_id = next_invalid_peak_id
                sum_intensities = np.nan
                max_intensities = np.nan
                meas_angs = np.full(3, np.nan)
                meas_xy = np.full(2, np.nan)

                patch_data = patch_data_raw

                # need to reshape eval pts for interpolation
                xy_eval = np.vstack(
                    [xy_eval[0].flatten(), xy_eval[1].flatten()]
                ).T

                if contains_signal:
                    # overwrite patch data if using bilinear option
                    if interp == 'bilinear':
                        patch_data = np.stack(
                            [
                                panel.interpolate_bilinear(
                                    xy_eval,
                                    omega_image_series[i_frame],
                                    pad_with_nans=False,
                                ).reshape(prows, pcols)
                                for i_frame in frame_indices
                            ],
                            axis=0,
                        )

                    # now we have interpolated patch data...
                    labels, num_peaks = ndimage.label(
                        patch_data > threshold, label_struct
                    )

                    if num_peaks == 0:
                        continue

                    peak_id = patch_id
                    coms = np.vstack(
                        [
                            x.weighted_centroid
                            for x in regionprops(labels, patch_data)
                        ]
                    )
                    closest_peak_idx = np.argmin(
                        np.sum(
                            (coms - np.array(patch_data.shape) / 2) ** 2,
                            axis=1,
                        )
                    )

                    mask = labels == (closest_peak_idx + 1)

                    sum_intensities = patch_data[mask].sum()
                    max_intensities = patch_data_raw[mask].max()

                    meas_angs = np.hstack(
                        [
                            tth_edges[0]
                            + (0.5 + coms[closest_peak_idx][2]) * delta_ttheta,
                            eta_edges[0]
                            + (0.5 + coms[closest_peak_idx][1]) * delta_eta,
                            mapAngle(
                                np.radians(
                                    (
                                        omega_edges
                                        + (0.5 + coms[closest_peak_idx][0])
                                        * delta_omega
                                    )
                                ),
                                ome_period,
                            ),
                        ]
                    )

                    meas_xy = self._get_meas_xy(
                        panel,
                        meas_angs,
                        rMat_c,
                        tVec_c,
                    )

                if peak_id < 0:
                    # The peak is invalid. Decrement next invalid peak ID.
                    next_invalid_peak_id -= 1

                if write_text:
                    writer.dump_patch(
                        peak_id,
                        hkl_id,
                        hkl,
                        sum_intensities,
                        max_intensities,
                        ang_centers[patch_id],
                        meas_angs,
                        xy_centers[patch_id],
                        meas_xy,
                    )
                elif write_hdf5:
                    writer.dump_patch(
                        detector_id,
                        patch_id,
                        peak_id,
                        hkl_id,
                        hkl,
                        tth_edges,
                        eta_edges,
                        np.radians(omega_eval),
                        xy_eval.T.reshape(2, prows, pcols),
                        ijs,
                        frame_indices,
                        patch_data,
                        ang_centers[patch_id],
                        xy_centers[patch_id],
                        meas_angs,
                        meas_xy,
                    )

                if return_spot_list:
                    # Full output
                    output[detector_id].append(
                        [
                            detector_id,
                            patch_id,
                            peak_id,
                            hkl_id,
                            hkl,
                            tth_edges,
                            eta_edges,
                            np.radians(omega_eval),
                            xy_eval.T.reshape(2, prows, pcols),
                            ijs,
                            frame_indices,
                            patch_data,
                            ang_centers[patch_id],
                            xy_centers[patch_id],
                            meas_angs,
                            meas_xy,
                        ]
                    )
                else:
                    # Trimmed output
                    output[detector_id].append(
                        [
                            peak_id,
                            hkl_id,
                            hkl,
                            sum_intensities,
                            max_intensities,
                            ang_centers[patch_id],
                            meas_angs,
                            meas_xy,
                        ]
                    )
            if write_text:
                writer.close()
        if write_hdf5:
            writer.close()
        return patch_has_signal, output

    def _pull_spots_check_only(
        self,
        plane_data: PlaneData,
        grain_params: tuple | np.ndarray,
        imgser_dict: dict,
        tth_tol: Optional[float] = 0.25,
        eta_tol: Optional[float] = 1.0,
        ome_tol: Optional[float] = 1.0,
        threshold: Optional[int] = 10,
        eta_ranges: Optional[tuple[tuple]] = [(-np.pi, np.pi)],
        ome_period: Optional[tuple] = None,
    ):
        rMat_c = make_rmat_of_expmap(grain_params[:3])
        tVec_c = grain_params[3:6]

        oims0 = next(iter(imgser_dict.values()))
        omega_ranges = [
            np.radians([i['ostart'], i['ostop']])
            for i in oims0.omegawedges.wedges
        ]
        if ome_period is None:
            ims = next(iter(imgser_dict.values()))
            ome_period = np.radians(ims.omega[0, 0] + np.array([0.0, 360.0]))

        delta_omega = oims0.omega[0, 1] - oims0.omega[0, 0]
        _, ome_grid = make_tolerance_grid(
            delta_omega, ome_tol, 1, adjust_window=True
        )

        sim_results = self.simulate_rotation_series(
            plane_data,
            [grain_params],
            eta_ranges=eta_ranges,
            ome_ranges=omega_ranges,
            ome_period=ome_period,
        )

        patch_has_signal = []
        output = defaultdict(list)
        for detector_id, panel in self.detectors.items():
            # pull out the OmegaImageSeries for this panel from input dict
            omega_image_series = _parse_imgser_dict(
                imgser_dict, detector_id, roi=panel.roi
            )

            hkl_ids, hkls_p, ang_centers, xy_centers, ang_pixel_size = [
                item[0] for item in sim_results[detector_id]
            ]

            det_xy, mask = self._get_panel_mask(
                tth_tol, eta_tol, ang_centers, panel, rMat_c, tVec_c
            )

            patch_xys = det_xy.reshape(-1, 4, 2)[mask]
            hkls_p = hkls_p[mask]
            ang_centers = ang_centers[mask]
            hkl_ids = hkl_ids[mask]
            xy_centers = xy_centers[mask]
            ang_pixel_size = ang_pixel_size[mask]

            for ang_index, angs in enumerate(ang_centers):
                omega_eval = np.degrees(angs[2]) + ome_grid

                frame_indices = [
                    omega_image_series.omega_to_frame(omega)[0]
                    for omega in omega_eval
                ]
                if -1 in frame_indices:
                    logging.info(
                        f"window for {hkls_p[ang_index, :]} falls outside omega range"
                    )
                    continue

                ijs = panel.cartToPixel(patch_xys[ang_index])
                ii, jj = polygon(ijs[:, 0], ijs[:, 1])

                patch_data_raw = np.stack(
                    [
                        omega_image_series[i_frame, ii, jj]
                        for i_frame in frame_indices
                    ],
                    axis=0,
                )
                patch_has_signal.append(np.any(patch_data_raw > threshold))
                output[detector_id].append((ii, jj, frame_indices))

        return patch_has_signal, output

    def _get_panel_mask(
        self,
        tth_tol: float,
        eta_tol: float,
        ang_centers: np.array,
        panel: Any,
        rMat_c: np.ndarray,
        tVec_c: np.ndarray,
    ) -> np.ndarray:
        offsets = 0.5 * np.radians(
            [
                [-tth_tol, -eta_tol, 0],
                [-tth_tol, eta_tol, 0],
                [tth_tol, eta_tol, 0],
                [tth_tol, -eta_tol, 0],
            ]
        )
        lab_coords = (ang_centers[:, None, :] + offsets).reshape(-1, 3)
        det_xy, _, _ = _project_on_detector_plane(
            lab_coords,
            panel.rmat,
            rMat_c,
            self.chi,
            panel.tvec,
            tVec_c,
            self.tvec,
            panel.distortion,
        )

        _, on_panel = panel.clip_to_panel(det_xy, buffer_edges=True)
        mask = on_panel.reshape(-1, 4).all(axis=1)
        return det_xy, mask

    def _get_panel_patches(
        self,
        panel: Any,
        ang_centers: np.ndarray,
        ang_pixel_size: np.array,
        tth_tol: float,
        eta_tol: float,
        rMat_c: np.array,
        tvec_c: np.array,
        npdiv: int,
    ):
        instrument_config = panel.config_dict(
            self.chi,
            self.tvec,
            beam_energy=self.beam_energy,
            beam_vector=self.beam_vector,
            style='hdf5',
        )
        patches = xrdutil.make_reflection_patches(
            instrument_config,
            ang_centers[:, :2],
            ang_pixel_size,
            omega=ang_centers[:, 2],
            tth_tol=tth_tol,
            eta_tol=eta_tol,
            rmat_c=rMat_c,
            tvec_c=tvec_c,
            npdiv=npdiv,
        )
        return patches

    def _get_meas_xy(
        self,
        panel: Any,
        meas_angles: np.array,
        rMat_c: np.array,
        tVec_c: np.array,
    ):
        gvec_c = angles_to_gvec(
            meas_angles,
            self.beam_vector,
            chi=self.chi,
            rmat_c=rMat_c,
        )
        rMat_s = make_sample_rmat(self.chi, meas_angles[2])
        meas_xy = gvec_to_xy(
            gvec_c,
            panel.rmat,
            rMat_s,
            rMat_c,
            panel.tvec,
            self.tvec,
            tVec_c,
            self.beam_vector,
        )

        if panel.distortion is not None:
            meas_xy = panel.distortion.apply_inverse(
                np.atleast_2d(meas_xy)
            ).flatten()
        return meas_xy

    def update_memoization_sizes(self):
        # Resize all known memoization functions to have a cache at least
        # the size of the number of detectors.
        all_panels = list(self.detectors.values())
        PlanarDetector.update_memoization_sizes(all_panels)
        CylindricalDetector.update_memoization_sizes(all_panels)

    def calc_transmission(
        self, rMat_s: np.ndarray = None
    ) -> dict[str, np.ndarray]:
        """calculate the transmission from the
        filter and polymer coating. the inverse of this
        number is the intensity correction that needs
        to be applied. actual computation is done inside
        the detector class
        """
        if rMat_s is None:
            rMat_s = ct.identity_3x3

        energy = self.beam_energy
        transmissions = {}
        for det_name, det in self.detectors.items():
            transmission_filter, transmission_phosphor = (
                det.calc_filter_coating_transmission(energy)
            )

            transmission = transmission_filter * transmission_phosphor

            if self.physics_package is not None:
                transmission_physics_package = (
                    det.calc_physics_package_transmission(
                        energy, rMat_s, self.physics_package
                    )
                )
                effective_pinhole_area = det.calc_effective_pinhole_area(
                    self.physics_package
                )

                transmission = (
                    transmission
                    * transmission_physics_package
                    * effective_pinhole_area
                )

            transmissions[det_name] = transmission
        return transmissions


# =============================================================================
# UTILITIES
# =============================================================================


class PatchDataWriter(object):
    """Class for dumping Bragg reflection data."""

    def __init__(self, filename):
        self._delim = '  '
        # fmt: off
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
        # fmt: on
        if isinstance(filename, IOBase):
            self.fid = filename
        else:
            self.fid = open(filename, 'w')
        print(self._header, file=self.fid)

    def __del__(self):
        self.close()

    def close(self):
        self.fid.close()

    def dump_patch(
        self, peak_id, hkl_id, hkl, spot_int, max_int, pangs, mangs, pxy, mxy
    ):
        """
        !!! maybe need to check that last four inputs are arrays
        """
        if mangs is None:
            spot_int = np.nan
            max_int = np.nan
            mangs = np.nan * np.ones(3)
            mxy = np.nan * np.ones(2)

        res = (
            [int(peak_id), int(hkl_id)]
            + np.array(hkl, dtype=int).tolist()
            + [spot_int, max_int]
            + pangs.tolist()
            + mangs.tolist()
            + pxy.tolist()
            + mxy.tolist()
        )

        output_str = self._delim.join(
            [
                self._delim.join(np.tile('{:<6d}', 5)).format(*res[:5]),
                self._delim.join(np.tile('{:<12e}', 2)).format(*res[5:7]),
                self._delim.join(np.tile('{:<23.16e}', 10)).format(*res[7:]),
            ]
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
                'GrainDataWriter must be specified with filename or array'
            )

        self.array = None
        self.fid = None

        # array supersedes filename
        if array is not None:
            assert (
                array.shape[1] == 21
            ), f'grain data table must have 21 columns not {array.shape[21]}'
            self.array = array
            self._array_row = 0
            return

        self._delim = '  '
        # fmt: off
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
        # fmt: on
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

    def dump_grain(self, grain_id, completeness, chisq, grain_params):
        assert (
            len(grain_params) == 12
        ), "len(grain_params) must be 12, not %d" % len(grain_params)

        # extract strain
        emat = logm(np.linalg.inv(mutil.vecMVToSymm(grain_params[6:])))
        evec = mutil.symmToVecMV(emat, scale=False)

        res = (
            [int(grain_id), completeness, chisq]
            + grain_params.tolist()
            + evec.tolist()
        )

        if self.array is not None:
            row = self._array_row
            assert (
                row < self.array.shape[0]
            ), f'invalid row {row} in array table'
            self.array[row] = res
            self._array_row += 1
            return res

        # (else) format and write to file
        output_str = self._delim.join(
            [
                self._delim.join(['{:<12d}', '{:<12f}', '{:<12e}']).format(
                    *res[:3]
                ),
                self._delim.join(np.tile('{:<23.16e}', len(res) - 3)).format(
                    *res[3:]
                ),
            ]
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
        rmat_c = make_rmat_of_expmap(grain_params[:3])
        tvec_c = np.array(grain_params[3:6]).flatten()
        vinv_s = np.array(grain_params[6:]).flatten()
        vmat_s = np.linalg.inv(mutil.vecMVToSymm(vinv_s))

        if use_attr:  # attribute version
            self.grain_grp.attrs.create('rmat_c', rmat_c)
            self.grain_grp.attrs.create('tvec_c', tvec_c)
            self.grain_grp.attrs.create('inv(V)_s', vinv_s)
            self.grain_grp.attrs.create('vmat_s', vmat_s)
        else:  # dataset version
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

    def dump_patch(
        self,
        panel_id,
        i_refl,
        peak_id,
        hkl_id,
        hkl,
        tth_edges,
        eta_edges,
        ome_centers,
        xy_centers,
        ijs,
        frame_indices,
        spot_data,
        pangs,
        pxy,
        mangs,
        mxy,
        gzip=1,
    ):
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
            mangs = np.nan * np.ones(3)
        spot_grp.attrs.create('measured_angles', mangs)
        if mxy is None:
            mxy = np.nan * np.ones(3)
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
        spot_grp.create_dataset(
            'tth_crd',
            data=tth_crd,
            compression="gzip",
            compression_opts=gzip,
            shuffle=shuffle_data,
        )
        spot_grp.create_dataset(
            'eta_crd',
            data=eta_crd,
            compression="gzip",
            compression_opts=gzip,
            shuffle=shuffle_data,
        )
        spot_grp.create_dataset(
            'ome_crd',
            data=ome_centers,
            compression="gzip",
            compression_opts=gzip,
            shuffle=shuffle_data,
        )
        spot_grp.create_dataset(
            'xy_centers',
            data=xy_centers,
            compression="gzip",
            compression_opts=gzip,
            shuffle=shuffle_data,
        )
        spot_grp.create_dataset(
            'ij_centers',
            data=ijs,
            compression="gzip",
            compression_opts=gzip,
            shuffle=shuffle_data,
        )
        spot_grp.create_dataset(
            'frame_indices',
            data=fi,
            compression="gzip",
            compression_opts=gzip,
            shuffle=shuffle_data,
        )
        spot_grp.create_dataset(
            'intensities',
            data=spot_data,
            compression="gzip",
            compression_opts=gzip,
            shuffle=shuffle_data,
        )
        return


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

    def __init__(
        self,
        image_series_dict,
        instrument,
        plane_data,
        active_hkls=None,
        eta_step=0.25,
        threshold=None,
        ome_period=(0, 360),
    ):
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
            self._iHKLList = plane_data.getHKLID(plane_data.hkls, master=True)
            n_rings = len(self._iHKLList)
        else:
            assert hasattr(
                active_hkls, '__len__'
            ), "active_hkls must be an iterable with __len__"
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
        ome_period = omegas_array[0, 0] + np.r_[0.0, 360.0]  # !!! be careful
        if this_det_ims.omegawedges.nwedges > 1:
            delta_omes = [
                (i['ostop'] - i['ostart']) / i['nsteps']
                for i in this_det_ims.omegawedges.wedges
            ]
            check_wedges = mutil.uniqueVectors(
                np.atleast_2d(delta_omes), tol=1e-6
            ).squeeze()
            assert (
                check_wedges.size == 1
            ), "all wedges must have the same delta omega to 1e-6"
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
            nsteps = int((ostop - ostart) / delta_ome)
            ome_edges_full = np.linspace(
                ostart, ostop, num=nsteps + 1, endpoint=True
            )
            omegas_array = np.vstack(
                [ome_edges_full[:-1], ome_edges_full[1:]]
            ).T
            ome_centers = np.average(omegas_array, axis=1)

            # use OmegaImageSeries method to determine which bins have data
            # !!! this array has -1 outside a wedge
            # !!! again assuming the valid frame order increases monotonically
            frame_mask = np.array(
                [
                    this_det_ims.omega_to_frame(ome)[0] != -1
                    for ome in ome_centers
                ]
            )

        # ???: need to pass a threshold?
        eta_mapping, etas = instrument.extract_polar_maps(
            plane_data,
            image_series_dict,
            active_hkls=active_hkls,
            threshold=threshold,
            tth_tol=None,
            eta_tol=eta_step,
        )

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
                tmp = np.ones((len(frame_mask), map_shape[1])) * np.nan
                tmp[frame_mask, :] = full_map
                full_map = tmp
            data_store.append(full_map)
        self._dataStore = data_store

        # set required attributes
        self._omegas = mapAngle(
            np.radians(np.average(omegas_array, axis=1)),
            np.radians(ome_period),
        )
        self._omeEdges = mapAngle(
            np.radians(np.r_[omegas_array[:, 0], omegas_array[-1, 1]]),
            np.radians(ome_period),
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
        self._etas = self._etaEdges[:-1] + 0.5 * np.radians(eta_step)

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


def _generate_ring_params(tthr, ptth, peta, eta_edges, delta_eta):
    # mark pixels in the spec'd tth range
    pixels_in_tthr = np.logical_and(ptth >= tthr[0], ptth <= tthr[1])

    # catch case where ring isn't on detector
    if not np.any(pixels_in_tthr):
        return None

    pixel_ids = np.where(pixels_in_tthr)

    # grab relevant eta coords using histogram
    pixel_etas = peta[pixel_ids]
    reta_hist = histogram(pixel_etas, eta_edges)
    bins_on_detector = np.where(reta_hist)[0]

    return pixel_etas, eta_edges, pixel_ids, bins_on_detector


def run_fast_histogram(x, bins, weights=None):
    return histogram1d(x, len(bins) - 1, (bins[0], bins[-1]), weights=weights)


def run_numpy_histogram(x, bins, weights=None):
    return histogram1d(x, bins=bins, weights=weights)[0]


histogram = run_fast_histogram if fast_histogram else run_numpy_histogram


def _run_histograms(rows, ims, tth_ranges, ring_maps, ring_params, threshold):
    for i_row in range(*rows):
        image = ims[i_row]

        # handle threshold if specified
        if threshold is not None:
            # !!! NaNs get preserved
            image = np.array(image)
            image[image < threshold] = 0.0

        for i_r, tthr in enumerate(tth_ranges):
            this_map = ring_maps[i_r]
            params = ring_params[i_r]
            if not params:
                # We are supposed to skip this ring...
                continue

            # Unpack the params
            pixel_etas, eta_edges, pixel_ids, bins_on_detector = params
            result = histogram(pixel_etas, eta_edges, weights=image[pixel_ids])

            # Note that this preserves nan values for bins not on the detector.
            this_map[i_row, bins_on_detector] = result[bins_on_detector]


def _extract_detector_line_positions(
    iter_args,
    plane_data,
    tth_tol,
    eta_tol,
    eta_centers,
    npdiv,
    collapse_tth,
    collapse_eta,
    do_interpolation,
    do_fitting,
    fitting_kwargs,
    tth_distortion,
    max_workers,
):
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
        plane_data,
        merge_hkls=True,
        delta_tth=tth_tol,
        delta_eta=eta_tol,
        eta_list=eta_centers,
        tth_distortion=tth_distr_cls,
    )

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
    pbar_rings = partial(
        tqdm, total=len(pow_angs), desc="Ringset", position=pbp
    )

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
    with ProcessPoolExecutor(
        mp_context=constants.mp_context, max_workers=max_workers
    ) as executor:
        return list(pbar_rings(executor.map(func, iter_arg)))


def _extract_ring_line_positions(
    iter_args,
    instr_cfg,
    panel,
    eta_tol,
    npdiv,
    collapse_tth,
    collapse_eta,
    images,
    do_interpolation,
    do_fitting,
    fitting_kwargs,
    tth_distortion,
):
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

    # SS 01/31/25 noticed some nans in xys even after clipping
    # going to do another round of masking to get rid of those
    nan_mask = ~np.logical_or(np.isnan(xys), np.isnan(angs))
    nan_mask = np.logical_or.reduce(nan_mask, 1)
    if angs.ndim > 1 and xys.ndim > 1:
        angs = angs[nan_mask, :]
        xys = xys[nan_mask, :]

    n_images = len(images)
    native_area = panel.pixel_area

    # make the tth,eta patches for interpolation
    patches = xrdutil.make_reflection_patches(
        instr_cfg,
        angs,
        panel.angularPixelSize(xys),
        tth_tol=tth_tol,
        eta_tol=eta_tol,
        npdiv=npdiv,
        quiet=True,
    )

    # loop over patches
    # FIXME: fix initialization
    if collapse_tth:
        patch_data = np.zeros((len(angs), n_images))
    else:
        patch_data = []
    for i_p, patch in enumerate(patches):
        # strip relevant objects out of current patch
        vtx_angs, vtx_xys, conn, areas, xys_eval, ijs = patch

        # These areas can be negative if the beam vector is in
        # the opposite direction than it normally is in (positive
        # Z instead of the usual negative Z). Take the absolute
        # value of the areas to ensure they are positive.
        areas = np.abs(areas)

        # need to reshape eval pts for interpolation
        xy_eval = np.vstack([xys_eval[0].flatten(), xys_eval[1].flatten()]).T

        _, on_panel = panel.clip_to_panel(xy_eval)

        if np.any(~on_panel):
            continue

        if collapse_tth:
            ang_data = (vtx_angs[0][0, [0, -1]], vtx_angs[1][[0, -1], 0])
        elif collapse_eta:
            # !!! yield the tth bin centers
            tth_centers = np.average(
                np.vstack([vtx_angs[0][0, :-1], vtx_angs[0][0, 1:]]), axis=0
            )
            ang_data = (tth_centers, angs[i_p][-1])
            if do_fitting:
                fit_data = []
        else:
            ang_data = vtx_angs

        prows, pcols = areas.shape
        area_fac = areas / float(native_area)

        # interpolate
        if not collapse_tth:
            ims_data = []
        for j_p in np.arange(len(images)):
            # catch interpolation type
            image = images[j_p]
            if do_interpolation:
                p_img = (
                    panel.interpolate_bilinear(
                        xy_eval,
                        image,
                    ).reshape(prows, pcols)
                    * area_fac
                )
            else:
                p_img = image[ijs[0], ijs[1]] * area_fac

            # catch flat spectrum data, which will cause
            # fitting to fail.
            # ???: best here, or make fitting handle it?
            mxval = np.max(p_img)
            mnval = np.min(p_img)
            if mxval == 0 or (1.0 - mnval / mxval) < 0.01:
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
                                        [
                                            np.radians(this_tth0),
                                            np.tile(
                                                ang_data[-1], len(this_tth0)
                                            ),
                                        ]
                                    ).T
                                ),
                                return_nominal=True,
                            )
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


class BufferShapeMismatchError(RuntimeError):
    # This is raised when the buffer shape does not match the detector shape
    pass


@contextmanager
def switch_xray_source(instr: HEDMInstrument, xray_source: Optional[str]):
    if xray_source is None:
        # If the x-ray source is None, leave it as the current active one
        yield
        return

    prev_beam_name = instr.active_beam_name
    instr.active_beam_name = xray_source
    try:
        yield
    finally:
        instr.active_beam_name = prev_beam_name
