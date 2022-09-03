#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:29:50 2022

@author: jbernier
"""
import numpy as np

from hexrd.instrument import PlanarDetector
from hexrd.transforms import xfcapi

detector_classes = (PlanarDetector, )


class SampleLayerDistortion(object):
    def __init__(self, detector,
                 layer_standoff, layer_thickness,
                 pinhole_thickness, source_distance):
        self._detector = detector
        self._standoff = layer_standoff
        self._thickness = layer_thickness
        self._ph_thickness = pinhole_thickness
        self._source_dist = source_distance

    @property
    def detector(self):
        return self._detector

    @detector.setter
    def detector(self, x):
        assert isinstance(x, detector_classes), \
            f"input must be one of {detector_classes}"
        self._detector = x

    @property
    def standoff(self):
        return self._standoff

    @standoff.setter
    def standoff(self, x):
        self._standoff = float(x)

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, x):
        self._thickness = float(x)

    @property
    def ph_thickness(self):
        return self._ph_thickness

    @ph_thickness.setter
    def ph_thickness(self, x):
        self._ph_thickness = float(x)

    @property
    def source_dist(self):
        return self._source_dist

    @source_dist.setter
    def source_dist(self, x):
        self._source_dist = float(x)

    def apply(self, xy_pts, return_nominal=True):
        """
        """
        return tth_corr_sample_layer(self.detector, xy_pts,
                                     self.standoff, self.thickness,
                                     self.ph_thickness, self.source_dist,
                                     return_nominal=return_nominal)


def tth_corr_sample_layer(detector, xy_pts,
                          layer_standoff, layer_thickness,
                          pinhole_thickness, source_distance,
                          return_nominal=True):
    """
    Compute the Bragg angle distortion associated with a specific sample
    layer in a pinhole camera.

    Parameters
    ----------
    detector : hexrd.instrument.PlanarDetector
        A detector instance.
    xy_pts : array_like
        The (n, 2) array of n (x, y) coordinates to be transformed in the raw
        detector coordinates (cartesian plane, origin at center).
    layer_standoff : scalar
        The sample layer standoff from the upstream face of the pinhole
        in microns.
    layer_thickness : scalar
        The thickness of the sample layer in microns.
    pinhole_thickness : scalar
        The thickenss (height) of the pinhole (cylinder) in microns
    source_distance : scalar
        he distance from the pinhole center to the X-ray source in microns.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    xy_pts = np.atleast_2d(xy_pts)
    npts = len(xy_pts)

    # !!! full z offset from center of pinhole to center of layer
    zs = layer_standoff + 0.5*layer_thickness + 0.5*pinhole_thickness

    ref_angs, _ = detector.cart_to_angles(xy_pts,
                                          rmat_s=None, tvec_s=None,
                                          tvec_c=None, apply_distortion=True)
    ref_tth = ref_angs[:, 0]

    crds = np.hstack([xy_pts, np.zeros((npts, 1))])
    dhats = xfcapi.unitRowVector(
        np.dot(crds, detector.rmat.T) + detector.tvec
    )
    cos_beta = -dhats[:, 2]
    cos_tthn = np.cos(ref_tth)
    sin_tthn = np.sin(ref_tth)
    tth_corr = np.arctan(sin_tthn/(source_distance*cos_beta/zs - cos_tthn))
    if return_nominal:
        return np.vstack([ref_tth - tth_corr, ref_angs[:, 1]]).T
    else:
        # !!! NEED TO CHECK THIS
        return np.vstack([-tth_corr, ref_angs[:, 1]]).T


def tth_corr_map_sample_layer(instrument,
                              layer_standoff, layer_thickness,
                              pinhole_thickness):
    """
    Compute the Bragg angle distortion fields for an instrument associated
    with a specific sample layer in a pinhole camera.

    Parameters
    ----------
    instrument : hexrd.instrument.HEDMInstrument
        The pionhole camera instrument object.
    layer_standoff : scalar
        The sample layer standoff from the upstream face of the pinhole
        in microns.
    layer_thickness : scalar
        The thickness of the sample layer in microns.
    pinhole_thickness : scalar
        The thickenss (height) of the pinhole (cylinder) in microns

    Returns
    -------
    tth_corr : dict
        The Bragg angle correction fields for each detector in `instrument`
        as 2θ_sam - 2θ_nom in radians.

    Notes
    -----
    source_distance : The distance from the pinhole center to
                      the X-ray source in microns.  Comes from the instr
                      attribute of the same name.

    """
    zs = layer_standoff + 0.5*layer_thickness + 0.5*pinhole_thickness
    tth_corr = dict.fromkeys(instrument.detectors)
    for det_key, det in instrument.detectors.items():
        ref_ptth, _ = det.pixel_angles()
        py, px = det.pixel_coords
        crds = np.vstack([px.flatten(), py.flatten(), np.zeros(px.size)])
        dhats = xfcapi.unitRowVector(
            (np.dot(det.rmat, crds) + det.tvec.reshape(3, 1)).T
        )
        cos_beta = -dhats[:, 2]
        cos_tthn = np.cos(ref_ptth.flatten())
        sin_tthn = np.sin(ref_ptth.flatten())
        tth_corr[det_key] = np.arctan(
            sin_tthn/(instrument.source_distance*cos_beta/zs - cos_tthn)
        ).reshape(det.shape)
    return tth_corr
