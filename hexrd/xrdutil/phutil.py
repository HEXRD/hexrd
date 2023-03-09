#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:29:50 2022

@author: jbernier
"""
import copy

import numpy as np

from hexrd import constants as ct
from hexrd.instrument import calc_angles_from_beam_vec, PlanarDetector
from hexrd.transforms import xfcapi

detector_classes = (PlanarDetector, )


class SampleLayerDistortion:
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


class PinholeDistortion:
    def __init__(self, detector,
                 pinhole_thickness, pinhole_radius):
        self._detector = detector
        self._ph_thickness = pinhole_thickness
        self._ph_radius = pinhole_radius

    @property
    def detector(self):
        return self._detector

    @detector.setter
    def detector(self, x):
        assert isinstance(x, detector_classes), \
            f"input must be one of {detector_classes}"
        self._detector = x

    @property
    def ph_thickness(self):
        return self._ph_thickness

    @ph_thickness.setter
    def ph_thickness(self, x):
        self._ph_thickness = float(x)

    @property
    def ph_radius(self):
        return self._ph_radius

    @ph_radius.setter
    def ph_radius(self, x):
        self._ph_radius = float(x)

    def apply(self, xy_pts, return_nominal=True):
        """
        """
        return tth_corr_pinhole(self.detector, xy_pts,
                                self.ph_thickness, self.ph_radius,
                                return_nominal=return_nominal)


class RyggPinholeDistortion:
    def __init__(self, detector, material,
                 pinhole_thickness, pinhole_radius, num_phi_elements=120):

        self.detector = detector
        self.material = material
        self.ph_thickness = pinhole_thickness
        self.ph_radius = pinhole_radius
        self.num_phi_elements = num_phi_elements

    def apply(self, xy_pts, return_nominal=True):
        return tth_corr_rygg_pinhole(self.detector, self.material, xy_pts,
                                     self.ph_thickness, self.ph_radius,
                                     return_nominal=return_nominal,
                                     num_phi_elements=self.num_phi_elements)


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
        in mm.
    layer_thickness : scalar
        The thickness of the sample layer in mm.
    pinhole_thickness : scalar
        The thickenss (height) of the pinhole (cylinder) in mm
    source_distance : scalar
        The distance from the pinhole center to the X-ray source in mm.

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
        in mm.
    layer_thickness : scalar
        The thickness of the sample layer in mm.
    pinhole_thickness : scalar
        The thickenss (height) of the pinhole (cylinder) in mm

    Returns
    -------
    tth_corr : dict
        The Bragg angle correction fields for each detector in `instrument`
        as 2θ_sam - 2θ_nom in radians.

    Notes
    -----
    source_distance : The distance from the pinhole center to
                      the X-ray source in mm.  Comes from the instr
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


def tth_corr_pinhole(detector, xy_pts,
                     pinhole_thickness, pinhole_radius,
                     return_nominal=True):
    """
    Compute the Bragg angle distortion associated with the pinhole as a source.

    Parameters
    ----------
    detector : hexrd.instrument.PlanarDetector
        A detector instance.
    xy_pts : array_like
        The (n, 2) array of n (x, y) coordinates to be transformed in the raw
        detector coordinates (cartesian plane, origin at center).
    pinhole_thickness : scalar
        The thickenss (height) of the pinhole (cylinder) in mm
    pinhole_radius : scalar
        The radius ofhte pinhole in mm.

    Returns
    -------
    TYPE
        DESCRIPTION.

    Notes
    -----
    The follows a slightly modified version of Jon Eggert's pinhole correction.

    """

    xy_pts = np.atleast_2d(xy_pts)
    npts = len(xy_pts)

    # first we need the reference etas of the points wrt the pinhole axis
    cp_det = copy.deepcopy(detector)
    cp_det.bvec = ct.beam_vec  # !!! [0, 0, -1]
    ref_angs, _ = cp_det.cart_to_angles(
        xy_pts,
        rmat_s=None, tvec_s=None,
        tvec_c=None, apply_distortion=True
    )
    ref_eta = ref_angs[:, 1]

    # These are the nominal tth values
    nom_angs, _ = detector.cart_to_angles(
        xy_pts,
        rmat_s=None, tvec_s=None,
        tvec_c=None, apply_distortion=True
    )
    nom_tth = nom_angs[:, 0]

    pin_tth = np.zeros(npts)
    for i, (pxy, reta) in enumerate(zip(xy_pts, ref_eta)):
        # !!! JHE used pinhole center, but the back surface
        #     seems to hew a bit closer to JRR's solution
        origin = -pinhole_radius*np.array(
            [np.cos(reta), np.sin(reta), 0.5*pinhole_thickness]
        )
        angs, _ = xfcapi.detectorXYToGvec(
            np.atleast_2d(pxy), detector.rmat, ct.identity_3x3,
            detector.tvec, ct.zeros_3, origin,
            beamVec=detector.bvec,
            etaVec=detector.evec)
        pin_tth[i] = angs[0]
    tth_corr = pin_tth - nom_tth
    if return_nominal:
        return np.vstack([nom_tth - tth_corr, nom_angs[:, 1]]).T
    else:
        # !!! NEED TO CHECK THIS
        return np.vstack([-tth_corr, nom_angs[:, 1]]).T


def tth_corr_map_pinhole(instrument, pinhole_thickness, pinhole_radius):
    """
    Compute the Bragg angle distortion fields for pinhole diffraction.

    Parameters
    ----------
    instrument : hexrd.instrument.HEDMInstrument
        The pionhole camera instrument object.
    pinhole_thickness : scalar
        The thickenss (height) of the pinhole (cylinder) in mm
    pinhole_radius : scalar
        The radius of the pinhole in mm

    Returns
    -------
    tth_corr : dict
        The Bragg angle correction fields for each detector in `instrument`
        as 2θ_pin - 2θ_nom in radians.

    Notes
    -----
    The follows a slightly modified version of Jon Eggert's pinhole correction.
    """
    cp_instr = copy.deepcopy(instrument)
    cp_instr.beam_vector = ct.beam_vec  # !!! [0, 0, -1]

    tth_corr = dict.fromkeys(instrument.detectors)
    for det_key, det in instrument.detectors.items():
        ref_ptth, ref_peta = cp_instr.detectors[det_key].pixel_angles()
        nom_ptth, _ = det.pixel_angles()

        dpy, dpx = det.pixel_coords
        pcrds = np.ascontiguousarray(
            np.vstack([dpx.flatten(), dpy.flatten()]).T
        )
        ref_peta = ref_peta.flatten()

        new_ptth = np.zeros(len(ref_peta))
        for i, (pxy, reta) in enumerate(zip(pcrds, ref_peta)):
            # !!! JHE used pinhole center, but the back surface
            #     seems to hew a bit closer to JRR's solution
            origin = -pinhole_radius*np.array(
                [np.cos(reta), np.sin(reta), 0.5*pinhole_thickness]
            )
            angs, g_vec = xfcapi.detectorXYToGvec(
                np.atleast_2d(pxy), det.rmat, ct.identity_3x3,
                det.tvec, ct.zeros_3, origin,
                beamVec=instrument.beam_vector,
                etaVec=instrument.eta_vector)
            new_ptth[i] = angs[0]
        tth_corr[det_key] = new_ptth.reshape(det.shape) - nom_ptth
    return tth_corr


def calc_phi_x(panel):
    """
    returns phi_x in RADIANS
    """
    bv = panel.bvec.copy()
    bv[2] = 0.
    bv = bv/np.linalg.norm(bv)
    return np.arccos(np.dot(bv, [0, -1, 0]))


def calc_tth_rygg_pinhole(panel, material, tth, eta, pinhole_thickness,
                          pinhole_radius, num_phi_elements=120):
    """Return pinhole twotheta [rad] and effective scattering volume [mm3].

    num_phi_elements: number of pinhole phi elements for integration
    """
    # Make sure these are at least 2D
    original_shape = tth.shape
    tth = np.atleast_2d(tth)
    eta = np.atleast_2d(eta)

    # ------ Determine geometric parameters ------

    # distance of xray source from origin (i. e., center of pinhole) [mm]
    r_x = panel.xrs_dist

    # Get our version of these beam angles
    # !!! these are in degrees. Theirs are in radians.
    azim, pola = calc_angles_from_beam_vec(panel.bvec)

    # zenith angle of the x-ray source from (negative) pinhole axis
    alpha = np.arccos(np.dot(panel.bvec, [0, 0, -1]))

    # azimuthal angle of the x-ray source around the pinhole axis
    phi_x = calc_phi_x(panel)

    # pinhole substrate thickness [mm]
    h_p = pinhole_thickness

    # pinhole aperture diameter [mm]
    d_p = pinhole_radius * 2

    # mu_p is the attenuation coefficent [um^-1]
    # This is the inverse of the absorption length, which is in [um]
    mu_p = 1 / material.absorption_length
    mu_p = 1000 * mu_p  # convert to [mm^-1]

    # Convert tth and eta to phi_d, beta, and r_d
    dvec_arg = np.vstack((tth.flatten(), eta.flatten(),
                          np.zeros(np.prod(eta.shape))))
    dvectors = xfcapi.anglesToDVec(dvec_arg.T, panel.bvec)

    phi_d = np.mod(np.arctan2(dvectors[:, 1], dvectors[:, 0]) + 1.5 * np.pi,
                   2 * np.pi).reshape(tth.shape)
    beta = np.arccos(np.dot(dvectors, [0, 0, -1])).reshape(tth.shape)

    # Compute r_d
    # We will first convert to Cartesian, then clip to the panel, add the
    # extra Z dimension, apply the rotation matrix, add the tvec, and then
    # compute the distance.
    angles_full = np.stack((tth, eta)).reshape((2, np.prod(tth.shape))).T
    cart = panel.angles_to_cart(angles_full)
    _, on_panel = panel.clip_to_panel(cart)
    cart[~on_panel] = np.nan
    cart = cart.T.reshape((2, *tth.shape))
    full_cart = np.stack((cart[0], cart[1], np.zeros(tth.shape)))
    flat_coords = full_cart.reshape((3, np.prod(tth.shape)))
    rotated = panel.rmat.dot(flat_coords).reshape(full_cart.shape).T
    full_vector = panel.tvec + rotated
    r_d = np.sqrt(np.sum((full_vector)**2, axis=2)).T

    # Store the nan values so we can use them again later
    is_nan = np.isnan(r_d)

    # reshape so D grids are on axes indices 2 and 3 [1 x 1 x Nu x Nv]
    r_d = np.atleast_2d(r_d)[None, None, :, :]
    beta = np.atleast_2d(beta)[None, None, :, :]
    phi_d = np.atleast_2d(phi_d)[None, None, :, :]

    Np = num_phi_elements

    # ------ define pinhole grid ------
    # approximately square pinhole surface elements
    Nz = max(3, int(Np * h_p / (np.pi * d_p)))
    dphi = 2 * np.pi / Np  # [rad] phi interval
    dl = d_p * dphi  # [mm] azimuthal distance increment
    dz = h_p / Nz  # [mm] axial distance increment
    dA = dz * dl  # [mm^2] area element
    dV_s = dA * mu_p**-1  # [mm^3] volume of surface element
    dV_e = dl * mu_p**-2  # [mm^3] volume of edge element

    phi_vec = np.arange(dphi / 2, 2 * np.pi, dphi)
    # includes elements for X and D edges
    z_vec = np.arange(-h_p/2 - dz/2, h_p/2 + dz/1.999, dz)
    z_vec[0] = -h_p/2  # X-side edge (negative z)
    z_vec[-1] = h_p/2  # D-side edge (positive z)
    phi_i, z_i = np.meshgrid(phi_vec, z_vec)  # [Nz x Np]
    phi_i = phi_i[:, :, None, None]    # [Nz x Np x 1 x 1]
    z_i = z_i[:, :, None, None]      # axes 0,1 => P; axes 2,3 => D

    # ------ calculate twotheta_i [a.k.a. qq_i], for each grid element ------
    bx, bd = (d_p / (2 * r_x),  d_p / (2 * r_d))
    sin_a, cos_a, tan_a = np.sin(alpha), np.cos(alpha), np.tan(alpha)
    sin_b, cos_b, tan_b = np.sin(beta),  np.cos(beta),  np.tan(beta)
    sin_phii, cos_phii = np.sin(phi_i), np.cos(phi_i)
    cos_dphi_x = np.cos(phi_i - phi_x + np.pi)  # [Nz x Np x Nu x Nv]
    cos_dphi_d = np.cos(phi_i - phi_d + np.pi)

    alpha_i = np.arctan2(np.sqrt(sin_a**2 + 2*bx*sin_a*cos_dphi_x + bx**2),
                         cos_a + z_i/r_x)
    beta_i = np.arctan2(np.sqrt(sin_b**2 + 2*bd*sin_b*cos_dphi_d + bd**2),
                        cos_b - z_i/r_d)
    phi_xi = np.arctan2(sin_a * np.sin(phi_x) - bx*sin_phii,
                        sin_a * np.cos(phi_x) - bx * cos_phii)
    phi_di = np.arctan2(sin_b * np.sin(phi_d) - bd*sin_phii,
                        sin_b * np.cos(phi_d) - bd * cos_phii)

    arg = (np.cos(alpha_i) * np.cos(beta_i) - np.sin(alpha_i) *
           np.sin(beta_i) * np.cos(phi_di - phi_xi))
    # scattering angle for each P to each D
    qq_i = np.arccos(np.clip(arg, -1, 1))

    # ------ calculate effective volumes: 1 (surface), 2 (Xedge), 3 (Dedge) ---
    sec_psi_x = 1 / (sin_a * cos_dphi_x)
    sec_psi_d = 1 / (sin_b * cos_dphi_d)
    sec_alpha = 1 / cos_a
    sec_beta = 1 / cos_b
    tan_eta_x = np.where(cos_dphi_x[0] <= 0, 0, cos_a * cos_dphi_x[0])
    tan_eta_d = np.where(cos_dphi_d[-1] <= 0, 0, cos_b * cos_dphi_d[-1])

    V_i = dV_s / (sec_psi_x + sec_psi_d)  # [mm^3]
    # X-side edge (z = -h_p / 2)
    V_i[0] = dV_e / (sec_psi_d[0] * (sec_alpha + sec_psi_d[0] * tan_eta_x))
    # D-side edge (z = +h_p / 2)
    V_i[-1] = dV_e / (sec_psi_x[-1] * (sec_beta + sec_psi_x[-1] * tan_eta_d))

    # ------ visibility of each grid element ------
    # pinhole surface
    is_seen = np.logical_and(z_i > h_p/2 - d_p/tan_b * cos_dphi_d,
                             z_i < -h_p/2 + d_p/tan_a * cos_dphi_x)
    # X-side edge
    is_seen[0] = np.where(h_p/d_p * tan_b < cos_dphi_d[0], 1, 0)
    # D-side edge
    is_seen[-1] = np.where(h_p/d_p * tan_a < cos_dphi_x[-1], 1, 0)

    # ------ weighted sum over elements to obtain average ------
    V_i *= is_seen  # zero weight to elements with no view of both X and D
    V_p = np.nansum(V_i, axis=(0, 1))  # [Nu x Nv] <= detector
    qq_p = np.nansum(V_i * qq_i, axis=(0, 1)) / V_p  # [Nu x Nv] <= detector

    # Set the nan values back to nan so it is clear they were not computed
    qq_p[is_nan] = np.nan

    return qq_p.reshape(original_shape)


def tth_corr_rygg_pinhole(panel, material, xy_pts,
                          pinhole_thickness, pinhole_radius,
                          return_nominal=True, num_phi_elements=120):
    # These are the nominal tth values
    nom_angs, _ = panel.cart_to_angles(
        xy_pts,
        rmat_s=None, tvec_s=None,
        tvec_c=None, apply_distortion=True
    )
    nom_tth, nom_eta = nom_angs[:, :2].T

    qq_p = calc_tth_rygg_pinhole(
        panel, material, nom_tth, nom_eta, pinhole_thickness,
        pinhole_radius, num_phi_elements)

    if return_nominal:
        return np.vstack([qq_p, nom_angs[:, 1]]).T
    else:
        # !!! NEED TO CHECK THIS
        return np.vstack([nom_tth - qq_p, nom_angs[:, 1]]).T


def tth_corr_map_rygg_pinhole(instrument, material, pinhole_thickness,
                              pinhole_radius, num_phi_elements=120):
    tth_corr = {}
    for det_key, panel in instrument.detectors.items():
        nom_ptth, nom_peta = panel.pixel_angles()
        qq_p = calc_tth_rygg_pinhole(
            panel, material, nom_ptth, nom_peta, pinhole_thickness,
            pinhole_radius, num_phi_elements)
        tth_corr[det_key] = qq_p - nom_ptth
    return tth_corr
