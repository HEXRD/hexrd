#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:29:50 2022

@author: jbernier
"""
import copy
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from hexrd import constants as ct
from hexrd.instrument import Detector
from hexrd.transforms import xfcapi
from hexrd.utils.concurrent import distribute_tasks


class SampleLayerDistortion:
    def __init__(self, panel,
                 layer_standoff, layer_thickness,
                 pinhole_thickness, pinhole_radius):
        self._panel = panel
        self._layer_standoff = layer_standoff
        self._layer_thickness = layer_thickness
        self._pinhole_thickness = pinhole_thickness
        self._pinhole_radius = pinhole_radius

    @property
    def panel(self):
        return self._panel

    @panel.setter
    def panel(self, x):
        assert isinstance(x, Detector), "input must be a detector"
        self._panel = x

    @property
    def layer_standoff(self):
        return self._layer_standoff

    @layer_standoff.setter
    def layer_standoff(self, x):
        self._layer_standoff = float(x)

    @property
    def layer_thickness(self):
        return self._layer_thickness

    @layer_thickness.setter
    def layer_thickness(self, x):
        self._layer_thickness = float(x)

    @property
    def pinhole_thickness(self):
        return self._pinhole_thickness

    @pinhole_thickness.setter
    def pinhole_thickness(self, x):
        self._pinhole_thickness = float(x)

    @property
    def pinhole_radius(self):
        return self._pinhole_radius

    @pinhole_radius.setter
    def pinhole_radius(self, x):
        self._pinhole_radius = float(x)

    def apply(self, xy_pts, return_nominal=True):
        """
        """
        return tth_corr_sample_layer(self.panel, xy_pts,
                                     self.layer_standoff, self.layer_thickness,
                                     self.pinhole_thickness,
                                     self.pinhole_radius,
                                     return_nominal=return_nominal)


class JHEPinholeDistortion:
    def __init__(self, panel,
                 pinhole_thickness, pinhole_radius):
        self._panel = panel
        self._pinhole_thickness = pinhole_thickness
        self._pinhole_radius = pinhole_radius

    @property
    def panel(self):
        return self._panel

    @panel.setter
    def panel(self, x):
        assert isinstance(x, Detector), "input must be a detector"
        self._panel = x

    @property
    def pinhole_thickness(self):
        return self._pinhole_thickness

    @pinhole_thickness.setter
    def pinhole_thickness(self, x):
        self._pinhole_thickness = float(x)

    @property
    def pinhole_radius(self):
        return self._pinhole_radius

    @pinhole_radius.setter
    def pinhole_radius(self, x):
        self._pinhole_radius = float(x)

    def apply(self, xy_pts, return_nominal=True):
        """
        """
        return tth_corr_pinhole(self.panel, xy_pts,
                                self.pinhole_thickness, self.pinhole_radius,
                                return_nominal=return_nominal)


# Make an alias to the name for backward compatibility
PinholeDistortion = JHEPinholeDistortion


class RyggPinholeDistortion:
    def __init__(self, panel, absorption_length,
                 pinhole_thickness, pinhole_radius, num_phi_elements=60):

        self.panel = panel
        self.absorption_length = absorption_length
        self.pinhole_thickness = pinhole_thickness
        self.pinhole_radius = pinhole_radius
        self.num_phi_elements = num_phi_elements

    def apply(self, xy_pts, return_nominal=True):
        return tth_corr_rygg_pinhole(self.panel, self.absorption_length,
                                     xy_pts, self.pinhole_thickness,
                                     self.pinhole_radius,
                                     return_nominal=return_nominal,
                                     num_phi_elements=self.num_phi_elements)


def tth_corr_sample_layer(panel, xy_pts,
                          layer_standoff, layer_thickness,
                          pinhole_thickness, pinhole_radius,
                          return_nominal=True):
    """
    Compute the Bragg angle distortion associated with a specific sample
    layer in a pinhole camera.

    Parameters
    ----------
    panel : hexrd.instrument.Detector
        A panel instance.
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
    pinhole_radius : scalar
        The radius of the pinhole in mm.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Compute the critical beta angle. Anything past this is invalid.
    critical_beta = np.arctan(2 * pinhole_radius / pinhole_thickness)

    source_distance = panel.xrs_dist

    xy_pts = np.atleast_2d(xy_pts)

    # !!! full z offset from center of pinhole to center of layer
    zs = layer_standoff + 0.5*layer_thickness + 0.5*pinhole_thickness

    ref_angs, _ = panel.cart_to_angles(xy_pts,
                                       rmat_s=None, tvec_s=None,
                                       tvec_c=None, apply_distortion=True)
    ref_tth = ref_angs[:, 0]

    dhats = xfcapi.unit_vector(panel.cart_to_dvecs(xy_pts))
    cos_beta = -dhats[:, 2]
    # Invalidate values past the critical beta
    cos_beta[np.arccos(cos_beta) > critical_beta] = np.nan
    cos_tthn = np.cos(ref_tth)
    sin_tthn = np.sin(ref_tth)
    tth_corr = np.arctan(sin_tthn/(source_distance*cos_beta/zs - cos_tthn))
    if return_nominal:
        return np.vstack([ref_tth - tth_corr, ref_angs[:, 1]]).T
    else:
        # !!! NEED TO CHECK THIS
        return np.vstack([-tth_corr, ref_angs[:, 1]]).T


def invalidate_past_critical_beta(panel: Detector, xy_pts: np.ndarray,
                                  pinhole_thickness: float,
                                  pinhole_radius: float) -> None:
    """Set any xy_pts past critical beta to be nan"""
    # Compute the critical beta angle. Anything past this is invalid.
    critical_beta = np.arctan(2 * pinhole_radius / pinhole_thickness)
    dhats = xfcapi.unit_vector(panel.cart_to_dvecs(xy_pts))
    cos_beta = -dhats[:, 2]
    xy_pts[np.arccos(cos_beta) > critical_beta] = np.nan


def tth_corr_map_sample_layer(instrument,
                              layer_standoff, layer_thickness,
                              pinhole_thickness, pinhole_radius):
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
    pinhole_radius : scalar
        The radius of the pinhole in mm.

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
    # We currently don't invalidate tth values after the critical beta for
    # this map, because it actually looks okay for warping to the polar
    # view. But that is something we could do in the future:
    # critical_beta = np.arctan(2 * pinhole_radius / pinhole_thickness)

    zs = layer_standoff + 0.5*layer_thickness + 0.5*pinhole_thickness
    tth_corr = dict.fromkeys(instrument.detectors)
    for det_key, panel in instrument.detectors.items():
        ref_ptth, _ = panel.pixel_angles()
        py, px = panel.pixel_coords
        xy_data = np.vstack((px.flatten(), py.flatten())).T
        dhats = xfcapi.unit_vector(panel.cart_to_dvecs(xy_data))
        cos_beta = -dhats[:, 2]
        # Invalidate values past the critical beta
        # cos_beta[np.arccos(cos_beta) > critical_beta] = np.nan
        cos_tthn = np.cos(ref_ptth.flatten())
        sin_tthn = np.sin(ref_ptth.flatten())
        tth_corr[det_key] = np.arctan(
            sin_tthn/(instrument.source_distance*cos_beta/zs - cos_tthn)
        ).reshape(panel.shape)
    return tth_corr


def tth_corr_pinhole(panel, xy_pts,
                     pinhole_thickness, pinhole_radius,
                     return_nominal=True):
    """
    Compute the Bragg angle distortion associated with the pinhole as a source.

    Parameters
    ----------
    panel : hexrd.instrument.Detector
        A detector instance.
    xy_pts : array_like
        The (n, 2) array of n (x, y) coordinates to be transformed in the raw
        detector coordinates (cartesian plane, origin at center).
    pinhole_thickness : scalar
        The thickenss (height) of the pinhole (cylinder) in mm
    pinhole_radius : scalar
        The radius of the pinhole in mm.

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
    cp_det = copy.deepcopy(panel)
    cp_det.bvec = ct.beam_vec  # !!! [0, 0, -1]
    ref_angs, _ = cp_det.cart_to_angles(
        xy_pts,
        rmat_s=None, tvec_s=None,
        tvec_c=None, apply_distortion=True
    )
    ref_eta = ref_angs[:, 1]

    # These are the nominal tth values
    nom_angs, _ = panel.cart_to_angles(
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
        angs, _ = panel.cart_to_angles(np.atleast_2d(pxy), tvec_c=origin)
        pin_tth[i] = angs[:, 0]
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
    for det_key, panel in instrument.detectors.items():
        ref_ptth, ref_peta = cp_instr.detectors[det_key].pixel_angles()
        nom_ptth, _ = panel.pixel_angles()

        dpy, dpx = panel.pixel_coords
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
            angs, _ = panel.cart_to_angles(np.atleast_2d(pxy), tvec_c=origin)
            new_ptth[i] = angs[:, 0]
        tth_corr[det_key] = new_ptth.reshape(panel.shape) - nom_ptth
    return tth_corr


def calc_phi_x(bvec, eHat_l):
    """
    returns phi_x in RADIANS
    """
    bv = np.array(bvec)
    bv[2] = 0.
    bv_norm = np.linalg.norm(bv)
    if np.isclose(bv_norm, 0):
        return 0.
    else:
        bv = bv / bv_norm
        return np.arccos(np.dot(bv, -eHat_l)).item()


def azimuth(vv, v0, v1):
    """Return azimuthal angle btwn vv and v0, with v1 defining phi=0.

    Originally written by Ryan Rygg. This is a modified version.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        n0 = np.cross(v0, v1)
        n0 /= np.linalg.norm(n0, axis=-1)[..., np.newaxis]
        nn = np.cross(v0, vv)
        nn /= np.linalg.norm(nn, axis=-1)[..., np.newaxis]

    azi = np.arccos(np.sum(nn * n0, -1))
    if len(np.shape(azi)) > 0:
        azi[np.dot(vv, n0) < 0] *= -1
        # arbitrary angle where vv is (anti)parallel to v0
        azi[np.isnan(azi)] = 0
    elif np.isnan(azi):
        return 0
    elif np.dot(vv, v0) < 1 and azi > 0:
        azi *= -1

    return azi


def _infer_instrument_type(panel):
    tardis_names = [
        'IMAGE-PLATE-1',
        'IMAGE-PLATE-2',
        'IMAGE-PLATE-3',
        'IMAGE-PLATE-4',
    ]

    pxrdip_names = [
        'IMAGE-PLATE-B',
        'IMAGE-PLATE-D',
        'IMAGE-PLATE-L',
        'IMAGE-PLATE-R',
        'IMAGE-PLATE-U',
    ]

    fiddle_names = [
        'CAMERA-02',
        'CAMERA-03',
        'CAMERA-05',
        'CAMERA-07',
        'CAMERA-08',
        'IMAGE-PLATE']

    if panel.name in tardis_names:
        return 'TARDIS'
    elif panel.name in pxrdip_names:
        return 'PXRDIP'
    elif panel.name in fiddle_names:
        return 'FIDDLE'

    raise NotImplementedError(f'Unknown detector name: {panel.name}')


def _infer_eHat_l(panel):
    instr_type = _infer_instrument_type(panel)

    eHat_l_dict = {
        'TARDIS': -ct.lab_x.reshape((3, 1)),
        'PXRDIP': -ct.lab_x.reshape((3, 1)),
        'FIDDLE': ct.lab_x.reshape((3, 1))
    }

    return eHat_l_dict[instr_type]


def _infer_eta_shift(panel):
    instr_type = _infer_instrument_type(panel)

    eta_shift_dict = {
        'TARDIS': -np.radians(180),
        'PXRDIP': -np.radians(180),
        'FIDDLE': 0.0,
    }

    return eta_shift_dict[instr_type]


def calc_tth_rygg_pinhole(panels, absorption_length, tth, eta,
                          pinhole_thickness, pinhole_radius,
                          num_phi_elements=60, clip_to_panel=True):
    """Return pinhole twotheta [rad] and effective scattering volume [mm3].

    num_phi_elements: number of pinhole phi elements for integration
    """
    # Make sure these are at least 2D
    original_shape = tth.shape

    if len(tth.shape) == 1:
        # Do these instead of atleast_2d(), because we want the 1
        # on the column, not on the row.
        tth = tth.reshape(tth.shape[0], 1)
        eta = eta.reshape(eta.shape[0], 1)

    if not isinstance(panels, (list, tuple)):
        panels = [panels]

    # ------ Determine geometric parameters ------

    # Grab info from the first panel that should be same across all panels
    first_panel = panels[0]
    bvec = first_panel.bvec
    eHat_l = _infer_eHat_l(first_panel)
    max_workers = first_panel.max_workers

    eta_shift = _infer_eta_shift(first_panel)
    # This code expects eta to be shifted by a certain amount
    # !! do not modify the original eta array
    eta = eta + eta_shift

    # distance of xray source from origin (i. e., center of pinhole) [mm]
    r_x = first_panel.xrs_dist

    # zenith angle of the x-ray source from (negative) pinhole axis
    alpha = np.arccos(np.dot(bvec, [0, 0, -1]))

    # azimuthal angle of the x-ray source around the pinhole axis
    phi_x = calc_phi_x(bvec, eHat_l)

    # pinhole substrate thickness [mm]
    h_p = pinhole_thickness

    # pinhole aperture diameter [mm]
    d_p = pinhole_radius * 2

    # mu_p is the attenuation coefficent [um^-1]
    # This is the inverse of the absorption length, which is in [um]
    mu_p = 1 / absorption_length
    mu_p = 1000 * mu_p  # convert to [mm^-1]

    # Convert tth and eta to phi_d, beta, and r_d
    dvec_arg = np.vstack((tth.flatten(), eta.flatten(),
                          np.zeros(np.prod(eta.shape))))
    dvectors = xfcapi.angles_to_dvec(dvec_arg.T, bvec, eta_vec=eHat_l)

    v0 = np.array([0, 0, 1])
    v1 = np.squeeze(eHat_l)
    phi_d = azimuth(dvectors, -v0, v1).reshape(tth.shape)
    beta = np.arccos(-dvectors[:, 2]).reshape(tth.shape)

    # Compute r_d
    # We will first convert to Cartesian, then clip to the panel, add the
    # extra Z dimension, apply the rotation matrix, add the tvec, and then
    # compute the distance.
    angles_full = np.stack((tth, eta)).reshape((2, np.prod(tth.shape))).T

    r_d = np.full(tth.shape, np.nan)
    for panel in panels:
        try:
            # Set the evec to eHat_l while converting to cartesian
            # This is important so that the r_d values end up in the right
            # spots
            old_evec = panel.evec
            panel.evec = eHat_l
            cart = panel.angles_to_cart(angles_full)
        finally:
            panel.evec = old_evec

        if clip_to_panel:
            _, on_panel = panel.clip_to_panel(cart)
            cart[~on_panel] = np.nan

        dvecs = panel.cart_to_dvecs(cart)
        full_dvecs = dvecs.T.reshape(3, *tth.shape).T
        panel_r_d = np.sqrt(np.sum((full_dvecs)**2, axis=2)).T

        # Only overwrite positions that are still nan on r_d
        r_d[np.isnan(r_d)] = panel_r_d[np.isnan(r_d)]

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

    alpha_i = np.arctan2(np.sqrt(sin_a**2 + 2*bx*sin_a*cos_dphi_x + bx**2),
                         cos_a + z_i/r_x)
    phi_xi = np.arctan2(sin_a * np.sin(phi_x) - bx*sin_phii,
                        sin_a * np.cos(phi_x) - bx * cos_phii)

    # !!! This section used 4D arrays before, which was very time consuming
    # for large grids. Instead, we now loop over the columns and do them
    # one at a time. For large arrays, this saves a huge amount of time and
    # memory.
    tasks = distribute_tasks(phi_d.shape[3], max_workers)
    func = partial(
        _run_compute_qq_p,
        # 4D variables (all have some axes == 1)
        phi_d=phi_d,
        sin_b=sin_b,
        bd=bd,
        cos_b=cos_b,
        tan_b=tan_b,
        r_d=r_d,
        # The variables with less than 4 dimensions
        sin_phii=sin_phii,
        cos_phii=cos_phii,
        alpha_i=alpha_i,
        phi_xi=phi_xi,
        sin_a=sin_a,
        cos_dphi_x=cos_dphi_x,
        cos_a=cos_a,
        dV_s=dV_s,
        dV_e=dV_e,
        z_i=z_i,
        h_p=h_p,
        d_p=d_p,
        tan_a=tan_a,
        phi_i=phi_i,
    )

    if len(tasks) == 1:
        # Don't use a thread pool if there is just one task
        # Also don't use numba, it appears to be slower for the
        # serial version.
        results = []
        for task in tasks:
            results.append(func(task, use_numba=False))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(func, tasks)

    i = 0
    qq_p = np.full(tth.shape, np.nan)
    for worker_output in results:
        for output in worker_output:
            qq_p[:, [i]] = output
            i += 1

    # Set the nan values back to nan so it is clear they were not computed
    qq_p[is_nan] = np.nan

    return qq_p.reshape(original_shape)


def _run_compute_qq_p(indices, *args, **kwargs):
    # We will reduce these to one column to avoid making the full 4D arrays
    reduce_column_vars = [
        'phi_d',
        'sin_b',
        'bd',
        'cos_b',
        'tan_b',
        'r_d',
    ]
    original_kwargs = kwargs.copy()

    output = []
    for i in range(*indices):
        for var_name in reduce_column_vars:
            kwargs[var_name] = original_kwargs[var_name][:, :, :, [i]]

        output.append(_compute_qq_p(*args, **kwargs))

    return output


def _compute_qq_p(use_numba=True, *args, **kwargs):
    # This function is separate from the `_compute_vi_qq_i` one because
    # this does the np.nansum(..., axis=(0, 1)), which numba cannot do.
    if use_numba:
        # The numbafied version is faster if we are computing a grid
        f = _compute_vi_qq_i_numba
    else:
        # The non-numbafied version is faster if we don't have a grid.
        f = _compute_vi_qq_i

    V_i, qq_i = f(*args, **kwargs)
    V_p = np.nansum(V_i, axis=(0, 1))  # [Nu x Nv] <= detector

    with np.errstate(divide='ignore', invalid='ignore'):
        # Ignore the errors this will inevitably produce
        return np.nansum(V_i * qq_i,
                         axis=(0, 1)) / V_p  # [Nu x Nv] <= detector


def _compute_vi_qq_i(phi_d, sin_b, bd, sin_phii, cos_phii, alpha_i, phi_xi,
                     sin_a, cos_dphi_x, cos_a, cos_b, dV_s, dV_e, z_i, h_p,
                     d_p, tan_b, tan_a, phi_i, r_d):
    # This function can be numbafied, and has a numbafied version below.

    # Compute V_i and qq_i

    cos_dphi_d = np.cos(phi_i - phi_d + np.pi)
    beta_i = np.arctan2(np.sqrt(sin_b**2 + 2*bd*sin_b*cos_dphi_d + bd**2),
                        cos_b - z_i/r_d)

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
    tan_eta_d = np.where(cos_dphi_d[-1] <= 0, 0, cos_b[0] * cos_dphi_d[-1])

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
    return V_i, qq_i


# The numba version (works better in conjunction with multi-threading)
_compute_vi_qq_i_numba = njit(
    nogil=True, cache=True)(_compute_vi_qq_i)


def tth_corr_rygg_pinhole(panel, absorption_length, xy_pts,
                          pinhole_thickness, pinhole_radius,
                          return_nominal=True, num_phi_elements=60):
    # These are the nominal tth values
    nom_angs, _ = panel.cart_to_angles(
        xy_pts,
        rmat_s=None, tvec_s=None,
        tvec_c=None, apply_distortion=True
    )
    nom_tth, nom_eta = nom_angs[:, :2].T

    # Don't clip these values to the panel because they will be shifted
    qq_p = calc_tth_rygg_pinhole(
        panel, absorption_length, nom_tth, nom_eta, pinhole_thickness,
        pinhole_radius, num_phi_elements, clip_to_panel=False)

    # Make the distortion shift to the left instead of the right
    # FIXME: why is qq_p shifting the data to the right instead of the left?
    qq_p = nom_tth - (qq_p - nom_tth)

    angs = np.vstack([qq_p, nom_eta]).T
    new_xy_pts = panel.angles_to_cart(angs)
    # Clip these to the panel now
    _, on_panel = panel.clip_to_panel(new_xy_pts)
    angs[~on_panel] = np.nan

    if return_nominal:
        return angs
    else:
        angs[:, 0] -= nom_tth
        return angs


def tth_corr_map_rygg_pinhole(instrument, absorption_length, pinhole_thickness,
                              pinhole_radius, num_phi_elements=60):
    tth_corr = {}
    for det_key, panel in instrument.detectors.items():
        nom_ptth, nom_peta = panel.pixel_angles()
        qq_p = calc_tth_rygg_pinhole(
            panel, absorption_length, nom_ptth, nom_peta, pinhole_thickness,
            pinhole_radius, num_phi_elements)
        tth_corr[det_key] = nom_ptth - qq_p
    return tth_corr


def polar_tth_corr_map_rygg_pinhole(tth, eta, instrument, absorption_length,
                                    pinhole_thickness, pinhole_radius,
                                    num_phi_elements=60):
    """Generate a polar tth corr map directly for all panels"""
    panels = list(instrument.detectors.values())
    return calc_tth_rygg_pinhole(panels, absorption_length, tth, eta,
                                 pinhole_thickness, pinhole_radius,
                                 num_phi_elements) - tth
