import numpy as np

from hexrd import constants as ct
from hexrd.transforms.xfcapi import (
    anglesToGVec,
    detectorXYToGvec,
    gvecToDetectorXY,
)
from hexrd.utils.decorators import memoize

from .detector import Detector


class PlanarDetector(Detector):
    """Base class for 2D planar, rectangular row-column detector"""

    def __init__(self, **detector_kwargs):
        super().__init__(**detector_kwargs)

    def detector_type(self):
        return 'planar'

    def cart_to_angles(self, xy_data,
                       rmat_s=None,
                       tvec_s=None, tvec_c=None,
                       apply_distortion=False):
        if rmat_s is None:
            rmat_s = ct.identity_3x3
        if tvec_s is None:
            tvec_s = ct.zeros_3
        if tvec_c is None:
            tvec_c = ct.zeros_3
        if apply_distortion and self.distortion is not None:
            xy_data = self.distortion.apply(xy_data)
        angs, g_vec = detectorXYToGvec(
            xy_data, self.rmat, rmat_s,
            self.tvec, tvec_s, tvec_c,
            beamVec=self.bvec, etaVec=self.evec)
        tth_eta = np.vstack([angs[0], angs[1]]).T
        return tth_eta, g_vec

    def angles_to_cart(self, tth_eta,
                       rmat_s=None, tvec_s=None,
                       rmat_c=None, tvec_c=None,
                       apply_distortion=False):
        if rmat_s is None:
            rmat_s = ct.identity_3x3
        if tvec_s is None:
            tvec_s = ct.zeros_3
        if rmat_c is None:
            rmat_c = ct.identity_3x3
        if tvec_c is None:
            tvec_c = ct.zeros_3

        # get chi and ome from rmat_s
        # !!! WARNING: API ambiguity
        # !!! this assumes rmat_s was made from the composition
        # !!! rmat_s = R(Xl, chi) * R(Yl, ome)
        chi = np.arccos(rmat_s[1, 1])
        ome = np.arccos(rmat_s[0, 0])

        angs = np.hstack([tth_eta, np.tile(ome, (len(tth_eta), 1))])
        xy_det = gvecToDetectorXY(
            anglesToGVec(angs, bHat_l=self.bvec, eHat_l=self.evec, chi=chi),
            self.rmat, rmat_s, rmat_c,
            self.tvec, tvec_s, tvec_c,
            beamVec=self.bvec)
        if apply_distortion and self.distortion is not None:
            xy_det = self.distortion.apply_inverse(xy_det)
        return xy_det

    def pixel_angles(self, origin=ct.zeros_3):
        return _pixel_angles(origin, self.pixel_coords, self.distortion,
                             self.rmat, self.tvec, self.bvec, self.evec,
                             self.rows, self.cols)

    def pixel_tth_gradient(self, origin=ct.zeros_3):
        return _pixel_tth_gradient(origin, self.pixel_coords, self.distortion,
                                   self.rmat, self.tvec, self.bvec, self.evec,
                                   self.rows, self.cols)

    def pixel_eta_gradient(self, origin=ct.zeros_3):
        return _pixel_eta_gradient(origin, self.pixel_coords, self.distortion,
                                   self.rmat, self.tvec, self.bvec, self.evec,
                                   self.rows, self.cols)

    @property
    def beam_position(self):
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

    @staticmethod
    def update_memoization_sizes(all_panels):
        Detector.update_memoization_sizes(all_panels)

        num_matches = sum(x is PlanarDetector for x in all_panels)
        funcs = [
            _pixel_angles,
            _pixel_tth_gradient,
            _pixel_eta_gradient,
        ]
        Detector.increase_memoization_sizes(funcs, num_matches)


@memoize
def _pixel_angles(origin, pixel_coords, distortion, rmat, tvec, bvec, evec,
                  rows, cols):
    assert len(origin) == 3, "origin must have 3 elements"

    pix_i, pix_j = pixel_coords
    xy = np.ascontiguousarray(
        np.vstack([
            pix_j.flatten(), pix_i.flatten()
            ]).T
        )

    if distortion is not None:
        xy = distortion.apply(xy)

    angs, g_vec = detectorXYToGvec(
        xy, rmat, ct.identity_3x3,
        tvec, ct.zeros_3, origin,
        beamVec=bvec, etaVec=evec)

    tth = angs[0].reshape(rows, cols)
    eta = angs[1].reshape(rows, cols)
    return tth, eta


@memoize
def _pixel_tth_gradient(origin, pixel_coords, distortion, rmat, tvec, bvec,
                        evec, rows, cols):
    assert len(origin) == 3, "origin must have 3 elements"
    ptth, _ = _pixel_angles(origin, pixel_coords, distortion, rmat, tvec,
                            bvec, evec, rows, cols)
    return np.linalg.norm(np.stack(np.gradient(ptth)), axis=0)


@memoize
def _pixel_eta_gradient(origin, pixel_coords, distortion, rmat, tvec, bvec,
                        evec, rows, cols):
    assert len(origin) == 3, "origin must have 3 elemnts"
    _, peta = _pixel_angles(origin, pixel_coords, distortion, rmat, tvec,
                            bvec, evec, rows, cols)

    peta_grad_row = np.gradient(peta, axis=0)
    peta_grad_col = np.gradient(peta, axis=1)

    # !!!: fix branch cut
    peta_grad_row = _fix_branch_cut_in_gradients(peta_grad_row)
    peta_grad_col = _fix_branch_cut_in_gradients(peta_grad_col)

    return np.linalg.norm(np.stack([peta_grad_col, peta_grad_row]), axis=0)


def _fix_branch_cut_in_gradients(pgarray):
    return np.min(
        np.abs(np.stack([pgarray - np.pi, pgarray, pgarray + np.pi])),
        axis=0
    )
