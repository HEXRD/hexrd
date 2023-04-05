import numpy as np

from hexrd import constants as ct
from hexrd import xrdutil
from hexrd.utils.decorators import memoize

from .detector import Detector


class CylindricalDetector(Detector):
    """Base class for 2D cylindrical detector

       A cylindrical detector is a simple rectangular
       row-column detector which has been bent in the
       shape of a cylinder. Inherting the PlanarDetector
       class except for a few changes to account for the
       cylinder ray intersection.
    """

    def __init__(self, **detector_kwargs):
        super().__init__(**detector_kwargs)

    @property
    def detector_type(self):
        return 'cylindrical'

    def _dewarp_from_cylinder(self, uvw):
        """
        routine to convert cylindrical coordinates
        to cartesian coordinates in image frame
        """
        cx = np.atleast_2d(self.caxis).T
        # px = np.atleast_2d(self.paxis).T
        num = uvw.shape[0]

        vx = uvw - np.tile(np.dot(uvw, cx), [1, 3]) * np.tile(cx, [1, num]).T
        vx = vx/np.tile(np.linalg.norm(vx, axis=1), [3, 1]).T
        sgn = np.sign(np.dot(self.paxis, vx.T))
        sgn[sgn == 0] = 1
        ang = np.abs(np.arccos(np.dot(self.tvec, vx.T)/self.radius))
        xcrd = self.radius*ang*sgn
        ycrd = np.dot(self.caxis, uvw.T)
        return np.vstack((xcrd, ycrd)).T

    def _valid_points(self, vecs):
        """
        this rotuine takes a list of vectors
        and checks if it falls inside the
        cylindrical panel

        returns the subset of vectors which fall
        on the panel
        """
        pass

    def _unitvec_to_cylinder(self, uvw):
        """
        get point where unitvector uvw
        intersect the cylindrical detector.
        this will give points which are
        outside the actual panel. the points
        will be clipped to the panel later
        NOTE: solves ray-cylinder intersection
        problem.

        Parameters
        ----------
        uvw : numpy.ndarray
        unit vectors stacked row wise (nx3) shape

        Returns
        -------
        numpy.ndarray
        (x,y,z) vectors point which intersect with
        the cylinder with (nx3) shape
        """
        num = uvw.shape[0]
        cx = np.atleast_2d(self.caxis).T
        dp = np.dot(uvw, cx)
        den = np.squeeze(np.sqrt(1 - dp**2))
        mask = den < 1E-8
        beta = np.zeros([num, ])
        beta[~mask] = self.radius/den[~mask]
        beta[mask] = np.nan

        return np.tile(beta, [3, 1]).T * uvw

    def _clip_to_cylindrical_detector(self, uvw):
        """
        takes in the intersection points uvw
        with the cylindrical detector and
        prunes out points which don't actually
        hit the actual panel

        Parameters
        ----------
        uvw : numpy.ndarray
        unit vectors stacked row wise (nx3) shape

        Returns
        -------
        numpy.ndarray
        (x,y,z) vectors point which fall on panel
        with (mx3) shape
        """
        # first get rid of points which are above
        # or below the detector
        size = self.physical_size
        tvec = np.atleast_2d(self.tvec).T
        cx = np.atleast_2d(self.caxis).T
        num = uvw.shape[0]
        ycomp = uvw - np.tile(self.tvec, [num, 1])
        ylen = np.squeeze(np.dot(ycomp, cx))
        mask = np.abs(ylen) <= size[0] * 0.5
        if not isinstance(mask, np.ndarray):
            mask = np.array([mask])
        res = uvw[mask, :]

        # next get rid of points that fall outside
        # the polar angle range
        num = res.shape[0]
        dp = np.squeeze(np.dot(res, cx))
        v = np.tile(cx, [1, num])*np.tile(dp, [3, 1])
        v = v.T
        xcomp = res - v
        magxcomp = np.linalg.norm(xcomp, axis=1)
        ang = np.squeeze(np.dot(xcomp, tvec)) / self.radius / magxcomp
        ang = np.arccos(ang)
        mask = ang < self.angle_extent
        if not isinstance(mask, np.ndarray):
            mask = np.array([mask])
        res = res[mask, :]

        return res

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

        dvecs = xrdutil.utils._warp_to_cylinder(xy_data,
                                                self.tvec,
                                                self.radius,
                                                self.caxis,
                                                self.paxis,
                                                normalize=True)
        tth, eta = xrdutil.utils._dvec_to_angs(dvecs, self.bvec, self.evec)
        tth_eta = np.vstack((tth, eta)).T
        return tth_eta, dvecs

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
        kwargs = {"beamVec": self.bvec}
        args = (angs, self.rmat, chi, self.tvec,
                self.caxis, self.paxis, self.radius,
                self.physical_size, self.angle_extent,
                self.distortion)

        proj_func = xrdutil.utils._project_on_detector_cylinder
        valid_xy, rMat_ss, valid_mask = proj_func(*args, **kwargs)
        xy_det = np.empty([angs.shape[0], 2])
        xy_det.fill(np.nan)
        xy_det[valid_mask, :] = valid_xy
        return xy_det

    def pixel_angles(self, origin=ct.zeros_3):
        args = (origin, self.pixel_coords, self.distortion, self.tvec,
                self.radius, self.caxis, self.paxis, self.bvec, self.evec,
                self.rows, self.cols)
        return _pixel_angles(*args)

    def pixel_tth_gradient(self, origin=ct.zeros_3):
        return _pixel_tth_gradient(origin, self.pixel_coords, self.distortion,
                                   self.rmat, self.tvec, self.bvec, self.evec,
                                   self.rows, self.cols)

    def pixel_eta_gradient(self, origin=ct.zeros_3):
        return _pixel_eta_gradient(origin, self.pixel_coords, self.distortion,
                                   self.rmat, self.tvec, self.bvec, self.evec,
                                   self.rows, self.cols)

    @property
    def caxis(self):
        # returns the cylinder axis
        return np.dot(self.rmat, ct.lab_y)

    @property
    def paxis(self):
        # returns the cylinder axis
        return np.dot(self.rmat, ct.lab_x)

    @property
    def radius(self):
        # units of mm
        return np.linalg.norm(self.tvec)

    @property
    def physical_size(self):
        # return physical size of detector
        # in mm after dewarped to rectangle
        return np.array([self.rows*self.pixel_size_row,
                         self.cols*self.pixel_size_col])

    @property
    def beam_position(self):
        """
        returns the coordinates of the beam in the cartesian detector
        frame {Xd, Yd, Zd}.  NaNs if no intersection.
        """
        output = np.nan * np.ones(2)
        pt_on_cylinder = self._unitvec_to_cylinder(
                             np.atleast_2d(self.bvec))
        pt_on_cylinder = self._clip_to_cylindrical_detector(pt_on_cylinder)
        output = self._dewarp_from_cylinder(pt_on_cylinder)
        return output

    @property
    def angle_extent(self):
        # extent is from -theta, theta
        sz = self.physical_size[1]
        return sz / self.radius / 2.0

    @staticmethod
    def update_memoization_sizes(all_panels):
        Detector.update_memoization_sizes(all_panels)

        num_matches = sum(x is CylindricalDetector for x in all_panels)
        funcs = [
            _pixel_angles,
            _pixel_tth_gradient,
            _pixel_eta_gradient,
        ]
        Detector.increase_memoization_sizes(funcs, num_matches)


@memoize
def _pixel_angles(origin,
                  pixel_coords,
                  distortion,
                  tVec_d,
                  radius,
                  caxis,
                  paxis,
                  bvec,
                  evec,
                  rows,
                  cols):
    assert len(origin) == 3, "origin must have 3 elements"

    pix_i, pix_j = pixel_coords
    xy = np.ascontiguousarray(
        np.vstack([
            pix_j.flatten(), pix_i.flatten()
            ]).T
        )

    if distortion is not None:
        xy = distortion.apply(xy)

    dvecs = xrdutil.utils._warp_to_cylinder(xy,
                                            tVec_d-origin,
                                            radius,
                                            caxis,
                                            paxis,
                                            normalize=True)

    angs = xrdutil.utils._dvec_to_angs(dvecs, bvec, evec)

    tth = angs[0].reshape(rows, cols)
    eta = angs[1].reshape(rows, cols)
    return tth, eta


@memoize
def _pixel_tth_gradient(origin, pixel_coords, distortion, caxis, paxis, tvec, bvec,
                        evec, rows, cols):
    assert len(origin) == 3, "origin must have 3 elements"
    ptth, _ = _pixel_angles(origin, pixel_coords, distortion, caxis, paxis, tvec,
                            bvec, evec, rows, cols)
    return np.linalg.norm(np.stack(np.gradient(ptth)), axis=0)


@memoize
def _pixel_eta_gradient(origin, pixel_coords, distortion, caxis, paxis, tvec, bvec,
                        evec, rows, cols):
    assert len(origin) == 3, "origin must have 3 elemnts"
    _, peta = _pixel_angles(origin, pixel_coords, distortion, caxis, paxis, tvec,
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
