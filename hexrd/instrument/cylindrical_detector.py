import numpy as np

from hexrd import constants as ct
from hexrd import xrdutil
from hexrd.utils.decorators import memoize

from .detector import (
    Detector, _solid_angle_of_triangle, _row_edge_vec, _col_edge_vec
)

from functools import partial
from hexrd.gridutil import cellConnectivity
from hexrd.utils.concurrent import distribute_tasks
from concurrent.futures import ProcessPoolExecutor


class CylindricalDetector(Detector):
    """Base class for 2D cylindrical detector

       A cylindrical detector is a simple rectangular
       row-column detector which has been bent in the
       shape of a cylinder. Inherting the PlanarDetector
       class except for a few changes to account for the
       cylinder ray intersection.
    """

    def __init__(self, radius=49.51, **detector_kwargs):
        self._radius = radius
        super().__init__(**detector_kwargs)

    @property
    def detector_type(self):
        return 'cylindrical'

    def cart_to_angles(self, xy_data,
                       rmat_s=None,
                       tvec_s=None, tvec_c=None,
                       apply_distortion=False):
        xy_data = np.asarray(xy_data)
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
        return _pixel_angles(origin=origin, **self._pixel_angle_kwargs)

    @property
    def _pixel_angle_kwargs(self):
        # kwargs used for pixel angles, pixel_tth_gradient,
        # and pixel_eta_gradient
        return {
            'pixel_coords': self.pixel_coords,
            'distortion': self.distortion,
            'caxis': self.caxis,
            'paxis': self.paxis,
            'tvec_d': self.tvec,
            'radius': self.radius,
            'bvec': self.bvec,
            'evec': self.evec,
            'rows': self.rows,
            'cols': self.cols,
        }

    def pixel_tth_gradient(self, origin=ct.zeros_3):
        return _pixel_tth_gradient(origin=origin, **self._pixel_angle_kwargs)

    def pixel_eta_gradient(self, origin=ct.zeros_3):
        return _pixel_eta_gradient(origin=origin, **self._pixel_angle_kwargs)

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
        return self._radius

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
        args = (np.atleast_2d(self.bvec), self.caxis, self.paxis,
                self.radius, self.tvec)
        pt_on_cylinder = xrdutil.utils._unitvec_to_cylinder(*args)

        args = (pt_on_cylinder, self.tvec, self.caxis,
                self.paxis, self.radius, self.physical_size,
                self.angle_extent)
        pt_on_cylinder, _ = xrdutil.utils._clip_to_cylindrical_detector(*args)

        args = (pt_on_cylinder, self.tvec, self.caxis, self.paxis, self.radius)
        output = xrdutil.utils._dewarp_from_cylinder(*args)
        return output

    @property
    def angle_extent(self):
        # extent is from -theta, theta
        sz = self.physical_size[1]
        return sz / self.radius / 2.0

    @property
    def pixel_solid_angles(self):
        kwargs = {
            'rows': self.rows,
            'cols': self.cols,
            'pixel_size_row': self.pixel_size_row,
            'pixel_size_col': self.pixel_size_col,
            'caxis': self.caxis,
            'paxis': self.paxis,
            'radius': self.radius,
            'tvec': self.tvec,
            'max_workers': self.max_workers,
        }
        return _pixel_solid_angles(**kwargs)

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

    @property
    def extra_config_kwargs(self):
        return {
            'radius': self.radius,
        }


@memoize
def _pixel_angles(origin,
                  pixel_coords,
                  distortion,
                  caxis,
                  paxis,
                  tvec_d,
                  radius,
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
                                            tvec_d-origin,
                                            radius,
                                            caxis,
                                            paxis,
                                            normalize=True)

    angs = xrdutil.utils._dvec_to_angs(dvecs, bvec, evec)

    tth = angs[0].reshape(rows, cols)
    eta = angs[1].reshape(rows, cols)
    return tth, eta


@memoize
def _pixel_tth_gradient(origin, **pixel_angle_kwargs):
    assert len(origin) == 3, "origin must have 3 elements"
    ptth, _ = _pixel_angles(origin=origin, **pixel_angle_kwargs)
    return np.linalg.norm(np.stack(np.gradient(ptth)), axis=0)


@memoize
def _pixel_eta_gradient(origin, **pixel_angle_kwargs):
    assert len(origin) == 3, "origin must have 3 elemnts"
    _, peta = _pixel_angles(origin=origin, **pixel_angle_kwargs)

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


def _generate_pixel_solid_angles(start_stop, rows, cols, pixel_size_row,
                                 pixel_size_col, caxis, paxis, radius, tvec):
    start, stop = start_stop
    row_edge_vec = _row_edge_vec(rows, pixel_size_row)
    col_edge_vec = _col_edge_vec(cols, pixel_size_col)

    # pixel vertex coords
    pvy, pvx = np.meshgrid(row_edge_vec, col_edge_vec, indexing='ij')
    xy_data = np.vstack((pvx.flatten(), pvy.flatten())).T

    # transform to lab frame using the _warp_to_cylinder
    # function
    pcrd_array_full = xrdutil.utils._warp_to_cylinder(xy_data,
                                                      tvec,
                                                      radius,
                                                      caxis,
                                                      paxis,
                                                      normalize=False)

    conn = cellConnectivity(rows, cols)

    ret = np.empty(len(range(start, stop)), dtype=float)

    for i, ipix in enumerate(range(start, stop)):
        pix_conn = conn[ipix]
        vtx_list = pcrd_array_full[pix_conn, :]
        ret[i] = (_solid_angle_of_triangle(vtx_list[[0, 1, 2], :]) +
                  _solid_angle_of_triangle(vtx_list[[2, 3, 0], :]))

    return ret


@memoize
def _pixel_solid_angles(rows, cols, pixel_size_row, pixel_size_col,
                        caxis, paxis, radius, tvec, max_workers):
    # connectivity array for pixels
    conn = cellConnectivity(rows, cols)

    # result
    solid_angs = np.empty(len(conn), dtype=float)

    # Distribute tasks to each process
    tasks = distribute_tasks(len(conn), max_workers)
    kwargs = {
        'rows': rows,
        'cols': cols,
        'pixel_size_row': pixel_size_row,
        'pixel_size_col': pixel_size_col,
        'caxis': caxis,
        'paxis': paxis,
        'radius': radius,
        'tvec': tvec,
    }
    func = partial(_generate_pixel_solid_angles, **kwargs)
    with ProcessPoolExecutor(mp_context=ct.mp_context,
                             max_workers=max_workers) as executor:
        results = executor.map(func, tasks)

    # Concatenate all the results together
    solid_angs[:] = np.concatenate(list(results))
    solid_angs = solid_angs.reshape(rows, cols)
    mi = solid_angs.min()
    if mi > 0.:
        solid_angs = solid_angs/mi

    return solid_angs
