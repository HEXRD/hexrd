import numpy as np

from hexrd.core import constants as ct
from hexrd.core.transforms.xfcapi import angles_to_gvec, xy_to_gvec, gvec_to_xy, make_beam_rmat, angles_to_dvec
from hexrd.core.utils.decorators import memoize

from .detector import Detector, _solid_angle_of_triangle, _row_edge_vec, _col_edge_vec
from ...hed.instrument.detector import Detector, _solid_angle_of_triangle, _row_edge_vec, _col_edge_vec
from ...hedm.instrument.detector import Detector, _solid_angle_of_triangle, _row_edge_vec, _col_edge_vec
from ...laue.instrument.detector import Detector, _solid_angle_of_triangle, _row_edge_vec, _col_edge_vec
from ...powder.instrument.detector import Detector, _solid_angle_of_triangle, _row_edge_vec, _col_edge_vec

from functools import partial
from hexrd.core.gridutil import cellConnectivity
from hexrd.core.utils.concurrent import distribute_tasks
from concurrent.futures import ProcessPoolExecutor


class PlanarDetector(Detector):
    """Base class for 2D planar, rectangular row-column detector"""

    def __init__(self, **detector_kwargs):
        super().__init__(**detector_kwargs)

    @property
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
        rmat_b = make_beam_rmat(self.bvec, self.evec)
        angs, g_vec = xy_to_gvec(
            xy_data, self.rmat, rmat_s,
            self.tvec, tvec_s, tvec_c,
            rmat_b=rmat_b)
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
        gvec = angles_to_gvec(angs, beam_vec=self.bvec, eta_vec=self.evec,
                              chi=chi)
        xy_det = gvec_to_xy(
            gvec,
            self.rmat, rmat_s, rmat_c,
            self.tvec, tvec_s, tvec_c,
            beam_vec=self.bvec)
        if apply_distortion and self.distortion is not None:
            xy_det = self.distortion.apply_inverse(xy_det)
        return xy_det

    def cart_to_dvecs(self, xy_data):
        npts = len(xy_data)
        crds = np.hstack([xy_data, np.zeros((npts, 1))])
        return np.dot(crds, self.rmat.T) + self.tvec

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

    def calc_filter_coating_transmission(self, energy: np.floating) -> tuple[np.ndarray, np.ndarray]:
        """
        calculate thetrnasmission after x-ray beam interacts
        with the filter and the mylar polymer coating.
        Specifications of the polymer coating is taken from:

        M. Stoeckl, A. A. Solodov
        Readout models for BaFBr0.85I0.15:Eu image plates
        Rev. Sci. Instrum. 89, 063101 (2018)

        Transmission Formulas are consistent with:

        Rygg et al., X-ray diffraction at the National
        Ignition Facility, Rev. Sci. Instrum. 91, 043902 (2020)
        """

        al_f = self.filter.absorption_length(energy)
        al_c = self.coating.absorption_length(energy)
        al_p = self.phosphor.energy_absorption_length(energy)

        t_f = self.filter.thickness
        t_c = self.coating.thickness
        t_p = self.phosphor.thickness
        L   = self.phosphor.readout_length
        pre_U0 = self.phosphor.pre_U0

        det_normal = -self.normal
        bvec = self.bvec

        tth, eta = self.pixel_angles()
        angs = np.vstack((tth.flatten(), eta.flatten(),
                          np.zeros(tth.flatten().shape))).T

        dvecs = angles_to_dvec(angs, beam_vec=bvec)

        secb = 1./np.dot(dvecs, det_normal).reshape(self.shape)

        transmission_filter  = self.calc_transmission_generic(secb, t_f, al_f)
        transmission_coating = self.calc_transmission_generic(secb, t_c, al_c)
        transmission_phosphor = (
            self.calc_transmission_phosphor(secb, t_p, al_p, L, energy, pre_U0))
        transmission_filter_coating = (
            transmission_filter * transmission_coating)

        return transmission_filter_coating, transmission_phosphor

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

    @property
    def pixel_solid_angles(self):
        kwargs = {
            'rows': self.rows,
            'cols': self.cols,
            'pixel_size_row': self.pixel_size_row,
            'pixel_size_col': self.pixel_size_col,
            'rmat': self.rmat,
            'tvec': self.tvec,
            'max_workers': self.max_workers,
        }
        return _pixel_solid_angles(**kwargs)

    @staticmethod
    def update_memoization_sizes(all_panels):
        Detector.update_memoization_sizes(all_panels)

        num_matches = sum(isinstance(x, PlanarDetector) for x in all_panels)
        funcs = [
            _pixel_angles,
            _pixel_tth_gradient,
            _pixel_eta_gradient,
            _pixel_solid_angles,
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
    rmat_b = make_beam_rmat(bvec, evec)
    angs, g_vec = xy_to_gvec(
        xy, rmat, ct.identity_3x3,
        tvec, ct.zeros_3, origin,
        rmat_b=rmat_b)

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


def _generate_pixel_solid_angles(start_stop, rows, cols, pixel_size_row,
                                 pixel_size_col, rmat, tvec):
    start, stop = start_stop
    row_edge_vec = _row_edge_vec(rows, pixel_size_row)
    col_edge_vec = _col_edge_vec(cols, pixel_size_col)

    nvtx = len(row_edge_vec) * len(col_edge_vec)
    # pixel vertex coords
    pvy, pvx = np.meshgrid(row_edge_vec, col_edge_vec, indexing='ij')

    # add Z_d coord and transform to lab frame
    pcrd_array_full = np.dot(
        np.vstack([pvx.flatten(), pvy.flatten(), np.zeros(nvtx)]).T,
        rmat.T
    ) + tvec

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
                        rmat, tvec, max_workers):
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
        'rmat': rmat,
        'tvec': tvec,
    }
    func = partial(_generate_pixel_solid_angles, **kwargs)
    with ProcessPoolExecutor(mp_context=ct.mp_context,
                             max_workers=max_workers) as executor:
        results = executor.map(func, tasks)

    # Concatenate all the results together
    solid_angs[:] = np.concatenate(list(results))
    solid_angs = solid_angs.reshape(rows, cols)

    return solid_angs
