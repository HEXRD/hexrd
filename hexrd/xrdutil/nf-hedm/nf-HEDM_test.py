"""
Refactor of simulate_nf so that an experiment is mocked up.

Also trying to minimize imports
"""

import os
import logging

import numpy as np
import numba
import yaml
import argparse
import timeit
import contextlib
import multiprocessing
import tempfile
import shutil
import socket

# import of hexrd modules
import hexrd
from hexrd import constants
from hexrd import instrument
from hexrd import material
from hexrd import rotations
from hexrd.transforms import xfcapi
from hexrd import valunits
from hexrd import xrdutil

from skimage.morphology import dilation as ski_dilation

hostname = socket.gethostname()

USE_MPI = False
rank = 0
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    USE_MPI = world_size > 1
    logging.info(f'{rank=} {world_size=} {hostname=}')
except ImportError:
    logging.warning(f'mpi4py failed to load on {hostname=}. MPI is disabled.')
    pass


beam = constants.beam_vec
Z_l = constants.lab_z
vInv_ref = constants.identity_6x1


# ==============================================================================
# %% SOME SCAFFOLDING
# ==============================================================================


class ProcessController:
    """This is a 'controller' that provides the necessary hooks to
    track the results of the process as well as to provide clues of
    the progress of the process"""

    def __init__(self, result_handler=None, progress_observer=None, ncpus=1,
                 chunk_size=100):
        self.rh = result_handler
        self.po = progress_observer
        self.ncpus = ncpus
        self.chunk_size = chunk_size
        self.limits = {}
        self.timing = []

    # progress handling -------------------------------------------------------

    def start(self, name, count):
        self.po.start(name, count)
        t = timeit.default_timer()
        self.timing.append((name, count, t))

    def finish(self, name):
        t = timeit.default_timer()
        self.po.finish()
        entry = self.timing.pop()
        assert name == entry[0]
        total = t - entry[2]
        logging.info("%s took %8.3fs (%8.6fs per item).",
                     entry[0], total, total/entry[1])

    def update(self, value):
        self.po.update(value)

    # result handler ----------------------------------------------------------

    def handle_result(self, key, value):
        logging.debug("handle_result (%(key)s)", locals())
        self.rh.handle_result(key, value)

    # value limitting ---------------------------------------------------------
    def set_limit(self, key, limit_function):
        if key in self.limits:
            logging.warn("Overwritting limit funtion for '%(key)s'", locals())

        self.limits[key] = limit_function

    def limit(self, key, value):
        try:
            value = self.limits[key](value)
        except KeyError:
            pass
        except Exception:
            logging.warn("Could not apply limit to '%(key)s'", locals())

        return value

    # configuration  ----------------------------------------------------------

    def get_process_count(self):
        return self.ncpus

    def get_chunk_size(self):
        return self.chunk_size


def null_progress_observer():
    class NullProgressObserver:
        def start(self, name, count):
            pass

        def update(self, value):
            pass

        def finish(self):
            pass

    return NullProgressObserver()


def progressbar_progress_observer():

    class ProgressBarProgressObserver:
        def start(self, name, count):
            from progressbar import ProgressBar, Percentage, Bar

            self.pbar = ProgressBar(widgets=[name, Percentage(), Bar()],
                                    maxval=count)
            self.pbar.start()

        def update(self, value):
            self.pbar.update(value)

        def finish(self):
            self.pbar.finish()

    return ProgressBarProgressObserver()


def forgetful_result_handler():
    class ForgetfulResultHandler:
        def handle_result(self, key, value):
            pass  # do nothing

    return ForgetfulResultHandler()


def saving_result_handler(filename):
    """returns a result handler that saves the resulting arrays into a file
    with name filename"""
    class SavingResultHandler:
        def __init__(self, file_name):
            self.filename = file_name
            self.arrays = {}

        def handle_result(self, key, value):
            self.arrays[key] = value

        def __del__(self):
            logging.debug("Writing arrays in %(filename)s", self.__dict__)
            try:
                np.savez_compressed(open(self.filename, "wb"), **self.arrays)
            except IOError:
                logging.error("Failed to write %(filename)s", self.__dict__)

    return SavingResultHandler(filename)


def checking_result_handler(filename):
    """returns a return handler that checks the results against a
    reference file.

    The Check will consider a FAIL either a result not present in the
    reference file (saved as a numpy savez or savez_compressed) or a
    result that differs. It will consider a PARTIAL PASS if the
    reference file has a shorter result, but the existing results
    match. A FULL PASS will happen when all existing results match

    """
    class CheckingResultHandler:
        def __init__(self, reference_file):
            """Checks the result against those save in 'reference_file'"""
            logging.info("Loading reference results from '%s'", reference_file)
            self.reference_results = np.load(open(reference_file, 'rb'))

        def handle_result(self, key, value):
            if key in ['experiment', 'image_stack']:
                return  # ignore these

            try:
                reference = self.reference_results[key]
            except KeyError as e:
                logging.warning("%(key)s: %(e)s", locals())
                reference = None

            if reference is None:
                msg = "'{0}': No reference result."
                logging.warn(msg.format(key))

            try:
                if key == "confidence":
                    reference = reference.T
                    value = value.T

                check_len = min(len(reference), len(value))
                test_passed = np.allclose(value[:check_len],
                                          reference[:check_len])

                if not test_passed:
                    msg = "'{0}': FAIL"
                    logging.warn(msg.format(key))
                    lvl = logging.WARN
                elif len(value) > check_len:
                    msg = "'{0}': PARTIAL PASS"
                    lvl = logging.WARN
                else:
                    msg = "'{0}': FULL PASS"
                    lvl = logging.INFO
                logging.log(lvl, msg.format(key))
            except Exception as e:
                msg = "%(key)s: Failure trying to check the results.\n%(e)s"
                logging.error(msg, locals())

    return CheckingResultHandler(filename)


# ==============================================================================
# %% SETUP FUNCTION
# ==============================================================================
def mockup_experiment():
    # user options
    # each grain is provided in the form of a quaternion.

    # The following array contains the quaternions for the array. Note that the
    # quaternions are in the columns, with the first row (row 0) being the real
    # part w. We assume that we are dealing with unit quaternions

    quats = np.array([[0.91836393,  0.90869942],
                      [0.33952917,  0.18348350],
                      [0.17216207,  0.10095837],
                      [0.10811041,  0.36111851]])

    n_grains = quats.shape[-1]  # last dimension provides the number of grains
    phis = 2.*np.arccos(quats[0, :])  # phis are the angles for the quaternion
    # ns contains the rotation axis as an unit vector
    ns = hexrd.matrixutil.unitVector(quats[1:, :])
    exp_maps = np.array([phis[i]*ns[:, i] for i in range(n_grains)])
    rMat_c = rotations.rotMatOfQuat(quats)

    cvec = np.arange(-25, 26)
    X, Y, Z = np.meshgrid(cvec, cvec, cvec)

    crd0 = 1e-3*np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    crd1 = crd0 + np.r_[0.100, 0.100, 0]
    crds = np.array([crd0, crd1])

    # make grain parameters
    grain_params = []
    for i in range(n_grains):
        for j in range(len(crd0)):
            grain_params.append(
                np.hstack([exp_maps[i, :], crds[i][j, :], vInv_ref])
            )

    # scan range and period
    ome_period = (0, 2*np.pi)
    ome_range = [ome_period, ]
    ome_step = np.radians(1.)
    nframes = 0
    for i in range(len(ome_range)):
        nframes += int((ome_range[i][1]-ome_range[i][0])/ome_step)

    ome_edges = np.arange(nframes+1)*ome_step

    # instrument
    with open('./retiga.yml', 'r') as fildes:
        instr = instrument.HEDMInstrument(yaml.safe_load(fildes))
    panel = next(iter(instr.detectors.values()))  # !!! there is only 1

    # tranform paramters
    #   Sample
    chi = instr.chi
    tVec_s = instr.tvec
    #   Detector
    rMat_d = panel.rmat
    tilt_angles_xyzp = np.asarray(rotations.angles_from_rmat_xyz(rMat_d))
    tVec_d = panel.tvec

    # pixels
    row_ps = panel.pixel_size_row
    col_ps = panel.pixel_size_col
    pixel_size = (row_ps, col_ps)
    nrows = panel.rows
    ncols = panel.cols

    # panel dimensions
    panel_dims = [tuple(panel.corner_ll),
                  tuple(panel.corner_ur)]

    x_col_edges = panel.col_edge_vec
    y_row_edges = panel.row_edge_vec
    rx, ry = np.meshgrid(x_col_edges, y_row_edges)

    max_pixel_tth = instrument.max_tth(instr)

    detector_params = np.hstack([tilt_angles_xyzp, tVec_d, chi, tVec_s])
    distortion = panel.distortion  # !!! must be None for now

    # a different parametrization for the sensor
    # (makes for faster quantization)
    base = np.array([x_col_edges[0],
                     y_row_edges[0],
                     ome_edges[0]])
    deltas = np.array([x_col_edges[1] - x_col_edges[0],
                       y_row_edges[1] - y_row_edges[0],
                       ome_edges[1] - ome_edges[0]])
    inv_deltas = 1.0/deltas
    clip_vals = np.array([ncols, nrows])

    # dilation
    max_diameter = np.sqrt(3)*0.005
    row_dilation = int(np.ceil(0.5 * max_diameter/row_ps))
    col_dilation = int(np.ceil(0.5 * max_diameter/col_ps))

    # crystallography data
    beam_energy = valunits.valWUnit("beam_energy", "energy", instr.beam_energy, "keV")
    beam_wavelength = constants.keVToAngstrom(beam_energy.getVal('keV'))
    dmin = valunits.valWUnit("dmin", "length",
                             0.5*beam_wavelength/np.sin(0.5*max_pixel_tth),
                             "angstrom")

    gold = material.Material()
    gold.latticeParameters = [4.0782]
    gold.dmin = dmin
    gold.beamEnergy = beam_energy
    gold.planeData.exclusions = None
    gold.planeData.tThMax = max_pixel_tth  # note this comes detector

    ns = argparse.Namespace()
    # grains related information
    ns.n_grains = n_grains  # this can be derived from other values...
    ns.rMat_c = rMat_c  # n_grains rotation matrices (one per grain)
    ns.exp_maps = exp_maps  # n_grains exp_maps (one per grain)

    ns.plane_data = gold.planeData
    ns.detector_params = detector_params
    ns.pixel_size = pixel_size
    ns.ome_range = ome_range
    ns.ome_period = ome_period
    ns.x_col_edges = x_col_edges
    ns.y_row_edges = y_row_edges
    ns.ome_edges = ome_edges
    ns.ncols = ncols
    ns.nrows = nrows
    ns.nframes = nframes  # used only in simulate...
    ns.rMat_d = rMat_d
    ns.tVec_d = tVec_d
    ns.chi = chi  # note this is used to compute S... why is it needed?
    ns.tVec_s = tVec_s
    ns.rMat_c = rMat_c
    ns.row_dilation = row_dilation
    ns.col_dilation = col_dilation
    ns.distortion = distortion
    ns.panel_dims = panel_dims  # used only in simulate...
    ns.base = base
    ns.inv_deltas = inv_deltas
    ns.clip_vals = clip_vals

    return grain_params, ns


# =============================================================================
# %% OPTIMIZED BITS
# =============================================================================

# Some basic 3d algebra =======================================================
@numba.njit(nogil=True, cache=True)
def _v3_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@numba.njit(nogil=True, cache=True)
def _m33_v3_multiply(m, v, dst):
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    dst[0] = m[0, 0]*v0 + m[0, 1]*v1 + m[0, 2]*v2
    dst[1] = m[1, 0]*v0 + m[1, 1]*v1 + m[1, 2]*v2
    dst[2] = m[2, 0]*v0 + m[2, 1]*v1 + m[2, 2]*v2

    return dst


@numba.njit(nogil=True, cache=True)
def _v3_normalized(src, dst):
    v0 = src[0]
    v1 = src[1]
    v2 = src[2]
    sqr_norm = v0*v0 + v1*v1 + v2*v2
    inv_norm = 1.0 if sqr_norm == 0.0 else 1./np.sqrt(sqr_norm)

    dst[0] = v0 * inv_norm
    dst[1] = v1 * inv_norm
    dst[2] = v2 * inv_norm

    return dst


@numba.njit(nogil=True, cache=True)
def _make_binary_rot_mat(src, dst):
    v0 = src[0]
    v1 = src[1]
    v2 = src[2]

    dst[0, 0] = 2.0*v0*v0 - 1.0
    dst[0, 1] = 2.0*v0*v1
    dst[0, 2] = 2.0*v0*v2
    dst[1, 0] = 2.0*v1*v0
    dst[1, 1] = 2.0*v1*v1 - 1.0
    dst[1, 2] = 2.0*v1*v2
    dst[2, 0] = 2.0*v2*v0
    dst[2, 1] = 2.0*v2*v1
    dst[2, 2] = 2.0*v2*v2 - 1.0

    return dst


# code transcribed in numba from transforms module ============================

# This is equivalent to the transform module anglesToGVec, but written in
# numba. This should end in a module to share with other scripts
@numba.njit(nogil=True, cache=True)
def _anglesToGVec(angs, rMat_ss, rMat_c):
    """From a set of angles return them in crystal space"""
    result = np.empty_like(angs)
    for i in range(len(angs)):
        cx = np.cos(0.5*angs[i, 0])
        sx = np.sin(0.5*angs[i, 0])
        cy = np.cos(angs[i, 1])
        sy = np.sin(angs[i, 1])
        g0 = cx*cy
        g1 = cx*sy
        g2 = sx

        # with g being [cx*xy, cx*sy, sx]
        # result = dot(rMat_c, dot(rMat_ss[i], g))
        t0_0 = \
            rMat_ss[i, 0, 0]*g0 + rMat_ss[i, 1, 0]*g1 + rMat_ss[i, 2, 0]*g2
        t0_1 = \
            rMat_ss[i, 0, 1]*g0 + rMat_ss[i, 1, 1]*g1 + rMat_ss[i, 2, 1]*g2
        t0_2 = \
            rMat_ss[i, 0, 2]*g0 + rMat_ss[i, 1, 2]*g1 + rMat_ss[i, 2, 2]*g2

        result[i, 0] = \
            rMat_c[0, 0]*t0_0 + rMat_c[1, 0]*t0_1 + rMat_c[2, 0]*t0_2
        result[i, 1] = \
            rMat_c[0, 1]*t0_0 + rMat_c[1, 1]*t0_1 + rMat_c[2, 1]*t0_2
        result[i, 2] = \
            rMat_c[0, 2]*t0_0 + rMat_c[1, 2]*t0_1 + rMat_c[2, 2]*t0_2

    return result


# This is equivalent to the transform's module gvec_to_xy,
# but written in numba.
# As of now, it is not a good replacement as efficient allocation of the
# temporary arrays is not competitive with the stack allocation using in
# the C version of the code (WiP)

# tC varies per coord
# gvec_cs, rSm varies per grain
#
# gvec_cs
@numba.njit(nogil=True, cache=True)
def _gvec_to_detector_array(vG_sn, rD, rSn, rC, tD, tS, tC):
    """ beamVec is the beam vector: (0, 0, -1) in this case """
    ztol = xrdutil.epsf
    p3_l = np.empty((3,))
    tmp_vec = np.empty((3,))
    vG_l = np.empty((3,))
    tD_l = np.empty((3,))
    norm_vG_s = np.empty((3,))
    norm_beam = np.empty((3,))
    tZ_l = np.empty((3,))
    brMat = np.empty((3, 3))
    result = np.empty((len(rSn), 2))

    _v3_normalized(beam, norm_beam)
    _m33_v3_multiply(rD, Z_l, tZ_l)

    for i in range(len(rSn)):
        _m33_v3_multiply(rSn[i], tC, p3_l)
        p3_l += tS
        p3_minus_p1_l = tD - p3_l

        num = _v3_dot(tZ_l, p3_minus_p1_l)
        _v3_normalized(vG_sn[i], norm_vG_s)

        _m33_v3_multiply(rC, norm_vG_s, tmp_vec)
        _m33_v3_multiply(rSn[i], tmp_vec, vG_l)

        bDot = -_v3_dot(norm_beam, vG_l)

        if bDot < ztol or bDot > 1.0 - ztol:
            result[i, 0] = np.nan
            result[i, 1] = np.nan
            continue

        _make_binary_rot_mat(vG_l, brMat)
        _m33_v3_multiply(brMat, norm_beam, tD_l)
        denom = _v3_dot(tZ_l, tD_l)

        if denom < ztol:
            result[i, 0] = np.nan
            result[i, 1] = np.nan
            continue

        u = num/denom
        tmp_res = u*tD_l - p3_minus_p1_l
        result[i, 0] = _v3_dot(tmp_res, rD[:, 0])
        result[i, 1] = _v3_dot(tmp_res, rD[:, 1])

    return result


@numba.njit(nogil=True, cache=True)
def _quant_and_clip_confidence(coords, angles, image,
                               base, inv_deltas, clip_vals):
    """quantize and clip the parametric coordinates in coords + angles

    coords - (..., 2) array: input 2d parametric coordinates
    angles - (...) array: additional dimension for coordinates
    base   - (3,) array: base value for quantization (for each dimension)
    inv_deltas - (3,) array: inverse of the quantum size (for each dimension)
    clip_vals - (2,) array: clip size (only applied to coords dimensions)

    clipping is performed on ranges [0, clip_vals[0]] for x and
    [0, clip_vals[1]] for y

    returns an array with the quantized coordinates, with coordinates
    falling outside the clip zone filtered out.

    """
    count = len(coords)

    in_sensor = 0
    matches = 0
    for i in range(count):
        xf = coords[i, 0]
        yf = coords[i, 1]

        xf = np.floor((xf - base[0]) * inv_deltas[0])
        if not xf >= 0.0:
            continue
        if not xf < clip_vals[0]:
            continue

        yf = np.floor((yf - base[1]) * inv_deltas[1])

        if not yf >= 0.0:
            continue
        if not yf < clip_vals[1]:
            continue

        zf = np.floor((angles[i] - base[2]) * inv_deltas[2])

        in_sensor += 1

        x, y, z = int(xf), int(yf), int(zf)

        x_byte = x // 8
        x_off = 7 - (x % 8)
        if image[z, y, x_byte] & (1 << x_off):
            matches += 1

    return 0 if in_sensor == 0 else float(matches)/float(in_sensor)


# ==============================================================================
# %% DIFFRACTION SIMULATION
# ==============================================================================

def get_simulate_diffractions(grain_params, experiment,
                              cache_file='gold_cubes.npy',
                              controller=None):
    """getter functions that handles the caching of the simulation"""
    try:
        image_stack = np.load(cache_file, mmap_mode='r', allow_pickle=False)
    except Exception:
        image_stack = simulate_diffractions(grain_params, experiment,
                                            controller=controller)
        np.save(cache_file, image_stack)

    controller.handle_result('image_stack', image_stack)

    return image_stack


def simulate_diffractions(grain_params, experiment, controller):
    """actual forward simulation of the diffraction"""

    # use a packed array for the image_stack
    array_dims = (experiment.nframes,
                  experiment.ncols,
                  ((experiment.nrows - 1)//8) + 1)
    image_stack = np.zeros(array_dims, dtype=np.uint8)

    count = len(grain_params)
    subprocess = 'simulate diffractions'

    _project = xrdutil._project_on_detector_plane
    rD = experiment.rMat_d
    chi = experiment.chi
    tD = experiment.tVec_d
    tS = experiment.tVec_s
    distortion = experiment.distortion

    eta_range = [(-np.pi, np.pi), ]
    ome_range = experiment.ome_range
    ome_period = (-np.pi, np.pi)

    full_hkls = xrdutil._fetch_hkls_from_planedata(experiment.plane_data)
    bMat = experiment.plane_data.latVecOps['B']
    wlen = experiment.plane_data.wavelength

    controller.start(subprocess, count)
    for i in range(count):
        rC = xfcapi.makeRotMatOfExpMap(grain_params[i][0:3])
        tC = np.ascontiguousarray(grain_params[i][3:6])
        vInv_s = np.ascontiguousarray(grain_params[i][6:12])
        ang_list = np.vstack(xfcapi.oscillAnglesOfHKLs(full_hkls[:, 1:], chi,
                                                       rC, bMat, wlen,
                                                       vInv=vInv_s))
        # hkls not needed here
        all_angs, _ = xrdutil._filter_hkls_eta_ome(full_hkls, ang_list,
                                                   eta_range, ome_range)
        all_angs[:, 2] = xfcapi.mapAngle(all_angs[:, 2], ome_period)

        proj_pts =  _project(all_angs, rD, rC, chi, tD,
                             tC, tS, distortion)
        det_xy = proj_pts[0]
        _write_pixels(det_xy, all_angs[:, 2], image_stack, experiment.base,
                      experiment.inv_deltas, experiment.clip_vals)

        controller.update(i + 1)

    controller.finish(subprocess)
    return image_stack


# ==============================================================================
# %% IMAGE DILATION
# ==============================================================================


def get_dilated_image_stack(image_stack, experiment, controller,
                            cache_file='gold_cubes_dilated.npy'):

    try:
        dilated_image_stack = np.load(cache_file, mmap_mode='r',
                                      allow_pickle=False)
    except Exception:
        dilated_image_stack = dilate_image_stack(image_stack, experiment,
                                                 controller)
        np.save(cache_file, dilated_image_stack)

    return dilated_image_stack


def dilate_image_stack(image_stack, experiment, controller):
    # first, perform image dilation ===========================================
    # perform image dilation (using scikit_image dilation)
    subprocess = 'dilate image_stack'
    dilation_shape = np.ones((2*experiment.row_dilation + 1,
                              2*experiment.col_dilation + 1),
                             dtype=np.uint8)
    image_stack_dilated = np.empty_like(image_stack)
    dilated = np.empty(
        (image_stack.shape[-2], image_stack.shape[-1] << 3),
        dtype=bool
    )
    n_images = len(image_stack)
    controller.start(subprocess, n_images)
    for i_image in range(n_images):
        to_dilate = np.unpackbits(image_stack[i_image], axis=-1)
        ski_dilation(to_dilate, dilation_shape,
                     out=dilated)
        image_stack_dilated[i_image] = np.packbits(dilated, axis=-1)
        controller.update(i_image + 1)
    controller.finish(subprocess)

    return image_stack_dilated


# This part is critical for the performance of simulate diffractions. It
# basically "renders" the "pixels". It takes the coordinates, quantizes to an
# image coordinate and writes to the appropriate image in the stack. Note
# that it also performs clipping based on inv_deltas and clip_vals.
#
# Note: This could be easily modified so that instead of using an array of
#       booleans, an array of uint8 could be used so the image is stored
#       with a bit per pixel.

@numba.njit(nogil=True, cache=True)
def _write_pixels(coords, angles, image, base, inv_deltas, clip_vals):
    count = len(coords)
    for i in range(count):
        x = int(np.floor((coords[i, 0] - base[0]) * inv_deltas[0]))

        if x < 0 or x >= clip_vals[0]:
            continue

        y = int(np.floor((coords[i, 1] - base[1]) * inv_deltas[1]))

        if y < 0 or y >= clip_vals[1]:
            continue

        z = int(np.floor((angles[i] - base[2]) * inv_deltas[2]))

        x_byte = x // 8
        x_off = 7 - (x % 8)
        image[z, y, x_byte] |= (1 << x_off)

def get_offset_size(n_coords):
    offset = 0
    size = n_coords
    if USE_MPI:
        coords_per_rank = n_coords // world_size
        offset = rank * coords_per_rank

        size = coords_per_rank
        if rank == world_size - 1:
            size = n_coords - offset

    return (offset, size)

def gather_confidence(controller, confidence, n_grains, n_coords):
    if rank == 0:
        global_confidence = np.empty(n_grains * n_coords, dtype=np.float64)
    else:
        global_confidence = None

    # Calculate the send buffer sizes
    coords_per_rank = n_coords // world_size
    send_counts = np.full(world_size, coords_per_rank * n_grains)
    send_counts[-1] = (n_coords - (coords_per_rank * (world_size-1))) * n_grains

    if rank == 0:
        # Time how long it takes to perform the MPI gather
        controller.start('gather_confidence', 1)

    # Transpose so the data will be more easily re-shaped into its final shape
    # Must be flattened as well so the underlying data is modified...
    comm.Gatherv(confidence.T.flatten(), (global_confidence, send_counts), root=0)
    if rank == 0:
        controller.finish('gather_confidence')
        confidence = global_confidence.reshape(n_coords, n_grains).T
        controller.handle_result("confidence", confidence)

# ==============================================================================
# %% ORIENTATION TESTING
# ==============================================================================
def test_orientations(image_stack, experiment, controller):
    """grand loop precomputing the grown image stack

    image-stack -- is the dilated image stack to be tested against.

    experiment  -- A bunch of experiment related parameters.

    controller  -- An external object implementing the hooks to notify progress
                   as well as figuring out what to do with results.
    """

    # extract some information needed =========================================
    # number of grains, number of coords (maybe limited by call), projection
    # function to use, chunk size to use if multiprocessing and the number
    # of cpus.
    n_grains = experiment.n_grains
    chunk_size = controller.get_chunk_size()
    ncpus = controller.get_process_count()

    # generate angles =========================================================
    # all_angles will be a list containing arrays for the different angles to
    # use, one entry per grain.
    #
    # Note that the angle generation is driven by the exp_maps
    # in the experiment
    all_angles = evaluate_diffraction_angles(experiment, controller)

    # generate coords =========================================================
    # The grid of coords to use to test
    test_crds = generate_test_grid(-0.25, 0.25, 101)
    n_coords = controller.limit('coords', len(test_crds))

    # precompute per-grain stuff ==============================================
    # gVec_cs and rmat_ss can be precomputed, do so.
    subprocess = 'precompute gVec_cs'
    controller.start(subprocess, len(all_angles))
    precomp = []
    for i, angs in enumerate(all_angles):
        rmat_ss = xfcapi.make_sample_rmat(experiment.chi, angs[:, 2])
        gvec_cs = _anglesToGVec(angs, rmat_ss, experiment.rMat_c[i])
        precomp.append((gvec_cs, rmat_ss))
    controller.finish(subprocess)

    # Divide coords by ranks
    (offset, size) = get_offset_size(n_coords)

    # grand loop ==============================================================
    # The near field simulation 'grand loop'. Where the bulk of computing is
    # performed. We are looking for a confidence matrix that has a n_grains
    chunks = range(offset, offset+size, chunk_size)

    subprocess = 'grand_loop'
    controller.start(subprocess, n_coords)
    finished = 0
    ncpus = min(ncpus, len(chunks))

    logging.info(f'For {rank=}, {offset=}, {size=}, {chunks=}, {len(chunks)=}, {ncpus=}')

    logging.info('Checking confidence for %d coords, %d grains.',
                 n_coords, n_grains)
    confidence = np.empty((n_grains, size))
    if ncpus > 1:
        global _multiprocessing_start_method
        logging.info('Running multiprocess %d processes (%s)',
                     ncpus, _multiprocessing_start_method)
        with grand_loop_pool(ncpus=ncpus,
                             state=(chunk_size,
                                    image_stack,
                                    all_angles, precomp,
                                    test_crds, experiment)) as pool:
            for rslice, rvalues in pool.imap_unordered(multiproc_inner_loop,
                                                       chunks):
                count = rvalues.shape[1]
                # We need to adjust this slice for the offset
                rslice = slice(rslice.start - offset, rslice.stop - offset)
                confidence[:, rslice] = rvalues
                finished += count
                controller.update(finished)
    else:
        logging.info('Running in a single process')
        for chunk_start in chunks:
            chunk_stop = min(n_coords, chunk_start+chunk_size)
            rslice, rvalues = _grand_loop_inner(
                image_stack, all_angles,
                precomp, test_crds, experiment,
                start=chunk_start,
                stop=chunk_stop
            )
            count = rvalues.shape[1]
            # We need to adjust this slice for the offset
            rslice = slice(rslice.start - offset, rslice.stop - offset)
            confidence[:, rslice] = rvalues
            finished += count
            controller.update(finished)

    controller.finish(subprocess)

    # Now gather result to rank 0
    if USE_MPI:
        gather_confidence(controller, confidence, n_grains, n_coords)
    else:
        controller.handle_result("confidence", confidence)


def evaluate_diffraction_angles(experiment, controller=None):
    """Uses simulateGVecs to generate the angles used per each grain.
    returns a list containg one array per grain.

    experiment -- a bag of experiment values, including the grains specs
                  and other required parameters.
    """
    # extract required data from experiment
    exp_maps = experiment.exp_maps
    plane_data = experiment.plane_data
    detector_params = experiment.detector_params
    pixel_size = experiment.pixel_size
    ome_range = experiment.ome_range
    ome_period = experiment.ome_period

    panel_dims_expanded = [(-10, -10), (10, 10)]
    subprocess = 'evaluate diffraction angles'
    pbar = controller.start(subprocess, len(exp_maps))
    all_angles = []
    ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])
    for i, exp_map in enumerate(exp_maps):
        gparams = np.hstack([exp_map, ref_gparams])
        sim_results = xrdutil.simulateGVecs(plane_data,
                                            detector_params,
                                            gparams,
                                            panel_dims=panel_dims_expanded,
                                            pixel_pitch=pixel_size,
                                            ome_range=ome_range,
                                            ome_period=ome_period,
                                            distortion=None)
        all_angles.append(sim_results[2])
        controller.update(i + 1)
        pass
    controller.finish(subprocess)

    return all_angles


def _grand_loop_inner(image_stack, angles, precomp,
                      coords, experiment, start=0, stop=None):
    """Actual simulation code for a chunk of data. It will be used both,
    in single processor and multiprocessor cases. Chunking is performed
    on the coords.

    image_stack -- the image stack from the sensors
    angles -- the angles (grains) to test
    coords -- all the coords to test
    precomp -- (gvec_cs, rmat_ss) precomputed for each grain
    experiment -- bag with experiment parameters
    start -- chunk start offset
    stop -- chunk end offset
    """

    t = timeit.default_timer()
    n_coords = len(coords)
    n_angles = len(angles)

    # experiment geometric layout parameters
    rD = experiment.rMat_d
    rCn = experiment.rMat_c
    tD = experiment.tVec_d
    tS = experiment.tVec_s

    # experiment panel related configuration
    base = experiment.base
    inv_deltas = experiment.inv_deltas
    clip_vals = experiment.clip_vals
    distortion = experiment.distortion

    _to_detector = xfcapi.gvec_to_xy
    # _to_detector = _gvec_to_detector_array
    stop = min(stop, n_coords) if stop is not None else n_coords

    # FIXME: distortion hanlding is broken!
    distortion_fn = None
    if distortion is not None and len(distortion > 0):
        distortion_fn, distortion_args = distortion

    acc_detector = 0.0
    acc_distortion = 0.0
    acc_quant_clip = 0.0
    confidence = np.zeros((n_angles, stop-start))
    grains = 0
    crds = 0

    if distortion_fn is None:
        for igrn in range(n_angles):
            angs = angles[igrn]
            rC = rCn[igrn]
            gvec_cs, rMat_ss = precomp[igrn]
            grains += 1
            for icrd in range(start, stop):
                t0 = timeit.default_timer()
                det_xy = _to_detector(
                    gvec_cs, rD, rMat_ss, rC, tD, tS, coords[icrd]
                )
                t1 = timeit.default_timer()
                c = _quant_and_clip_confidence(det_xy, angs[:, 2], image_stack,
                                               base, inv_deltas, clip_vals)
                t2 = timeit.default_timer()
                acc_detector += t1 - t0
                acc_quant_clip += t2 - t1
                crds += 1
                confidence[igrn, icrd - start] = c
    else:
        for igrn in range(n_angles):
            angs = angles[igrn]
            rC = rCn[igrn]
            gvec_cs, rMat_ss = precomp[igrn]
            grains += 1
            for icrd in range(start, stop):
                t0 = timeit.default_timer()
                det_xy = _to_detector(
                    gvec_cs, rD, rMat_ss, rC, tD, tS, coords[icrd]
                )
                t1 = timeit.default_timer()
                det_xy = distortion_fn(tmp_xys, distortion_args, invert=True)
                t2 = timeit.default_timer()
                c = _quant_and_clip_confidence(det_xy, angs[:, 2], image_stack,
                                               base, inv_deltas, clip_vals)
                t3 = timeit.default_timer()
                acc_detector += t1 - t0
                acc_distortion += t2 - t1
                acc_quant_clip += t3 - t2
                crds += 1
                confidence[igrn, icrd - start] = c

    t = timeit.default_timer() - t
    return slice(start, stop), confidence


def generate_test_grid(low, top, samples):
    """generates a test grid of coordinates"""
    cvec_s = np.linspace(low, top, samples)
    Xs, Ys, Zs = np.meshgrid(cvec_s, cvec_s, cvec_s)
    return np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T


# Multiprocessing bits ========================================================
#
# The parallellized part of test_orientations uses some big arrays as part of
# the state that needs to be communicated to the spawn processes.
#
# On fork platforms, take advantage of process memory inheritance.
#
# On non fork platforms, rely on joblib dumping the state to disk and loading
# back in the target processes, pickling only the minimal information to load
# state back. Pickling the big arrays directly was causing memory errors and
# would be less efficient in memory (as joblib memmaps by default the big
# arrays, meaning they may be shared between processes).

def multiproc_inner_loop(chunk):
    """function to use in multiprocessing that computes the simulation over the
    task's alloted chunk of data"""

    chunk_size = _mp_state[0]
    n_coords = len(_mp_state[4])

    (offset, size) = get_offset_size(n_coords)

    chunk_stop = min(offset+size, chunk+chunk_size)
    return _grand_loop_inner(*_mp_state[1:], start=chunk, stop=chunk_stop)


def worker_init(id_state, id_exp):
    """process initialization function. This function is only used when the
    child processes are spawned (instead of forked). When using the fork model
    of multiprocessing the data is just inherited in process memory."""
    import joblib

    global _mp_state
    state = joblib.load(id_state)
    experiment = joblib.load(id_exp)
    _mp_state = state + (experiment,)


@contextlib.contextmanager
def grand_loop_pool(ncpus, state):
    """function that handles the initialization of multiprocessing. It handles
    properly the use of spawned vs forked multiprocessing. The multiprocessing
    can be either 'fork' or 'spawn', with 'spawn' being required in non-fork
    platforms (like Windows) and 'fork' being preferred on fork platforms due
    to its efficiency.
    """
    # state = ( chunk_size,
    #           image_stack,
    #           angles,
    #           precomp,
    #           coords,
    #           experiment )
    global _multiprocessing_start_method

    multiprocessing.set_start_method(_multiprocessing_start_method)

    if _multiprocessing_start_method == 'fork':
        # Use FORK multiprocessing.

        # All read-only data can be inherited in the process. So we "pass" it
        # as a global that the child process will be able to see. At the end of
        # theprocessing the global is removed.
        global _mp_state
        _mp_state = state
        pool = multiprocessing.Pool(ncpus)
        yield pool
        del (_mp_state)
    else:
        # Use SPAWN multiprocessing.

        # As we can not inherit process data, all the required data is
        # serialized into a temporary directory using joblib. The
        # multiprocessing pool will have the "worker_init" as initialization
        # function that takes the key for the serialized data, which will be
        # used to load the parameter memory into the spawn process (also using
        # joblib). In theory, joblib uses memmap for arrays if they are not
        # compressed, so no compression is used for the bigger arrays.
        import joblib
        tmp_dir = tempfile.mkdtemp(suffix='-nf-grand-loop')
        try:
            # dumb dumping doesn't seem to work very well.. do something ad-hoc
            logging.info('Using "%s" as temporary directory.', tmp_dir)

            id_exp = joblib.dump(state[-1],
                                 os.path.join(tmp_dir,
                                              'grand-loop-experiment.gz'),
                                 compress=True)
            id_state = joblib.dump(state[:-1],
                                   os.path.join(tmp_dir, 'grand-loop-data'))
            pool = multiprocessing.Pool(ncpus, worker_init,
                                        (id_state[0], id_exp[0]))
            yield pool
        finally:
            logging.info('Deleting "%s".', tmp_dir)
            shutil.rmtree(tmp_dir)


# ==============================================================================
# %% SCRIPT ENTRY AND PARAMETER HANDLING
# ==============================================================================
def main(args, controller):
    grain_params, experiment = mockup_experiment()
    controller.handle_result('experiment', experiment)
    controller.handle_result('grain_params', grain_params)
    image_stack = get_simulate_diffractions(grain_params, experiment,
                                            controller=controller)
    image_stack = get_dilated_image_stack(image_stack, experiment,
                                          controller)

    test_orientations(image_stack, experiment,
                      controller=controller)


def parse_args():
    try:
        default_ncpus = multiprocessing.cpu_count()
    except NotImplementedError:
        default_ncpus = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--inst-profile", action='append', default=[],
                        help="instrumented profile")
    parser.add_argument("--generate",
                        help="generate file with intermediate results")
    parser.add_argument("--check",
                        help="check against an file with intermediate results")
    parser.add_argument("--limit", type=int,
                        help="limit the size of the run")
    parser.add_argument("--ncpus", type=int, default=default_ncpus,
                        help="number of processes to use")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="chunk size for use in multiprocessing/reporting")
    parser.add_argument("--force-spawn-multiprocessing", action='store_true',
                        help="force using spawn as the multiprocessing method")
    args = parser.parse_args()

    '''
    keys = [
        'inst_profile',
        'generate',
        'check',
        'limit',
        'ncpus',
        'chunk_size']
    print(
        '\n'.join([': '.join([key, str(getattr(args, key))]) for key in keys])
    )
    '''
    return args


def build_controller(args):
    # builds the controller to use based on the args

    # result handle
    try:
        import progressbar
        progress_handler = progressbar_progress_observer()
    except ImportError:
        progress_handler = null_progress_observer()

    if args.check is not None:
        if args.generate is not None:
            logging.warn(
                "generating and checking can not happen at the same time, "
                + "going with checking")

        result_handler = checking_result_handler(args.check)
    elif args.generate is not None:
        result_handler = saving_result_handler(args.generate)
    else:
        result_handler = forgetful_result_handler()

    # if args.ncpus > 1 and os.name == 'nt':
    #     logging.warn("Multiprocessing on Windows is disabled for now")
    #     args.ncpus = 1

    controller = ProcessController(result_handler, progress_handler,
                                   ncpus=args.ncpus,
                                   chunk_size=args.chunk_size)
    if args.limit is not None:
        controller.set_limit('coords', lambda x: min(x, args.limit))

    return controller


# assume that if os has fork, it will be used by multiprocessing.
# note that on python > 3.4 we could use multiprocessing get_start_method and
# set_start_method for a cleaner implementation of this functionality.
_multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

if __name__ == '__main__':
    LOG_LEVEL = logging.INFO
    FORMAT="%(relativeCreated)12d [%(process)6d/%(thread)6d] %(levelname)8s: %(message)s"

    logging.basicConfig(level=LOG_LEVEL, format=FORMAT)

    # Setting the root log level via logging.basicConfig() doesn't always work.
    # The next line ensures that it will get set.
    logging.getLogger().setLevel(LOG_LEVEL)

    args = parse_args()

    if len(args.inst_profile) > 0:
        from hexrd.utils import profiler

        logging.debug("Instrumenting functions")
        profiler.instrument_all(args.inst_profile)

    if args.force_spawn_multiprocessing:
        _multiprocessing_start_method = 'spawn'

    controller = build_controller(args)
    main(args, controller)
    del controller

    if len(args.inst_profile) > 0:
        logging.debug("Dumping profiler results")
        profiler.dump_results(args.inst_profile)
