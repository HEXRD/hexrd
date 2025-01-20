"""
Refactor of simulate_nf so that an experiment is mocked up.

Also trying to minimize imports
"""

import os
import logging
import h5py

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
import copy

# import of hexrd modules
# import hexrd
from hexrd.core import constants
from hexrd.core import instrument
from hexrd.core import material
from hexrd.core import rotations
from hexrd.core.transforms import xfcapi
from hexrd.core import valunits
from hexrd.hedm import xrdutil

from skimage.morphology import dilation as ski_dilation

import matplotlib.pyplot as plt

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


# Import of image loading, this should probably be done properly with preprocessed frame-cache binaries

import scipy.ndimage as img
import skimage.filters as filters

try:
    import imageio as imgio
except(ImportError):
    from skimage import io as imgio


def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)

# %%


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
                               base, inv_deltas, clip_vals, bsp):
    """quantize and clip the parametric coordinates in coords + angles

    coords - (..., 2) array: input 2d parametric coordinates
    angles - (...) array: additional dimension for coordinates
    base   - (3,) array: base value for quantization (for each dimension)
    inv_deltas - (3,) array: inverse of the quantum size (for each dimension)
    clip_vals - (2,) array: clip size (only applied to coords dimensions)
    bsp - (2,) array: beam stop vertical position and width in mm

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


        # does not count intensity which is covered by the beamstop dcp 5.13.21
        if np.abs(yf-bsp[0])<(bsp[1]/2.):
            continue

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

        # x_byte = x // 8
        # x_off = 7 - (x % 8)
        # if image[z, y, x_byte] & (1 << x_off):
        #     matches += 1

        if image[z, y, x]:
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
        rC = xfcapi.make_rmat_of_expmap(grain_params[i][0:3])
        tC = np.ascontiguousarray(grain_params[i][3:6])
        vInv_s = np.ascontiguousarray(grain_params[i][6:12])
        ang_list = np.vstack(
            xfcapi.oscill_angles_of_hkls(
                full_hkls[:, 1:], chi, rC, bMat, wlen, v_inv=vInv_s
            )
        )
        # hkls not needed here
        all_angs, _ = xrdutil._filter_hkls_eta_ome(
            full_hkls, ang_list, eta_range, ome_range
        )
        all_angs[:, 2] = rotations.mapAngle(all_angs[:, 2], ome_period)

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
def test_orientations(image_stack, experiment, test_crds, controller, multiprocessing_start_method='fork'):
    """grand loop precomputing the grown image stack

    image-stack -- is the dilated image stack to be tested against.

    experiment  -- A bunch of experiment related parameters.

    test_crds  -- Coordinates to test orientations on, units mm.


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
        _multiprocessing_start_method=multiprocessing_start_method
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


    return confidence


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
    bsp = experiment.bsp #beam stop vertical center and width

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
                                               base, inv_deltas, clip_vals, bsp)
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
                tmp_xys = _to_detector(
                    gvec_cs, rD, rMat_ss, rC, tD, tS, coords[icrd]
                ) #changed to tmp_xys from det_xy, dcp 2021_05_30
                t1 = timeit.default_timer()
                det_xy = distortion_fn(tmp_xys, distortion_args, invert=True)
                t2 = timeit.default_timer()
                c = _quant_and_clip_confidence(det_xy, angs[:, 2], image_stack,
                                               base, inv_deltas, clip_vals,bsp)
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

    try:
        multiprocessing.set_start_method(_multiprocessing_start_method)
    except:
        print('Multiprocessing context already set')

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


# %% Test Grid Generation


def gen_nf_test_grid(cross_sectional_dim, v_bnds, voxel_spacing):

    Zs_list=np.arange(-cross_sectional_dim/2.+voxel_spacing/2.,cross_sectional_dim/2.,voxel_spacing)
    Xs_list=np.arange(-cross_sectional_dim/2.+voxel_spacing/2.,cross_sectional_dim/2.,voxel_spacing)


    if v_bnds[0]==v_bnds[1]:
        Xs,Ys,Zs=np.meshgrid(Xs_list,v_bnds[0],Zs_list)
    else:
        Xs,Ys,Zs=np.meshgrid(Xs_list,np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing),Zs_list)
        #note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))



    test_crds = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds)


    return test_crds, n_crds, Xs, Ys, Zs


def gen_nf_test_grid_tomo(x_dim_pnts, z_dim_pnts, v_bnds, voxel_spacing):

    if v_bnds[0]==v_bnds[1]:
        Xs,Ys,Zs=np.meshgrid(np.arange(x_dim_pnts),v_bnds[0],np.arange(z_dim_pnts))
    else:
        Xs,Ys,Zs=np.meshgrid(np.arange(x_dim_pnts),np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing),np.arange(z_dim_pnts))
        #note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))


    Zs=(Zs-(z_dim_pnts/2))*voxel_spacing
    Xs=(Xs-(x_dim_pnts/2))*voxel_spacing


    test_crds = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds)

    return test_crds, n_crds, Xs, Ys, Zs


# %%

def gen_nf_dark(data_folder,img_nums,num_for_dark,nrows,ncols,dark_type='median',stem='nf_',num_digits=5,ext='.tif'):

    dark_stack=np.zeros([num_for_dark,nrows,ncols])

    print('Loading data for dark generation...')
    for ii in np.arange(num_for_dark):
        print('Image #: ' + str(ii))
        dark_stack[ii,:,:]=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)
        #image_stack[ii,:,:]=np.flipud(tmp_img>threshold)

    if dark_type=='median':
        print('making median...')
        dark=np.median(dark_stack,axis=0)
    elif dark_type=='min':
        print('making min...')
        dark=np.min(dark_stack,axis=0)

    return dark


# %%


def gen_nf_cleaned_image_stack(data_folder,img_nums,dark,nrows,ncols, \
                               process_type='gaussian',process_args=[4.5,5], \
                               threshold=1.5,ome_dilation_iter=1,stem='nf_', \
                               num_digits=5,ext='.tif'):

    image_stack=np.zeros([img_nums.shape[0],nrows,ncols],dtype=bool)

    print('Loading and Cleaning Images...')


    if process_type=='gaussian':
        sigma=process_args[0]
        size=process_args[1].astype(int) #needs to be int

        for ii in np.arange(img_nums.shape[0]):
            print('Image #: ' + str(ii))
            tmp_img=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
            #image procesing
            tmp_img = filters.gaussian(tmp_img, sigma=sigma)

            tmp_img = img.morphology.grey_closing(tmp_img,size=(size,size))

            binary_img = img.morphology.binary_fill_holes(tmp_img>threshold)
            image_stack[ii,:,:]=binary_img

    else:

        num_erosions=process_args[0]
        num_dilations=process_args[1]


        for ii in np.arange(img_nums.shape[0]):
            print('Image #: ' + str(ii))
            tmp_img=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
            #image procesing
            image_stack[ii,:,:]=img.morphology.binary_erosion(tmp_img>threshold,iterations=num_erosions)
            image_stack[ii,:,:]=img.morphology.binary_dilation(image_stack[ii,:,:],iterations=num_dilations)


    #%A final dilation that includes omega
    print('Final Dilation Including Omega....')
    image_stack=img.morphology.binary_dilation(image_stack,iterations=ome_dilation_iter)


    return image_stack


# %%


def gen_trial_exp_data(grain_out_file,det_file,mat_file, mat_name, max_tth, comp_thresh, chi2_thresh, misorientation_bnd, \
                       misorientation_spacing,ome_range_deg, nframes, beam_stop_parms):


    print('Loading Grain Data...')
    #gen_grain_data
    ff_data=np.loadtxt(grain_out_file)

    #ff_data=np.atleast_2d(ff_data[2,:])

    exp_maps=ff_data[:,3:6]
    t_vec_ds=ff_data[:,6:9]


    #
    completeness=ff_data[:,1]

    chi2=ff_data[:,2]

    n_grains=exp_maps.shape[0]

    rMat_c = rotations.rotMatOfExpMap(exp_maps.T)

    cut=np.where(np.logical_and(completeness>comp_thresh,chi2<chi2_thresh))[0]
    exp_maps=exp_maps[cut,:]
    t_vec_ds=t_vec_ds[cut,:]
    chi2=chi2[cut]


    # Add Misorientation
    mis_amt=misorientation_bnd*np.pi/180.
    spacing=misorientation_spacing*np.pi/180.

    mis_steps = int(misorientation_bnd/misorientation_spacing)

    ori_pts = np.arange(-mis_amt, (mis_amt+(spacing*0.999)),spacing)
    num_ori_grid_pts=ori_pts.shape[0]**3
    num_oris=exp_maps.shape[0]


    XsO, YsO, ZsO = np.meshgrid(ori_pts, ori_pts, ori_pts)

    grid0 = np.vstack([XsO.flatten(), YsO.flatten(), ZsO.flatten()]).T


    exp_maps_expanded=np.zeros([num_ori_grid_pts*num_oris,3])
    t_vec_ds_expanded=np.zeros([num_ori_grid_pts*num_oris,3])


    for ii in np.arange(num_oris):
        pts_to_use=np.arange(num_ori_grid_pts)+ii*num_ori_grid_pts
        exp_maps_expanded[pts_to_use,:]=grid0+np.r_[exp_maps[ii,:] ]
        t_vec_ds_expanded[pts_to_use,:]=np.r_[t_vec_ds[ii,:] ]


    exp_maps=exp_maps_expanded
    t_vec_ds=t_vec_ds_expanded

    n_grains=exp_maps.shape[0]

    rMat_c = rotations.rotMatOfExpMap(exp_maps.T)


    print('Loading Instrument Data...')
    ome_period_deg=(ome_range_deg[0][0], (ome_range_deg[0][0]+360.)) #degrees
    ome_step_deg=(ome_range_deg[0][1]-ome_range_deg[0][0])/nframes #degrees


    ome_period = (ome_period_deg[0]*np.pi/180.,ome_period_deg[1]*np.pi/180.)
    ome_range = [(ome_range_deg[0][0]*np.pi/180.,ome_range_deg[0][1]*np.pi/180.)]
    ome_step = ome_step_deg*np.pi/180.



    ome_edges = np.arange(nframes+1)*ome_step+ome_range[0][0]#fixed 2/26/17


    instr=load_instrument(det_file)
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

    # # dilation
    # max_diameter = np.sqrt(3)*0.005
    # row_dilation = int(np.ceil(0.5 * max_diameter/row_ps))
    # col_dilation = int(np.ceil(0.5 * max_diameter/col_ps))



    print('Loading Materials Data...')
    # crystallography data
    beam_energy = valunits.valWUnit("beam_energy", "energy", instr.beam_energy, "keV")
    beam_wavelength = constants.keVToAngstrom(beam_energy.getVal('keV'))
    dmin = valunits.valWUnit("dmin", "length",
                             0.5*beam_wavelength/np.sin(0.5*max_pixel_tth),
                             "angstrom")

    # material loading
    mats = material.load_materials_hdf5(mat_file, dmin=dmin,kev=beam_energy)
    pd = mats[mat_name].planeData

    if max_tth is not None:
         pd.tThMax = np.amax(np.radians(max_tth))
    else:
        pd.tThMax = np.amax(max_pixel_tth)



    print('Final Assembly...')
    experiment = argparse.Namespace()
    # grains related information
    experiment.n_grains = n_grains  # this can be derived from other values...
    experiment.rMat_c = rMat_c  # n_grains rotation matrices (one per grain)
    experiment.exp_maps = exp_maps  # n_grains exp_maps (one per grain)

    experiment.plane_data = pd
    experiment.detector_params = detector_params
    experiment.pixel_size = pixel_size
    experiment.ome_range = ome_range
    experiment.ome_period = ome_period
    experiment.x_col_edges = x_col_edges
    experiment.y_row_edges = y_row_edges
    experiment.ome_edges = ome_edges
    experiment.ncols = ncols
    experiment.nrows = nrows
    experiment.nframes = nframes  # used only in simulate...
    experiment.rMat_d = rMat_d
    experiment.tVec_d = tVec_d
    experiment.chi = chi  # note this is used to compute S... why is it needed?
    experiment.tVec_s = tVec_s
    experiment.rMat_c = rMat_c
    # ns.row_dilation = 0 #done beforehand row_dilation, disabled
    # experiemnt.col_dilation = 0 #col_dilation
    experiment.distortion = distortion
    experiment.panel_dims = panel_dims  # used only in simulate...
    experiment.base = base
    experiment.inv_deltas = inv_deltas
    experiment.clip_vals = clip_vals
    experiment.bsp = beam_stop_parms


    if mis_steps ==0:
        nf_to_ff_id_map = cut
    else:
        nf_to_ff_id_map=np.tile(cut,3**3*mis_steps)

    return experiment, nf_to_ff_id_map

def process_raw_confidence(raw_confidence,vol_shape=None,id_remap=None,min_thresh=0.0):

    print('Compiling Confidence Map...')
    if vol_shape == None:
        confidence_map=np.max(raw_confidence,axis=0)
        grain_map=np.argmax(raw_confidence,axis=0)
    else:
        confidence_map=np.max(raw_confidence,axis=0).reshape(vol_shape)
        grain_map=np.argmax(raw_confidence,axis=0).reshape(vol_shape)


    #fix grain indexing
    not_indexed=np.where(confidence_map<=min_thresh)
    grain_map[not_indexed] =-1


    if id_remap is not None:
        max_grain_no=np.max(grain_map)
        grain_map_copy=copy.copy(grain_map)
        print('Remapping grain ids to ff...')
        for ii in np.arange(max_grain_no):
            this_grain=np.where(grain_map==ii)
            grain_map_copy[this_grain]=id_remap[ii]
        grain_map=grain_map_copy

    return grain_map.astype(int), confidence_map


# %%
def save_raw_confidence(save_dir,save_stem,raw_confidence,id_remap=None):
    print('Saving raw confidence, might take a while...')
    if id_remap is not None:
        np.savez(save_dir+save_stem+'_raw_confidence.npz',raw_confidence=raw_confidence,id_remap=id_remap)
    else:
        np.savez(save_dir+save_stem+'_raw_confidence.npz',raw_confidence=raw_confidence)
# %%

def save_nf_data(save_dir,save_stem,grain_map,confidence_map,Xs,Ys,Zs,ori_list,id_remap=None):
    print('Saving grain map data...')
    if id_remap is not None:
        np.savez(save_dir+save_stem+'_grain_map_data.npz',grain_map=grain_map,confidence_map=confidence_map,Xs=Xs,Ys=Ys,Zs=Zs,ori_list=ori_list,id_remap=id_remap)
    else:
        np.savez(save_dir+save_stem+'_grain_map_data.npz',grain_map=grain_map,confidence_map=confidence_map,Xs=Xs,Ys=Ys,Zs=Zs,ori_list=ori_list)


# %%

def scan_detector_parm(image_stack, experiment,test_crds,controller,parm_to_opt,parm_range,slice_shape,ang='deg'):
    # 0-distance
    # 1-x center
    # 2-y center
    # 3-xtilt
    # 4-ytilt
    # 5-ztilt

    parm_vector=np.arange(parm_range[0],parm_range[1]+1e-6,(parm_range[1]-parm_range[0])/parm_range[2])

    if parm_to_opt>2 and ang=='deg':
        parm_vector=parm_vector*np.pi/180.

    multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

    # current detector parameters, note the value for the actively optimized
    # parameters will be ignored
    distance=experiment.detector_params[5]#mm
    x_cen=experiment.detector_params[3]#mm
    y_cen=experiment.detector_params[4]#mm
    xtilt=experiment.detector_params[0]
    ytilt=experiment.detector_params[1]
    ztilt=experiment.detector_params[2]
    ome_range=copy.copy(experiment.ome_range)
    ome_period=copy.copy(experiment.ome_period)
    ome_edges=copy.copy(experiment.ome_edges)

    num_parm_pts=len(parm_vector)

    trial_data=np.zeros([num_parm_pts,slice_shape[0],slice_shape[1]])

    tmp_td=copy.copy(experiment.tVec_d)
    for jj in np.arange(num_parm_pts):
        print('cycle %d of %d'%(jj+1,num_parm_pts))

        # overwrite translation vector components
        if parm_to_opt==0:
            tmp_td[2]=parm_vector[jj]

        if parm_to_opt==1:
            tmp_td[0]=parm_vector[jj]

        if parm_to_opt==2:
            tmp_td[1]=parm_vector[jj]

        if parm_to_opt == 3:
            rMat_d_tmp = xfcapi.make_detector_rmat(
                [parm_vector[jj], ytilt, ztilt]
            )
        elif parm_to_opt == 4:
            rMat_d_tmp = xfcapi.make_detector_rmat(
                [xtilt, parm_vector[jj], ztilt]
            )
        elif parm_to_opt == 5:
            rMat_d_tmp = xfcapi.make_detector_rmat(
                [xtilt, ytilt, parm_vector[jj]]
            )
        else:
            rMat_d_tmp = xfcapi.make_detector_rmat([xtilt, ytilt, ztilt])

        experiment.rMat_d = rMat_d_tmp
        experiment.tVec_d = tmp_td

        if parm_to_opt==6:

            experiment.ome_range = [
                (
                    ome_range[0][0] - parm_vector[jj],
                    ome_range[0][1] - parm_vector[jj],
                )
            ]
            experiment.ome_period = (
                ome_period[0] - parm_vector[jj],
                ome_period[1] - parm_vector[jj],
            )
            experiment.ome_edges = np.array(ome_edges - parm_vector[jj])
            experiment.base[2] = experiment.ome_edges[0]

            # print(experiment.ome_range)
            # print(experiment.ome_period)
            # print(experiment.ome_edges)
            # print(experiment.base)

        conf=test_orientations(image_stack, experiment,test_crds,controller, \
                               multiprocessing_start_method)

        trial_data[jj]=np.max(conf,axis=0).reshape(slice_shape)

    return trial_data, parm_vector

# %%

def plot_ori_map(grain_map, confidence_map, exp_maps, layer_no,mat,id_remap=None):

    grains_plot=np.squeeze(grain_map[layer_no,:,:])
    conf_plot=np.squeeze(confidence_map[layer_no,:,:])
    n_grains=len(exp_maps)

    rgb_image = np.zeros(
        [grains_plot.shape[0], grains_plot.shape[1], 4], dtype='float32')
    rgb_image[:, :, 3] = 1.

    for ii in np.arange(n_grains):
        if id_remap is not None:
            this_grain = np.where(np.squeeze(grains_plot) == id_remap[ii])
        else:
            this_grain = np.where(np.squeeze(grains_plot) == ii)
        if np.sum(this_grain[0]) > 0:

            ori = exp_maps[ii, :]

            rmats = rotations.rotMatOfExpMap(ori)
            rgb = mat.unitcell.color_orientations(
                rmats, ref_dir=np.array([0., 1., 0.]))

            #color mapping
            rgb_image[this_grain[0], this_grain[1], 0] = rgb[0][0]
            rgb_image[this_grain[0], this_grain[1], 1] = rgb[0][1]
            rgb_image[this_grain[0], this_grain[1], 2] = rgb[0][2]



    fig1 = plt.figure()
    plt.imshow(rgb_image, interpolation='none')
    plt.title('Layer %d Grain Map' % layer_no)
    #plt.show()
    plt.hold(True)
    #fig2 = plt.figure()
    plt.imshow(conf_plot, vmin=0.0, vmax=1.,
               interpolation='none', cmap=plt.cm.gray, alpha=0.5)
    plt.title('Layer %d Confidence Map' % layer_no)
    plt.show()
# ==============================================================================
# %% SCRIPT ENTRY AND PARAMETER HANDLING
# ==============================================================================
# def main(args, controller):
#     grain_params, experiment = mockup_experiment()
#     controller.handle_result('experiment', experiment)
#     controller.handle_result('grain_params', grain_params)
#     image_stack = get_simulate_diffractions(grain_params, experiment,
#                                             controller=controller)
#     image_stack = get_dilated_image_stack(image_stack, experiment,
#                                           controller)

#     test_orientations(image_stack, experiment,
#                       controller=controller)


# def parse_args():
#     try:
#         default_ncpus = multiprocessing.cpu_count()
#     except NotImplementedError:
#         default_ncpus = 1

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--inst-profile", action='append', default=[],
#                         help="instrumented profile")
#     parser.add_argument("--generate",
#                         help="generate file with intermediate results")
#     parser.add_argument("--check",
#                         help="check against an file with intermediate results")
#     parser.add_argument("--limit", type=int,
#                         help="limit the size of the run")
#     parser.add_argument("--ncpus", type=int, default=default_ncpus,
#                         help="number of processes to use")
#     parser.add_argument("--chunk-size", type=int, default=100,
#                         help="chunk size for use in multiprocessing/reporting")
#     parser.add_argument("--force-spawn-multiprocessing", action='store_true',
#                         help="force using spawn as the multiprocessing method")
#     args = parser.parse_args()

#     '''
#     keys = [
#         'inst_profile',
#         'generate',
#         'check',
#         'limit',
#         'ncpus',
#         'chunk_size']
#     print(
#         '\n'.join([': '.join([key, str(getattr(args, key))]) for key in keys])
#     )
#     '''
#     return args


def build_controller(check=None,generate=None,ncpus=2,chunk_size=10,limit=None):
    # builds the controller to use based on the args

    # result handle
    try:
        import progressbar
        progress_handler = progressbar_progress_observer()
    except ImportError:
        progress_handler = null_progress_observer()

    if check is not None:
        if generate is not None:
            logging.warn(
                "generating and checking can not happen at the same time, "
                + "going with checking")

        result_handler = checking_result_handler(check)
    elif generate is not None:
        result_handler = saving_result_handler(generate)
    else:
        result_handler = forgetful_result_handler()

    # if args.ncpus > 1 and os.name == 'nt':
    #     logging.warn("Multiprocessing on Windows is disabled for now")
    #     args.ncpus = 1

    controller = ProcessController(result_handler, progress_handler,
                                   ncpus=ncpus,
                                   chunk_size=chunk_size)
    if limit is not None:
        controller.set_limit('coords', lambda x: min(x, limit))

    return controller

def output_grain_map(data_location,data_stems,output_stem,vol_spacing,top_down=True,save_type=['npz']):

    num_scans=len(data_stems)

    confidence_maps=[None]*num_scans
    grain_maps=[None]*num_scans
    Xss=[None]*num_scans
    Yss=[None]*num_scans
    Zss=[None]*num_scans

    if len(vol_spacing)==1:
        vol_shifts=np.arange(0,vol_spacing[0]*num_scans+1e-12,vol_spacing[0])
    else:
        vol_shifts=vol_spacing


    for ii in np.arange(num_scans):
        print('Loading Volume %d ....'%(ii))
        conf_data=np.load(os.path.join(data_location,data_stems[ii]+'_grain_map_data.npz'))

        confidence_maps[ii]=conf_data['confidence_map']
        grain_maps[ii]=conf_data['grain_map']
        Xss[ii]=conf_data['Xs']
        Yss[ii]=conf_data['Ys']
        Zss[ii]=conf_data['Zs']

    #assumes all volumes to be the same size
    num_layers=grain_maps[0].shape[0]

    total_layers=num_layers*num_scans

    num_rows=grain_maps[0].shape[1]
    num_cols=grain_maps[0].shape[2]

    grain_map_stitched=np.zeros((total_layers,num_rows,num_cols))
    confidence_stitched=np.zeros((total_layers,num_rows,num_cols))
    Xs_stitched=np.zeros((total_layers,num_rows,num_cols))
    Ys_stitched=np.zeros((total_layers,num_rows,num_cols))
    Zs_stitched=np.zeros((total_layers,num_rows,num_cols))


    for ii in np.arange(num_scans):
        if top_down==True:
            grain_map_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=grain_maps[num_scans-1-ii]
            confidence_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=confidence_maps[num_scans-1-ii]
            Xs_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=\
                Xss[num_scans-1-ii]
            Zs_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=\
                Zss[num_scans-1-ii]
            Ys_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=Yss[num_scans-1-ii]+vol_shifts[ii]
        else:

            grain_map_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=grain_maps[ii]
            confidence_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=confidence_maps[ii]
            Xs_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=Xss[ii]
            Zs_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=Zss[ii]
            Ys_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=Yss[ii]+vol_shifts[ii]

    for ii in np.arange(len(save_type)):

        if save_type[ii] == 'hdf5':

            print('Writing HDF5 data...')

            hf = h5py.File(output_stem + '_assembled.h5', 'w')
            hf.create_dataset('grain_map', data=grain_map_stitched)
            hf.create_dataset('confidence', data=confidence_stitched)
            hf.create_dataset('Xs', data=Xs_stitched)
            hf.create_dataset('Ys', data=Ys_stitched)
            hf.create_dataset('Zs', data=Zs_stitched)

        elif save_type[ii]=='npz':

            print('Writing NPZ data...')

            np.savez(output_stem + '_assembled.npz',\
             grain_map=grain_map_stitched,confidence=confidence_stitched,
             Xs=Xs_stitched,Ys=Ys_stitched,Zs=Zs_stitched)

        elif save_type[ii]=='vtk':


            print('Writing VTK data...')
            # VTK Dump
            Xslist=Xs_stitched[:,:,:].ravel()
            Yslist=Ys_stitched[:,:,:].ravel()
            Zslist=Zs_stitched[:,:,:].ravel()

            grainlist=grain_map_stitched[:,:,:].ravel()
            conflist=confidence_stitched[:,:,:].ravel()

            num_pts=Xslist.shape[0]
            num_cells=(total_layers-1)*(num_rows-1)*(num_cols-1)

            f = open(os.path.join(output_stem +'_assembled.vtk'), 'w')


            f.write('# vtk DataFile Version 3.0\n')
            f.write('grainmap Data\n')
            f.write('ASCII\n')
            f.write('DATASET UNSTRUCTURED_GRID\n')
            f.write('POINTS %d double\n' % (num_pts))

            for i in np.arange(num_pts):
                f.write('%e %e %e \n' %(Xslist[i],Yslist[i],Zslist[i]))

            scale2=num_cols*num_rows
            scale1=num_cols

            f.write('CELLS %d %d\n' % (num_cells, 9*num_cells))
            for k in np.arange(Xs_stitched.shape[0]-1):
                for j in np.arange(Xs_stitched.shape[1]-1):
                    for i in np.arange(Xs_stitched.shape[2]-1):
                        base=scale2*k+scale1*j+i
                        p1=base
                        p2=base+1
                        p3=base+1+scale1
                        p4=base+scale1
                        p5=base+scale2
                        p6=base+scale2+1
                        p7=base+scale2+scale1+1
                        p8=base+scale2+scale1

                        f.write('8 %d %d %d %d %d %d %d %d \n' \
                                %(p1,p2,p3,p4,p5,p6,p7,p8))


            f.write('CELL_TYPES %d \n' % (num_cells))
            for i in np.arange(num_cells):
                f.write('12 \n')

            f.write('POINT_DATA %d \n' % (num_pts))
            f.write('SCALARS grain_id int \n')
            f.write('LOOKUP_TABLE default \n')
            for i in np.arange(num_pts):
                f.write('%d \n' %(grainlist[i]))

            f.write('FIELD FieldData 1 \n' )
            f.write('confidence 1 %d float \n' % (num_pts))
            for i in np.arange(num_pts):
                f.write('%e \n' %(conflist[i]))


            f.close()

        else:
            print('Not a valid save option, npz, vtk, or hdf5 allowed.')

    return grain_map_stitched, confidence_stitched, Xs_stitched, Ys_stitched, \
            Zs_stitched


# # assume that if os has fork, it will be used by multiprocessing.
# # note that on python > 3.4 we could use multiprocessing get_start_method and
# # set_start_method for a cleaner implementation of this functionality.
# _multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

# if __name__ == '__main__':
#     LOG_LEVEL = logging.INFO
#     FORMAT="%(relativeCreated)12d [%(process)6d/%(thread)6d] %(levelname)8s: %(message)s"

#     logging.basicConfig(level=LOG_LEVEL, format=FORMAT)

#     # Setting the root log level via logging.basicConfig() doesn't always work.
#     # The next line ensures that it will get set.
#     logging.getLogger().setLevel(LOG_LEVEL)

#     args = parse_args()

#     if len(args.inst_profile) > 0:
#         from hexrd.core.utils import profiler

#         logging.debug("Instrumenting functions")
#         profiler.instrument_all(args.inst_profile)

#     if args.force_spawn_multiprocessing:
#         _multiprocessing_start_method = 'spawn'

#     controller = build_controller(args)
#     main(args, controller)
#     del controller

#     if len(args.inst_profile) > 0:
#         logging.debug("Dumping profiler results")
#         profiler.dump_results(args.inst_profile)
