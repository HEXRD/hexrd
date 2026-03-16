from dataclasses import dataclass
import copy
import logging
import os
import timeit

from typing import Optional

import numpy as np
from numpy.typing import NDArray

# np.seterr(over='ignore', invalid='ignore')

# import tqdm

import scipy.cluster as cluster
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from hexrd.core import constants as const
from hexrd.core import matrixutil as mutil
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.core.material.crystallography import PlaneData
from hexrd.hedm import indexer
from hexrd.core import instrument
from hexrd.core.imageutil import find_peaks_2d
from hexrd.core import rotations as rot
from hexrd.core.transforms import xfcapi
from hexrd.core.utils.concurrent import distribute_tasks
from hexrd.hedm.xrdutil import EtaOmeMaps
from hexrd.hedm.config import root

# just require scikit-learn?
have_sklearn = False
try:
    from sklearn.cluster import dbscan
    from sklearn.metrics.pairwise import pairwise_distances

    have_sklearn = True
except ImportError:
    pass


save_as_ascii = False  # FIXME LATER...
filter_stdev_DFLT = 1.0

logger = logging.getLogger(__name__)
pairwiseConsensusMP = None
reflectionStatisticsMP = None


def _pool_chunksize(num_items, ncpus, max_chunksize=10):
    num_items = int(num_items)
    ncpus = max(int(ncpus), 1)
    if num_items <= 0:
        return 1

    return max(1, num_items // (max_chunksize * ncpus))


def _spawn_pool_is_expensive() -> bool:
    return const.mp_context.get_start_method() == 'spawn'


def _pool_worker_count(num_items, requested_ncpus) -> int:
    return min(max(int(requested_ncpus), 1), max(int(num_items), 1))


# =============================================================================
# FUNCTIONS
# =============================================================================


@dataclass
class SeedReflection:
    seed_index: int
    hkl_id: int
    hkl: NDArray[np.float64]
    tth: float
    eta: float
    ome: float
    gvec_s: NDArray[np.float64]
    intensity: float = 0.0
    support: int = 1


@dataclass
class SeedPeak:
    eta: float
    ome: float
    intensity: float
    support: int = 1


@dataclass
class ReflectionStatistics:
    sample_count: int
    active_reflections_per_grain: NDArray[np.int64]
    seed_reflections_raw_per_grain: NDArray[np.int64]
    seed_reflections_reduced_per_grain: NDArray[np.int64]
    seed_hkls_per_grain: NDArray[np.int64]
    seed_reflections_raw_by_hkl: dict[int, NDArray[np.int64]]
    seed_reflections_reduced_by_hkl: dict[int, NDArray[np.int64]]

    def seed_reflections_per_grain(
        self,
        use_friedel_pairing: bool,
    ) -> NDArray[np.int64]:
        if use_friedel_pairing:
            return self.seed_reflections_reduced_per_grain

        return self.seed_reflections_raw_per_grain

    def seed_reflections_by_hkl(
        self,
        use_friedel_pairing: bool,
    ) -> dict[int, NDArray[np.int64]]:
        if use_friedel_pairing:
            return self.seed_reflections_reduced_by_hkl

        return self.seed_reflections_raw_by_hkl


def _wrapped_angle_difference(angle_0, angle_1) -> float:
    return float(
        rot.angularDifference(
            np.asarray([angle_0], dtype=float),
            np.asarray([angle_1], dtype=float),
        )[0]
    )


def _rotate_vectors_about_axis(
    vecs: NDArray[np.float64],
    axis: NDArray[np.float64],
    angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis /= np.linalg.norm(axis)

    vecs = np.asarray(vecs, dtype=float)
    angles = np.asarray(angles, dtype=float)

    cos_ang = np.cos(angles)
    sin_ang = np.sin(angles)
    cross_term = np.cross(np.tile(axis, (vecs.shape[1], 1)), vecs.T).T
    dot_term = np.dot(axis, vecs)
    return (
        vecs * cos_ang
        + cross_term * sin_ang
        + axis.reshape(3, 1) * dot_term * (1.0 - cos_ang)
    )


def _predict_friedel_pair_angles(
    tth: NDArray[np.float64] | float,
    eta0: NDArray[np.float64],
    ome0: NDArray[np.float64],
    chi: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    eta0 = rot.mapAngle(np.asarray(eta0, dtype=float), [-np.pi, np.pi])
    ome0 = rot.mapAngle(np.asarray(ome0, dtype=float), [-np.pi, np.pi])
    tth = np.asarray(tth, dtype=float)
    tth, eta0, ome0 = np.broadcast_arrays(tth, eta0, ome0)

    tht0 = 0.5 * tth

    cchi = np.cos(chi)
    schi = np.sin(chi)
    ceta = np.cos(eta0)
    seta = np.sin(eta0)
    ctht = np.cos(tht0)
    stht = np.sin(tht0)

    a = cchi * ceta * ctht
    b = -cchi * stht
    c = stht + schi * seta * ctht

    ab_mag = np.sqrt(a * a + b * b)
    if np.any(ab_mag <= const.sqrt_epsf):
        raise RuntimeError("Beam vector specification is infeasible!")

    phase_ang = np.arctan2(b, a)
    rhs = c / ab_mag
    invalid = np.abs(rhs) > 1.0
    rhs = np.clip(rhs, -1.0, 1.0)
    rhs_ang = np.arcsin(rhs)

    ome_1 = rot.mapAngle(rhs_ang - phase_ang, [-np.pi, np.pi])
    ome_2 = rot.mapAngle(np.pi - rhs_ang - phase_ang, [-np.pi, np.pi])

    ome_stack = np.vstack([ome_1, ome_2])
    min_idx = np.argmin(np.abs(ome_stack), axis=0)
    ome_delta = ome_stack[min_idx, np.arange(ome_stack.shape[1])]
    ome_delta[invalid] = np.nan

    axis = np.array([0.0, cchi, schi], dtype=float)
    ghat0_l = -np.vstack([ceta * ctht, seta * ctht, stht])
    rotated_gvecs = _rotate_vectors_about_axis(ghat0_l, axis, ome_delta)

    partner_ome = rot.mapAngle(ome0 + ome_delta, [-np.pi, np.pi])
    partner_eta = rot.mapAngle(
        np.arctan2(rotated_gvecs[1], rotated_gvecs[0]),
        [-np.pi, np.pi],
    )
    partner_ome[invalid] = np.nan
    partner_eta[invalid] = np.nan
    return partner_ome, partner_eta


def _pair_friedel_seed_peaks(
    peaks: list[SeedPeak],
    tth: float,
    chi: float,
    eta_tol: float,
    ome_tol: float,
) -> list[SeedPeak]:
    if len(peaks) < 2:
        return peaks

    eta_tol = max(float(eta_tol), np.radians(0.25))
    ome_tol = max(float(ome_tol), np.radians(0.25))

    etas = rot.mapAngle(
        np.asarray([peak.eta for peak in peaks], dtype=float),
        [-np.pi, np.pi],
    )
    omes = rot.mapAngle(
        np.asarray([peak.ome for peak in peaks], dtype=float),
        [-np.pi, np.pi],
    )
    intensities = np.asarray(
        [max(float(peak.intensity), const.sqrt_epsf) for peak in peaks],
        dtype=float,
    )

    pred_omes, pred_etas = _predict_friedel_pair_angles(tth, etas, omes, chi=chi)
    valid = ~(np.isnan(pred_omes) | np.isnan(pred_etas))
    if np.count_nonzero(valid) < 2:
        return peaks

    scale = np.array([eta_tol, ome_tol], dtype=float)
    obs_coords = np.column_stack([etas, omes]) / scale

    periodic_shifts = np.array(
        [
            [deta, dome]
            for deta in (-2.0 * np.pi, 0.0, 2.0 * np.pi)
            for dome in (-2.0 * np.pi, 0.0, 2.0 * np.pi)
        ],
        dtype=float,
    ) / scale
    tiled_coords = np.vstack([obs_coords + shift for shift in periodic_shifts])
    source_ids = np.tile(np.arange(len(peaks)), len(periodic_shifts))
    tree = cKDTree(tiled_coords)

    best_match = np.full(len(peaks), -1, dtype=int)
    best_cost = np.full(len(peaks), np.inf)
    search_radius = np.sqrt(2.0)
    intensity_weight = 0.05

    for i in np.where(valid)[0]:
        pred_point = np.array([pred_etas[i], pred_omes[i]], dtype=float) / scale
        candidate_ids = np.unique(source_ids[tree.query_ball_point(pred_point, search_radius)])

        for j in candidate_ids:
            if i == j or not valid[j]:
                continue

            forward_eta = _wrapped_angle_difference(pred_etas[i], etas[j])
            forward_ome = _wrapped_angle_difference(pred_omes[i], omes[j])
            if forward_eta > eta_tol or forward_ome > ome_tol:
                continue

            reverse_eta = _wrapped_angle_difference(pred_etas[j], etas[i])
            reverse_ome = _wrapped_angle_difference(pred_omes[j], omes[i])
            if reverse_eta > eta_tol or reverse_ome > ome_tol:
                continue

            cost = (
                (forward_eta / eta_tol) ** 2
                + (forward_ome / ome_tol) ** 2
                + (reverse_eta / eta_tol) ** 2
                + (reverse_ome / ome_tol) ** 2
                + intensity_weight
                * abs(np.log((intensities[i] + 1.0) / (intensities[j] + 1.0)))
            )
            if cost < best_cost[i]:
                best_cost[i] = cost
                best_match[i] = j

    reduced_peaks = []
    used = np.zeros(len(peaks), dtype=bool)

    for i in np.argsort(best_cost):
        if used[i]:
            continue

        j = best_match[i]
        if j < 0 or used[j] or best_match[j] != i:
            continue

        rep_idx = i if intensities[i] >= intensities[j] else j
        rep_peak = peaks[rep_idx]
        reduced_peaks.append(
            SeedPeak(
                eta=rep_peak.eta,
                ome=rep_peak.ome,
                intensity=float(peaks[i].intensity + peaks[j].intensity),
                support=peaks[i].support + peaks[j].support,
            )
        )
        used[i] = True
        used[j] = True

    for i, peak in enumerate(peaks):
        if not used[i]:
            reduced_peaks.append(peak)

    return reduced_peaks


def write_scored_orientations(results, cfg):
    """Write scored orientations to a file

    PARAMETERS
    ----------
    results: dict
       output of main `find_orientations` function
    cfg: Config instance
       the main Config input file for `find-orientations`
    """
    np.savez_compressed(
        cfg.find_orientations.orientation_maps.scored_orientations_file,
        **results['scored_orientations'],
    )


def _process_omegas(
    omegaimageseries_dict: dict[str, OmegaImageSeries]
) -> tuple[NDArray[np.float64], list[list[int]]]:
    """Extract omega period and ranges from an OmegaImageseries dictionary."""
    oims = next(iter(omegaimageseries_dict.values()))
    ome_period = oims.omega[0, 0] + np.r_[0.0, 360.0]
    ome_ranges = [([i['ostart'], i['ostop']]) for i in oims.omegawedges.wedges]
    return ome_period, ome_ranges


def clean_map(this_map):
    # !!! need to remove NaNs from map in case of eta gaps
    # !!! doing offset and truncation by median value now
    nan_mask = np.isnan(this_map)
    med_val = np.median(this_map[~nan_mask])
    this_map[nan_mask] = med_val
    this_map[this_map <= med_val] = med_val
    this_map -= np.min(this_map)


def _seed_pairing_tolerances(cfg, eta_ome) -> tuple[float, float]:
    del_ome = eta_ome.omegas[1] - eta_ome.omegas[0]
    del_eta = eta_ome.etas[1] - eta_ome.etas[0]
    pair_eta_tol = max(
        abs(del_eta),
        np.radians(cfg.find_orientations.eta.tolerance),
    )
    pair_ome_tol = max(
        abs(del_ome),
        np.radians(cfg.find_orientations.omega.tolerance),
    )
    return float(pair_eta_tol), float(pair_ome_tol)


def _collect_seed_reflections(cfg, eta_ome):
    chi = cfg.instrument.hedm.chi
    seed_hkl_ids = cfg.find_orientations.seed_search.hkl_seeds
    method_dict = cfg.find_orientations.seed_search.method
    use_friedel_pairing = cfg.find_orientations.seed_search.friedel_pairing

    method = next(iter(method_dict.keys()))
    method_kwargs = method_dict[method]
    logger.debug('\tusing "%s" method for fiber generation' % method)

    pd = eta_ome.planeData
    tTh = pd.getTTh()
    bMat = pd.latVecOps['B']
    csym = pd.laueGroup
    instr = cfg.instrument.hedm
    eta_ranges = np.radians(cfg.find_orientations.eta.range)
    ome_period, ome_ranges = _process_omegas(cfg.image_series)

    pd_hkl_ids = eta_ome.iHKLList[seed_hkl_ids]
    pd_hkl_idx = pd.getHKLID(pd.getHKLs(*eta_ome.iHKLList).T, master=False)
    seed_hkls = pd.getHKLs(*pd_hkl_ids)
    seed_tths = tTh[pd_hkl_idx][seed_hkl_ids]
    logger.info('\tusing seed hkls: %s' % [str(i) for i in seed_hkls])

    del_ome = eta_ome.omegas[1] - eta_ome.omegas[0]
    del_eta = eta_ome.etas[1] - eta_ome.etas[0]
    pair_eta_tol, pair_ome_tol = _seed_pairing_tolerances(cfg, eta_ome)

    sym_hkls = pd.getSymHKLs()
    seed_crystal_dirs = []
    for i in seed_hkl_ids:
        crystal_dirs = mutil.unitVector(np.dot(bMat, sym_hkls[pd_hkl_idx[i]]))
        crystal_dirs = np.asarray(crystal_dirs)
        if crystal_dirs.ndim == 1:
            crystal_dirs = crystal_dirs.reshape(3, 1)
        seed_crystal_dirs.append(crystal_dirs)

    reflections = []
    total_raw_spots = 0
    total_reduced_spots = 0
    for seed_index, (active_hkl_index, this_hkl, this_tth) in enumerate(
        zip(seed_hkl_ids, seed_hkls, seed_tths)
    ):
        this_map = np.array(eta_ome.dataStore[active_hkl_index], copy=True)
        clean_map(this_map)
        num_spots, coms = find_peaks_2d(this_map, method, method_kwargs)
        seed_peaks = []
        for ispot in range(num_spots):
            if np.isnan(coms[ispot][0]):
                continue

            ome_c = eta_ome.omeEdges[0] + (0.5 + coms[ispot][0]) * del_ome
            eta_c = eta_ome.etaEdges[0] + (0.5 + coms[ispot][1]) * del_eta
            ome_idx = int(np.clip(np.rint(coms[ispot][0]), 0, this_map.shape[0] - 1))
            eta_idx = int(np.clip(np.rint(coms[ispot][1]), 0, this_map.shape[1] - 1))
            seed_peaks.append(
                SeedPeak(
                    eta=float(eta_c),
                    ome=float(ome_c),
                    intensity=float(this_map[ome_idx, eta_idx]),
                )
            )

        total_raw_spots += len(seed_peaks)
        if use_friedel_pairing:
            seed_peaks = _pair_friedel_seed_peaks(
                seed_peaks,
                float(this_tth),
                chi,
                pair_eta_tol,
                pair_ome_tol,
            )
        total_reduced_spots += len(seed_peaks)

        for seed_peak in seed_peaks:
            gvec_s = xfcapi.angles_to_gvec(
                np.atleast_2d([this_tth, seed_peak.eta, seed_peak.ome]),
                chi=chi,
            ).T.reshape(3)
            reflections.append(
                SeedReflection(
                    seed_index=seed_index,
                    hkl_id=int(pd_hkl_ids[seed_index]),
                    hkl=np.asarray(this_hkl, dtype=float),
                    tth=float(this_tth),
                    eta=seed_peak.eta,
                    ome=seed_peak.ome,
                    gvec_s=np.asarray(gvec_s, dtype=float),
                    intensity=seed_peak.intensity,
                    support=seed_peak.support,
                )
            )

    if use_friedel_pairing and total_raw_spots:
        logger.info(
            "\tFriedel pairing reduced seed spots from %d to %d",
            total_raw_spots,
            total_reduced_spots,
        )

    seed_plane_data = copy.deepcopy(pd)
    seed_exclusions = np.ones(seed_plane_data.getNhklRef(), dtype=bool)
    seed_exclusions[np.asarray(pd_hkl_ids, dtype=int)] = False
    seed_plane_data.exclusions = seed_exclusions

    params = dict(
        bMat=bMat,
        chi=chi,
        csym=csym,
        instr=instr,
        seed_plane_data=seed_plane_data,
        eta_ranges=eta_ranges,
        ome_ranges=np.radians(ome_ranges),
        ome_period=np.radians(ome_period),
        pair_eta_tol=pair_eta_tol,
        pair_ome_tol=pair_ome_tol,
        use_friedel_pairing=use_friedel_pairing,
        seed_tth_by_hkl={
            int(hkl_id): float(tth)
            for hkl_id, tth in zip(pd_hkl_ids, seed_tths)
        },
    )
    return reflections, seed_crystal_dirs, params


def generate_orientation_fibers(cfg, eta_ome):
    """
    From ome-eta maps and hklid spec, generate list of
    quaternions from fibers
    """
    ncpus = cfg.multiprocessing
    fiber_ndiv = cfg.find_orientations.seed_search.fiber_ndiv
    reflections, _seed_crystal_dirs, params = _collect_seed_reflections(cfg, eta_ome)
    params['fiber_ndiv'] = fiber_ndiv

    input_p = [
        np.hstack([reflection.hkl, reflection.gvec_s])
        for reflection in reflections
    ]
    if not input_p:
        logger.warning("\tno seed reflections were found for fiber generation")
        return np.empty((4, 0))

    # do the mapping
    start = timeit.default_timer()
    qfib = None
    nworkers = _pool_worker_count(len(input_p), ncpus)
    if nworkers > 1 and _spawn_pool_is_expensive():
        logger.info(
            "\tusing serial fiber generation on spawn multiprocessing context"
        )
        nworkers = 1
    if nworkers > 1:
        # multiple process version
        chunksize = _pool_chunksize(len(input_p), nworkers)
        with const.mp_context.Pool(
            nworkers,
            discretefiber_init,
            (params,),
        ) as pool:
            qfib = pool.map(discretefiber_reduced, input_p, chunksize=chunksize)
    else:
        # single process version.
        discretefiber_init(params)  # sets paramMP

        # We convert to a list to ensure the map is full iterated before
        # clean up. Otherwise discretefiber_reduced will be called
        # after cleanup.
        qfib = list(map(discretefiber_reduced, input_p))

        discretefiber_cleanup()
    elapsed = timeit.default_timer() - start
    logger.info("\tfiber generation took %.3f seconds", elapsed)
    return np.hstack(qfib)


def _rotation_from_vector_pair(c1, c2, s1, s2):
    c1 = mutil.unitVector(np.asarray(c1).reshape(3, 1)).reshape(3)
    c2 = mutil.unitVector(np.asarray(c2).reshape(3, 1)).reshape(3)
    s1 = mutil.unitVector(np.asarray(s1).reshape(3, 1)).reshape(3)
    s2 = mutil.unitVector(np.asarray(s2).reshape(3, 1)).reshape(3)

    c2_perp = c2 - np.dot(c1, c2) * c1
    s2_perp = s2 - np.dot(s1, s2) * s1

    c2_perp_norm = np.linalg.norm(c2_perp)
    s2_perp_norm = np.linalg.norm(s2_perp)
    if c2_perp_norm <= const.sqrt_epsf or s2_perp_norm <= const.sqrt_epsf:
        raise ValueError('vector pairs must not be parallel')

    c2_perp /= c2_perp_norm
    s2_perp /= s2_perp_norm

    c3 = np.cross(c1, c2_perp)
    s3 = np.cross(s1, s2_perp)

    crystal_basis = np.column_stack([c1, c2_perp, c3])
    sample_basis = np.column_stack([s1, s2_perp, s3])
    return np.dot(sample_basis, crystal_basis.T)


def _compress_pairwise_candidates(quats, qsym, pair_tol, max_candidates):
    if quats.size == 0:
        return np.empty((4, 0)), np.array([], dtype=int)

    exp_maps = rot.expMapOfQuat(quats)
    if exp_maps.ndim == 1:
        exp_maps = exp_maps.reshape(3, 1)

    bucket_scale = max(pair_tol, np.radians(0.25))
    buckets = np.rint((exp_maps / bucket_scale).T).astype(int)
    _uniq, inverse, counts = np.unique(
        buckets,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    order = np.argsort(counts)[::-1]
    if max_candidates is not None:
        order = order[:max_candidates]

    compressed = np.zeros((4, len(order)))
    for out_idx, bucket_id in enumerate(order):
        compressed[:, out_idx] = rot.quatAverageCluster(
            quats[:, inverse == bucket_id],
            qsym,
        ).flatten()

    return compressed, counts[order]


def _pairwise_quaternions_for_reflection_pair(
    reflection_i,
    reflection_j,
    seed_crystal_dirs,
    csym,
    pair_tol,
):
    min_pair_angle = max(pair_tol, np.radians(1.0))
    min_pair_sin = np.sin(min_pair_angle)

    s1 = reflection_i.gvec_s
    s2 = reflection_j.gvec_s
    if np.linalg.norm(np.cross(s1, s2)) <= min_pair_sin:
        return np.empty((4, 0))

    crystal_dirs_i = seed_crystal_dirs[reflection_i.seed_index]
    crystal_dirs_j = seed_crystal_dirs[reflection_j.seed_index]

    sample_dot = np.clip(np.dot(s1, s2), -1.0, 1.0)
    sample_angle = np.arccos(sample_dot)
    crystal_dot = np.clip(np.dot(crystal_dirs_i.T, crystal_dirs_j), -1.0, 1.0)
    crystal_angles = np.arccos(crystal_dot)

    raw_candidates = []
    match_i, match_j = np.where(np.abs(crystal_angles - sample_angle) <= pair_tol)
    for sym_i, sym_j in zip(match_i, match_j):
        c1 = crystal_dirs_i[:, sym_i]
        c2 = crystal_dirs_j[:, sym_j]
        if np.linalg.norm(np.cross(c1, c2)) <= min_pair_sin:
            continue

        try:
            rot_mat = _rotation_from_vector_pair(c1, c2, s1, s2)
        except ValueError:
            continue

        quat = rot.quatOfRotMat(rot_mat)
        quat = np.asarray(quat)
        if quat.ndim == 1:
            quat = quat.reshape(4, 1)
        raw_candidates.append(rot.toFundamentalRegion(quat, crysSym=csym))

    if not raw_candidates:
        return np.empty((4, 0))

    return np.hstack(raw_candidates)


def _candidate_quaternions_from_pairwise_intersections(
    reflections,
    seed_crystal_dirs,
    csym,
    qsym,
    pair_tol,
    max_candidates,
):
    if len(reflections) < 2:
        return np.empty((4, 0)), 0, np.array([], dtype=int)

    raw_candidates = []

    for i, reflection_i in enumerate(reflections[:-1]):
        for reflection_j in reflections[i + 1 :]:
            pair_candidates = _pairwise_quaternions_for_reflection_pair(
                reflection_i,
                reflection_j,
                seed_crystal_dirs,
                csym,
                pair_tol,
            )
            if pair_candidates.size:
                raw_candidates.append(pair_candidates)

    if not raw_candidates:
        return np.empty((4, 0)), 0, np.array([], dtype=int)

    quats = np.hstack(raw_candidates)
    compressed, counts = _compress_pairwise_candidates(
        quats,
        qsym,
        pair_tol,
        max_candidates,
    )
    return compressed, quats.shape[1], counts


def _consensus_proposal_sort_key(proposal):
    pair_hits, seed_support, hkl_support, support_weight, proximity_score, mean_distance, _quat = proposal
    return (
        int(pair_hits),
        int(seed_support),
        int(hkl_support),
        float(support_weight),
        float(proximity_score),
        -float(mean_distance),
    )


def _compress_consensus_proposals(proposals, qsym, pair_tol, max_candidates):
    if not proposals:
        return []

    quats = np.column_stack([proposal[-1] for proposal in proposals])
    exp_maps = rot.expMapOfQuat(quats)
    if exp_maps.ndim == 1:
        exp_maps = exp_maps.reshape(3, 1)

    bucket_scale = max(pair_tol, np.radians(0.25))
    buckets = np.rint((exp_maps / bucket_scale).T).astype(int)
    _uniq, inverse = np.unique(buckets, axis=0, return_inverse=True)

    compressed = []
    for bucket_id in range(np.max(inverse) + 1):
        indices = np.where(inverse == bucket_id)[0]
        bucket_quats = quats[:, indices]
        avg_quat = rot.quatAverageCluster(bucket_quats, qsym).flatten()

        bucket_props = [proposals[idx] for idx in indices]
        compressed.append(
            (
                int(sum(prop[0] for prop in bucket_props)),
                max(prop[1] for prop in bucket_props),
                max(prop[2] for prop in bucket_props),
                max(prop[3] for prop in bucket_props),
                max(prop[4] for prop in bucket_props),
                min(prop[5] for prop in bucket_props),
                avg_quat,
            )
        )

    compressed.sort(key=_consensus_proposal_sort_key, reverse=True)
    if max_candidates is not None:
        compressed = compressed[:max_candidates]

    return compressed


def _proposal_quaternions_and_metrics(proposals):
    if not proposals:
        return np.empty((4, 0)), {
            'pair_hits': np.array([], dtype=int),
            'seed_support': np.array([], dtype=int),
            'hkl_support': np.array([], dtype=int),
            'support_weight': np.array([], dtype=int),
            'proximity_score': np.array([], dtype=float),
        }

    quats = np.column_stack([proposal[-1] for proposal in proposals])
    metrics = {
        'pair_hits': np.asarray([proposal[0] for proposal in proposals], dtype=int),
        'seed_support': np.asarray([proposal[1] for proposal in proposals], dtype=int),
        'hkl_support': np.asarray([proposal[2] for proposal in proposals], dtype=int),
        'support_weight': np.asarray([proposal[3] for proposal in proposals], dtype=int),
        'proximity_score': np.asarray([proposal[4] for proposal in proposals], dtype=float),
    }
    return quats, metrics


def _clipped_percentile(values, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0

    percentile = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(values, percentile))


def reflection_statistics_init(params):
    global reflectionStatisticsMP
    reflectionStatisticsMP = params


def reflection_statistics_cleanup():
    global reflectionStatisticsMP
    reflectionStatisticsMP = None


def reflection_statistics_reduced(grain_range):
    params = reflectionStatisticsMP
    grain_param_list = params['grain_param_list'][slice(*grain_range)]
    seed_hkl_ids = params['seed_hkl_ids']
    seed_index_by_hkl = params['seed_index_by_hkl']

    sim_results = params['instr'].simulate_rotation_series(
        params['plane_data'],
        grain_param_list,
        eta_ranges=params['eta_ranges'],
        ome_ranges=params['ome_ranges'],
        ome_period=params['ome_period'],
    )

    ngrains = grain_param_list.shape[0]
    active_counts = np.zeros(ngrains, dtype=np.int64)
    seed_counts_raw = np.zeros(ngrains, dtype=np.int64)
    seed_counts_reduced = np.zeros(ngrains, dtype=np.int64)
    seed_hkl_counts = np.zeros(ngrains, dtype=np.int64)
    per_seed_raw = np.zeros((len(seed_hkl_ids), ngrains), dtype=np.int64)
    per_seed_reduced = np.zeros((len(seed_hkl_ids), ngrains), dtype=np.int64)
    seed_peaks_by_grain = [dict() for _ in range(ngrains)]

    for sim_result in sim_results.values():
        valid_ids_by_grain = sim_result[0]
        valid_angs_by_grain = sim_result[2]
        for igrain, (valid_ids, valid_angs) in enumerate(
            zip(valid_ids_by_grain, valid_angs_by_grain)
        ):
            valid_ids = np.asarray(valid_ids, dtype=int)
            if valid_ids.size == 0:
                continue

            valid_angs = np.asarray(valid_angs, dtype=float)
            active_counts[igrain] += int(
                np.count_nonzero(np.isin(valid_ids, params['active_hkl_ids']))
            )

            seed_mask = np.isin(valid_ids, seed_hkl_ids)
            if not np.any(seed_mask):
                continue

            seed_ids = valid_ids[seed_mask]
            seed_angs = valid_angs[seed_mask]
            seed_counts_raw[igrain] += int(seed_ids.size)
            for hkl_id, angs in zip(seed_ids, seed_angs):
                hkl_id = int(hkl_id)
                seed_idx = seed_index_by_hkl[hkl_id]
                per_seed_raw[seed_idx, igrain] += 1
                seed_peaks_by_grain[igrain].setdefault(hkl_id, []).append(
                    SeedPeak(
                        eta=float(angs[1]),
                        ome=float(
                            rot.mapAngle(
                                np.asarray([angs[2]], dtype=float),
                                [-np.pi, np.pi],
                            )[0]
                        ),
                        intensity=1.0,
                    )
                )

    use_friedel_pairing = params['use_friedel_pairing']
    for igrain, predicted_by_hkl in enumerate(seed_peaks_by_grain):
        seed_hkl_counts[igrain] = len(predicted_by_hkl)
        if not predicted_by_hkl:
            continue

        for hkl_id, peaks in predicted_by_hkl.items():
            reduced_peaks = peaks
            if use_friedel_pairing:
                reduced_peaks = _pair_friedel_seed_peaks(
                    peaks,
                    params['seed_tth_by_hkl'][hkl_id],
                    params['chi'],
                    params['pair_eta_tol'],
                    params['pair_ome_tol'],
                )

            seed_idx = seed_index_by_hkl[hkl_id]
            reduced_count = len(reduced_peaks)
            per_seed_reduced[seed_idx, igrain] = reduced_count
            seed_counts_reduced[igrain] += reduced_count

    if not use_friedel_pairing:
        per_seed_reduced = per_seed_raw.copy()
        seed_counts_reduced = seed_counts_raw.copy()

    return {
        'active_reflections_per_grain': active_counts,
        'seed_reflections_raw_per_grain': seed_counts_raw,
        'seed_reflections_reduced_per_grain': seed_counts_reduced,
        'seed_hkls_per_grain': seed_hkl_counts,
        'seed_reflections_raw_by_hkl': per_seed_raw,
        'seed_reflections_reduced_by_hkl': per_seed_reduced,
    }


def compute_reflection_statistics(cfg, eta_ome) -> ReflectionStatistics:
    plane_data = cfg.material.plane_data
    instr = cfg.instrument.hedm
    active_hkl_ids = np.asarray(eta_ome.iHKLList, dtype=int)
    seed_hkl_indices = np.asarray(
        cfg.find_orientations.seed_search.hkl_seeds,
        dtype=int,
    )
    seed_hkl_ids = np.asarray(active_hkl_ids[seed_hkl_indices], dtype=int)
    percentile = cfg.find_orientations.seed_search.reflection_statistics_percentile
    sample_count = max(
        int(cfg.find_orientations.seed_search.reflection_statistics_samples),
        1,
    )
    sample_seed = int(cfg.find_orientations.seed_search.reflection_statistics_seed)
    use_friedel_pairing = cfg.find_orientations.seed_search.friedel_pairing

    eta_ranges = np.radians(cfg.find_orientations.eta.range)
    ome_period, ome_ranges = _process_omegas(cfg.image_series)
    pair_eta_tol, pair_ome_tol = _seed_pairing_tolerances(cfg, eta_ome)

    pd_hkl_idx = plane_data.getHKLID(
        plane_data.getHKLs(*active_hkl_ids).T,
        master=False,
    )
    tth = plane_data.getTTh()
    seed_tths = tth[pd_hkl_idx][seed_hkl_indices]
    seed_tth_by_hkl = {
        int(hkl_id): float(tth_value)
        for hkl_id, tth_value in zip(seed_hkl_ids, seed_tths)
    }

    rng = np.random.default_rng(sample_seed)
    rand_q = mutil.unitVector(rng.normal(size=(4, sample_count)))
    rand_e = rot.expMapOfQuat(rand_q)
    if rand_e.ndim == 1:
        rand_e = rand_e.reshape(3, 1)
    grain_param_list = np.vstack(
        [
            rand_e,
            np.zeros((3, sample_count)),
            np.tile(const.identity_6x1, (sample_count, 1)).T,
        ]
    ).T

    params = {
        'plane_data': plane_data,
        'instr': instr,
        'grain_param_list': grain_param_list,
        'active_hkl_ids': active_hkl_ids,
        'seed_hkl_ids': seed_hkl_ids,
        'seed_index_by_hkl': {
            int(hkl_id): idx for idx, hkl_id in enumerate(seed_hkl_ids)
        },
        'eta_ranges': eta_ranges,
        'ome_ranges': np.radians(ome_ranges),
        'ome_period': np.radians(ome_period),
        'use_friedel_pairing': use_friedel_pairing,
        'seed_tth_by_hkl': seed_tth_by_hkl,
        'chi': cfg.instrument.hedm.chi,
        'pair_eta_tol': pair_eta_tol,
        'pair_ome_tol': pair_ome_tol,
    }

    nworkers = _pool_worker_count(sample_count, cfg.multiprocessing)
    if nworkers > 1 and _spawn_pool_is_expensive():
        logger.info(
            "\tusing serial reflection-statistics simulation on spawn multiprocessing context"
        )
        nworkers = 1

    grain_ranges = distribute_tasks(sample_count, nworkers)
    start = timeit.default_timer()
    if nworkers > 1:
        with const.mp_context.Pool(
            nworkers,
            reflection_statistics_init,
            (params,),
        ) as pool:
            results = pool.map(reflection_statistics_reduced, grain_ranges)
    else:
        reflection_statistics_init(params)
        try:
            results = list(map(reflection_statistics_reduced, grain_ranges))
        finally:
            reflection_statistics_cleanup()
    elapsed = timeit.default_timer() - start

    active_reflections_per_grain = np.concatenate(
        [result['active_reflections_per_grain'] for result in results]
    )
    seed_reflections_raw_per_grain = np.concatenate(
        [result['seed_reflections_raw_per_grain'] for result in results]
    )
    seed_reflections_reduced_per_grain = np.concatenate(
        [result['seed_reflections_reduced_per_grain'] for result in results]
    )
    seed_hkls_per_grain = np.concatenate(
        [result['seed_hkls_per_grain'] for result in results]
    )
    seed_reflections_raw_by_hkl = {
        int(hkl_id): np.concatenate(
            [result['seed_reflections_raw_by_hkl'][idx] for result in results]
        )
        for idx, hkl_id in enumerate(seed_hkl_ids)
    }
    seed_reflections_reduced_by_hkl = {
        int(hkl_id): np.concatenate(
            [result['seed_reflections_reduced_by_hkl'][idx] for result in results]
        )
        for idx, hkl_id in enumerate(seed_hkl_ids)
    }

    stats = ReflectionStatistics(
        sample_count=sample_count,
        active_reflections_per_grain=active_reflections_per_grain,
        seed_reflections_raw_per_grain=seed_reflections_raw_per_grain,
        seed_reflections_reduced_per_grain=seed_reflections_reduced_per_grain,
        seed_hkls_per_grain=seed_hkls_per_grain,
        seed_reflections_raw_by_hkl=seed_reflections_raw_by_hkl,
        seed_reflections_reduced_by_hkl=seed_reflections_reduced_by_hkl,
    )

    logger.info(
        "\treflection statistics (%d samples, p%.1f) took %.3f seconds",
        sample_count,
        float(np.clip(percentile, 0.0, 100.0)),
        elapsed,
    )
    logger.info(
        "\tseed reflections/grain raw p%.1f=%.1f mean=%.1f; reduced p%.1f=%.1f mean=%.1f; active reflections mean=%.1f",
        float(np.clip(percentile, 0.0, 100.0)),
        _clipped_percentile(stats.seed_reflections_raw_per_grain, percentile),
        float(np.mean(stats.seed_reflections_raw_per_grain)),
        float(np.clip(percentile, 0.0, 100.0)),
        _clipped_percentile(stats.seed_reflections_reduced_per_grain, percentile),
        float(np.mean(stats.seed_reflections_reduced_per_grain)),
        float(np.mean(stats.active_reflections_per_grain)),
    )
    logger.info(
        "\tvisible seed hkl families/grain p%.1f=%.1f mean=%.1f",
        float(np.clip(percentile, 0.0, 100.0)),
        _clipped_percentile(stats.seed_hkls_per_grain, percentile),
        float(np.mean(stats.seed_hkls_per_grain)),
    )

    return stats


def _pairwise_consensus_support_thresholds(
    cfg,
    stats: ReflectionStatistics,
    reflections,
):
    percentile = cfg.find_orientations.seed_search.reflection_statistics_percentile
    compl_thresh = cfg.find_orientations.clustering.completeness
    use_friedel_pairing = cfg.find_orientations.seed_search.friedel_pairing

    unique_seed_count = len({reflection.seed_index for reflection in reflections})
    unique_hkl_count = len({reflection.hkl_id for reflection in reflections})

    seed_support_floor = _clipped_percentile(
        stats.seed_reflections_per_grain(use_friedel_pairing),
        percentile,
    )
    hkl_support_floor = _clipped_percentile(
        stats.seed_hkls_per_grain,
        percentile,
    )

    min_seed_support = min(
        unique_seed_count,
        max(2, int(np.ceil(compl_thresh * seed_support_floor))),
    )
    min_hkl_support = min(
        unique_hkl_count,
        max(1, int(np.ceil(compl_thresh * hkl_support_floor))),
    )
    if unique_hkl_count > 1:
        min_hkl_support = max(2, min_hkl_support)

    return int(min_seed_support), int(min_hkl_support)


def _reflection_support_metrics(
    reflections,
    quat,
    qsym,
    bmat,
    claim_tol,
    active_mask=None,
):
    quat = np.asarray(quat)
    if quat.ndim == 1:
        quat = quat.reshape(4, 1)

    if active_mask is None:
        active_indices = range(len(reflections))
    else:
        active_indices = np.where(active_mask)[0]

    support_indices = []
    support_weight = 0
    proximity_score = 0.0
    distance_sum = 0.0
    seed_ids = set()
    hkl_ids = set()

    for idx in active_indices:
        reflection = reflections[idx]
        distance = rot.distanceToFiber(
            reflection.hkl.reshape(3, 1),
            reflection.gvec_s,
            quat,
            qsym,
            centrosymmetry=True,
            bmatrix=bmat,
        )
        distance = float(np.ravel(distance)[0])
        if distance <= claim_tol:
            support_indices.append(idx)
            support_weight += reflections[idx].support
            seed_ids.add(reflections[idx].seed_index)
            hkl_ids.add(reflections[idx].hkl_id)
            if claim_tol > const.sqrt_epsf:
                proximity_score += reflections[idx].support * max(
                    0.0,
                    1.0 - distance / claim_tol,
                )
            distance_sum += distance

    mean_distance = (
        float(distance_sum / len(support_indices))
        if support_indices
        else float('inf')
    )
    return (
        np.asarray(support_indices, dtype=int),
        int(support_weight),
        len(seed_ids),
        len(hkl_ids),
        float(proximity_score),
        mean_distance,
    )


def _reflection_support_mask(
    reflections,
    active_mask,
    quat,
    qsym,
    bmat,
    claim_tol,
):
    support_mask = np.zeros(len(reflections), dtype=bool)
    support_indices, *_metrics = _reflection_support_metrics(
        reflections,
        quat,
        qsym,
        bmat,
        claim_tol,
        active_mask=active_mask,
    )
    support_mask[support_indices] = True

    return support_mask


def pairwise_consensus_init(params):
    global pairwiseConsensusMP
    pairwiseConsensusMP = params


def pairwise_consensus_cleanup():
    global pairwiseConsensusMP
    pairwiseConsensusMP = None


def pairwise_consensus_reduced(anchor_range):
    params = pairwiseConsensusMP
    reflections = params['reflections']
    order = params['order']
    order_positions = params['order_positions']
    seed_crystal_dirs = params['seed_crystal_dirs']
    csym = params['csym']
    qsym = params['qsym']
    bmat = params['bMat']
    pair_tol = params['pair_tol']
    claim_tol = params['claim_tol']
    max_partners = params['max_partners']
    local_keep = params['local_keep']
    min_seed_support = params['min_seed_support']
    min_hkl_support = params['min_hkl_support']

    proposals = []
    pair_tests = 0
    start, stop = anchor_range
    for anchor_pos in range(start, stop):
        anchor_idx = order[anchor_pos]
        anchor = reflections[anchor_idx]
        remaining = order[anchor_pos + 1 :]
        if not remaining:
            continue

        distinct_seed_partners = [
            idx
            for idx in remaining
            if reflections[idx].seed_index != anchor.seed_index
        ]
        same_seed_partners = [
            idx
            for idx in remaining
            if reflections[idx].seed_index == anchor.seed_index
        ]
        partner_order = distinct_seed_partners + same_seed_partners
        if max_partners > 0:
            partner_order = partner_order[:max_partners]

        anchor_proposals = []
        for partner_idx in partner_order:
            if order_positions[partner_idx] <= anchor_pos:
                continue

            partner = reflections[partner_idx]
            pair_tests += 1
            pair_candidates = _pairwise_quaternions_for_reflection_pair(
                anchor,
                partner,
                seed_crystal_dirs,
                csym,
                pair_tol,
            )
            if pair_candidates.size == 0:
                continue

            pair_candidates, _ = _compress_pairwise_candidates(
                pair_candidates,
                qsym,
                pair_tol,
                None,
            )

            for cand_idx in range(pair_candidates.shape[1]):
                quat = pair_candidates[:, cand_idx]
                (
                    _support_indices,
                    support_weight,
                    seed_support,
                    hkl_support,
                    proximity_score,
                    mean_distance,
                ) = _reflection_support_metrics(
                    reflections,
                    quat,
                    qsym,
                    bmat,
                    claim_tol,
                )
                if seed_support < min_seed_support or hkl_support < min_hkl_support:
                    continue

                anchor_proposals.append(
                    (
                        1,
                        seed_support,
                        hkl_support,
                        support_weight,
                        proximity_score,
                        mean_distance,
                        quat.copy(),
                    )
                )

        if anchor_proposals:
            anchor_proposals = _compress_consensus_proposals(
                anchor_proposals,
                qsym,
                pair_tol,
                local_keep,
            )
            proposals.extend(anchor_proposals)

    return proposals, pair_tests


def _candidate_quaternions_from_pairwise_consensus(
    reflections,
    seed_crystal_dirs,
    csym,
    qsym,
    bmat,
    pair_tol,
    max_candidates,
    max_partners,
    min_seed_support,
    min_hkl_support,
    ncpus=1,
):
    if len(reflections) < 2:
        return np.empty((4, 0)), 0, 0, {
            'pair_hits': np.array([], dtype=int),
            'seed_support': np.array([], dtype=int),
            'hkl_support': np.array([], dtype=int),
            'support_weight': np.array([], dtype=int),
            'proximity_score': np.array([], dtype=float),
        }

    order = sorted(
        range(len(reflections)),
        key=lambda idx: (
            -reflections[idx].support,
            -reflections[idx].intensity,
            reflections[idx].seed_index,
            reflections[idx].hkl_id,
        ),
    )
    claim_tol = pair_tol
    local_keep = min(max(4, max_candidates // max(len(order), 1) + 1), 8)

    params = {
        'reflections': reflections,
        'order': order,
        'order_positions': {idx: pos for pos, idx in enumerate(order)},
        'seed_crystal_dirs': seed_crystal_dirs,
        'csym': csym,
        'qsym': qsym,
        'bMat': bmat,
        'pair_tol': pair_tol,
        'claim_tol': claim_tol,
        'max_partners': int(max_partners),
        'local_keep': int(local_keep),
        'min_seed_support': int(min_seed_support),
        'min_hkl_support': int(min_hkl_support),
    }

    anchor_count = max(len(order) - 1, 0)
    if anchor_count == 0:
        return np.empty((4, 0)), 0, 0, {
            'pair_hits': np.array([], dtype=int),
            'seed_support': np.array([], dtype=int),
            'hkl_support': np.array([], dtype=int),
            'support_weight': np.array([], dtype=int),
            'proximity_score': np.array([], dtype=float),
        }

    nworkers = _pool_worker_count(anchor_count, ncpus)
    if nworkers > 1 and _spawn_pool_is_expensive():
        logger.info(
            "\tusing serial pairwise consensus generation on spawn multiprocessing context"
        )
        nworkers = 1

    anchor_ranges = distribute_tasks(anchor_count, nworkers)
    if nworkers > 1:
        with const.mp_context.Pool(
            nworkers,
            pairwise_consensus_init,
            (params,),
        ) as pool:
            results = pool.map(pairwise_consensus_reduced, anchor_ranges)
    else:
        pairwise_consensus_init(params)
        try:
            results = list(map(pairwise_consensus_reduced, anchor_ranges))
        finally:
            pairwise_consensus_cleanup()

    proposals = []
    pair_tests = 0
    for worker_proposals, worker_pair_tests in results:
        proposals.extend(worker_proposals)
        pair_tests += worker_pair_tests

    compressed = _compress_consensus_proposals(
        proposals,
        qsym,
        pair_tol,
        max_candidates,
    )
    quats, metrics = _proposal_quaternions_and_metrics(compressed)
    return quats, pair_tests, len(proposals), metrics


def _simulate_seed_peak_map(
    quat,
    params,
):
    quat = np.asarray(quat)
    if quat.ndim == 1:
        quat = quat.reshape(4, 1)

    grain_params = np.hstack(
        [
            rot.expMapOfQuat(quat).reshape(3),
            np.zeros(3),
            const.identity_6x1.flatten(),
        ]
    )
    sim_results = params['instr'].simulate_rotation_series(
        params['seed_plane_data'],
        [grain_params],
        eta_ranges=params['eta_ranges'],
        ome_ranges=params['ome_ranges'],
        ome_period=params['ome_period'],
    )

    predicted_by_hkl = {}
    for sim_result in sim_results.values():
        valid_ids = np.asarray(sim_result[0][0], dtype=int)
        valid_angs = np.asarray(sim_result[2][0], dtype=float)
        if valid_ids.size == 0:
            continue

        for hkl_id, angs in zip(valid_ids, valid_angs):
            predicted_by_hkl.setdefault(int(hkl_id), []).append(
                SeedPeak(
                    eta=float(angs[1]),
                    ome=float(rot.mapAngle(np.asarray([angs[2]]), [-np.pi, np.pi])[0]),
                    intensity=1.0,
                )
            )

    if params['use_friedel_pairing']:
        for hkl_id, peaks in list(predicted_by_hkl.items()):
            predicted_by_hkl[hkl_id] = _pair_friedel_seed_peaks(
                peaks,
                params['seed_tth_by_hkl'][int(hkl_id)],
                params['chi'],
                params['pair_eta_tol'],
                params['pair_ome_tol'],
            )

    return predicted_by_hkl


def _match_predicted_seed_peaks(
    reflections,
    active_mask,
    predicted_by_hkl,
    observed_by_hkl,
    eta_tol,
    ome_tol,
):
    support_mask = np.zeros(len(reflections), dtype=bool)
    predicted_total = 0
    matched_total = 0
    matched_support = 0
    matched_seed_ids = set()

    for hkl_id, pred_peaks in predicted_by_hkl.items():
        if not pred_peaks:
            continue

        predicted_total += len(pred_peaks)
        obs_indices = [
            idx for idx in observed_by_hkl.get(int(hkl_id), []) if active_mask[idx]
        ]
        if not obs_indices:
            continue

        pred_eta = np.asarray([peak.eta for peak in pred_peaks], dtype=float)
        pred_ome = np.asarray([peak.ome for peak in pred_peaks], dtype=float)
        obs_eta = np.asarray([reflections[idx].eta for idx in obs_indices], dtype=float)
        obs_ome = np.asarray([reflections[idx].ome for idx in obs_indices], dtype=float)

        eta_diff = np.abs(
            np.arctan2(
                np.sin(pred_eta[:, None] - obs_eta[None, :]),
                np.cos(pred_eta[:, None] - obs_eta[None, :]),
            )
        )
        ome_diff = np.abs(
            np.arctan2(
                np.sin(pred_ome[:, None] - obs_ome[None, :]),
                np.cos(pred_ome[:, None] - obs_ome[None, :]),
            )
        )
        cost = (eta_diff / eta_tol) ** 2 + (ome_diff / ome_tol) ** 2

        max_cost = 2.0
        padded_cost = np.array(cost, copy=True)
        padded_cost[padded_cost > max_cost] = max_cost + 1.0
        row_ind, col_ind = linear_sum_assignment(padded_cost)

        for row_idx, col_idx in zip(row_ind, col_ind):
            if cost[row_idx, col_idx] > max_cost:
                continue

            obs_idx = obs_indices[col_idx]
            support_mask[obs_idx] = True
            matched_total += 1
            matched_support += reflections[obs_idx].support
            matched_seed_ids.add(reflections[obs_idx].seed_index)

    return support_mask, predicted_total, matched_total, matched_support, len(
        matched_seed_ids
    )


def _simulate_seed_peak_matches(
    reflections,
    active_mask,
    quat,
    observed_by_hkl,
    params,
):
    predicted_by_hkl = _simulate_seed_peak_map(quat, params)
    return _match_predicted_seed_peaks(
        reflections,
        active_mask,
        predicted_by_hkl,
        observed_by_hkl,
        params['pair_eta_tol'],
        params['pair_ome_tol'],
    )


def _score_quaternion_completeness(cfg, eta_ome, quat):
    quat = np.asarray(quat)
    if quat.ndim == 1:
        quat = quat.reshape(4, 1)

    return float(
        indexer.paintGrid(
            quat,
            eta_ome,
            etaRange=np.radians(cfg.find_orientations.eta.range),
            omeTol=np.radians(cfg.find_orientations.omega.tolerance),
            etaTol=np.radians(cfg.find_orientations.eta.tolerance),
            omePeriod=np.radians(cfg.find_orientations.omega.period),
            threshold=cfg.find_orientations.threshold,
            doMultiProc=False,
            nCPUs=1,
        )[0]
    )


def generate_orientation_candidates_pairwise(cfg, eta_ome):
    pair_tol = np.radians(cfg.find_orientations.seed_search.pairwise_tolerance)
    max_candidates = cfg.find_orientations.seed_search.pairwise_max_candidates

    reflections, seed_crystal_dirs, params = _collect_seed_reflections(cfg, eta_ome)

    start = timeit.default_timer()
    qbar, raw_candidate_count, counts = _candidate_quaternions_from_pairwise_intersections(
        reflections,
        seed_crystal_dirs,
        params['csym'],
        eta_ome.planeData.q_sym,
        pair_tol,
        max_candidates,
    )
    elapsed = timeit.default_timer() - start

    logger.info("\tpairwise candidate generation took %.3f seconds", elapsed)
    logger.info(
        "\tgenerated %d raw pairwise candidates and retained %d",
        raw_candidate_count,
        qbar.shape[1],
    )
    if counts.size:
        logger.info(
            "\tstrongest pairwise support count: %d",
            int(np.max(counts)),
        )

    return qbar


def generate_orientation_candidates_pairwise_consensus(cfg, eta_ome):
    pair_tol = np.radians(cfg.find_orientations.seed_search.pairwise_tolerance)
    max_candidates = cfg.find_orientations.seed_search.pairwise_max_candidates
    max_partners = cfg.find_orientations.seed_search.pairwise_max_partners

    reflections, seed_crystal_dirs, params = _collect_seed_reflections(cfg, eta_ome)
    stats = compute_reflection_statistics(cfg, eta_ome)
    min_seed_support, min_hkl_support = _pairwise_consensus_support_thresholds(
        cfg,
        stats,
        reflections,
    )

    start = timeit.default_timer()
    qbar, pair_tests, raw_proposals, metrics = (
        _candidate_quaternions_from_pairwise_consensus(
            reflections,
            seed_crystal_dirs,
            params['csym'],
            eta_ome.planeData.q_sym,
            params['bMat'],
            pair_tol,
            max_candidates,
            max_partners,
            min_seed_support,
            min_hkl_support,
            ncpus=cfg.multiprocessing,
        )
    )
    elapsed = timeit.default_timer() - start

    logger.info("\tpairwise consensus candidate generation took %.3f seconds", elapsed)
    logger.info(
        "\tpairwise consensus tested %d reflection pairs and retained %d candidates from %d ranked proposals",
        pair_tests,
        qbar.shape[1],
        raw_proposals,
    )
    if metrics['pair_hits'].size:
        logger.info(
            "\tstrongest pairwise consensus: %d pair hits, %d seeds, %d hkls, weight %d",
            int(np.max(metrics['pair_hits'])),
            int(np.max(metrics['seed_support'])),
            int(np.max(metrics['hkl_support'])),
            int(np.max(metrics['support_weight'])),
        )
    logger.info(
        "\tpairwise consensus thresholds: %d seed reflections across %d hkl families",
        min_seed_support,
        min_hkl_support,
    )

    return qbar


def generate_orientation_candidates_pairwise_greedy(cfg, eta_ome):
    pair_tol = np.radians(cfg.find_orientations.seed_search.pairwise_tolerance)
    claim_tol = pair_tol
    max_candidates = cfg.find_orientations.seed_search.pairwise_max_candidates
    compl_thresh = cfg.find_orientations.clustering.completeness

    reflections, seed_crystal_dirs, params = _collect_seed_reflections(cfg, eta_ome)
    if len(reflections) < 2:
        return np.empty((4, 0))

    qsym = eta_ome.planeData.q_sym
    active_mask = np.ones(len(reflections), dtype=bool)
    unique_seed_count = len({reflection.seed_index for reflection in reflections})
    min_total_support = min(3, sum(reflection.support for reflection in reflections))
    min_seed_support = min(2, unique_seed_count)

    order = sorted(
        range(len(reflections)),
        key=lambda idx: (-reflections[idx].support, -reflections[idx].intensity),
    )

    accepted = []
    pair_tests = 0
    scored_candidates = 0

    start = timeit.default_timer()
    for anchor_idx in order:
        if not active_mask[anchor_idx]:
            continue

        anchor = reflections[anchor_idx]
        partner_order = sorted(
            [
                idx
                for idx in order
                if idx != anchor_idx and active_mask[idx]
            ],
            key=lambda idx: (
                reflections[idx].seed_index == anchor.seed_index,
                -reflections[idx].support,
                -reflections[idx].intensity,
            ),
        )

        accepted_anchor = False
        for partner_idx in partner_order:
            partner = reflections[partner_idx]
            pair_tests += 1
            pair_candidates = _pairwise_quaternions_for_reflection_pair(
                anchor,
                partner,
                seed_crystal_dirs,
                params['csym'],
                pair_tol,
            )
            if pair_candidates.size == 0:
                continue

            pair_candidates, _ = _compress_pairwise_candidates(
                pair_candidates,
                qsym,
                pair_tol,
                None,
            )

            for cand_idx in range(pair_candidates.shape[1]):
                quat = pair_candidates[:, cand_idx]
                support_mask = _reflection_support_mask(
                    reflections,
                    active_mask,
                    quat,
                    qsym,
                    params['bMat'],
                    claim_tol,
                )
                support_weight = int(
                    sum(
                        reflections[idx].support
                        for idx in np.where(support_mask)[0]
                    )
                )
                seed_support = len(
                    {
                        reflections[idx].seed_index
                        for idx in np.where(support_mask)[0]
                    }
                )
                if support_weight < min_total_support or seed_support < min_seed_support:
                    continue

                scored_candidates += 1
                completeness = _score_quaternion_completeness(cfg, eta_ome, quat)
                if completeness < compl_thresh:
                    continue

                accepted.append(quat.reshape(4, 1))
                active_mask[support_mask] = False
                accepted_anchor = True
                break

            if accepted_anchor or len(accepted) >= max_candidates:
                break

        if len(accepted) >= max_candidates:
            break

    elapsed = timeit.default_timer() - start
    logger.info("\tgreedy pairwise candidate generation took %.3f seconds", elapsed)
    logger.info(
        "\tgreedy pairwise tested %d reflection pairs and scored %d candidates",
        pair_tests,
        scored_candidates,
    )
    logger.info(
        "\tgreedy pairwise retained %d candidates and left %d active reflections",
        len(accepted),
        int(np.count_nonzero(active_mask)),
    )

    if not accepted:
        return np.empty((4, 0))

    return np.hstack(accepted)


def generate_orientation_candidates(cfg, eta_ome):
    generator = cfg.find_orientations.seed_search.candidate_generator
    if generator == 'pairwise':
        return generate_orientation_candidates_pairwise(cfg, eta_ome)
    if generator == 'pairwise-consensus':
        return generate_orientation_candidates_pairwise_consensus(cfg, eta_ome)
    if generator == 'pairwise-greedy':
        return generate_orientation_candidates_pairwise_greedy(cfg, eta_ome)

    return generate_orientation_fibers(cfg, eta_ome)


def discretefiber_init(params):
    global paramMP
    paramMP = params


def discretefiber_cleanup():
    global paramMP
    del paramMP


def discretefiber_reduced(params_in):
    """
    input parameters are [hkl, gvec_s]
    """
    global paramMP
    bMat = paramMP['bMat']
    csym = paramMP['csym']
    fiber_ndiv = paramMP['fiber_ndiv']

    hkl = params_in[:3].reshape(3, 1)
    gVec_s = params_in[3:].reshape(3, 1)

    tmp = mutil.uniqueVectors(
        rot.discreteFiber(
            hkl, gVec_s, B=bMat, ndiv=fiber_ndiv, invert=False, csym=csym
        )[0]
    )
    return tmp


def run_cluster(
    compl, qfib, qsym, cfg, min_samples=None, compl_thresh=None, radius=None
):
    """ """
    algorithm = cfg.find_orientations.clustering.algorithm

    cl_radius = cfg.find_orientations.clustering.radius
    min_compl = cfg.find_orientations.clustering.completeness

    ncpus = cfg.multiprocessing

    # check for override on completeness threshold
    if compl_thresh is not None:
        min_compl = compl_thresh

    # check for override on radius
    if radius is not None:
        cl_radius = radius

    start = timeit.default_timer()  # timeit this

    num_above = sum(np.array(compl) > min_compl)
    if num_above == 0:
        # nothing to cluster
        qbar = cl = np.array([])
    elif num_above == 1:
        # short circuit
        qbar = qfib[:, np.array(compl) > min_compl]
        cl = [1]
    else:
        # use compiled module for distance
        # just to be safe, must order qsym as C-contiguous
        qsym = np.array(qsym.T, order='C').T

        def quat_distance(x, y):
            return xfcapi.quat_distance(
                np.array(x, order='C'), np.array(y, order='C'), qsym
            )

        qfib_r = qfib[:, np.array(compl) > min_compl]

        num_ors = qfib_r.shape[1]

        if num_ors > 25000:
            if algorithm == 'sph-dbscan' or algorithm == 'fclusterdata':
                logger.info("falling back to euclidean DBSCAN")
                algorithm = 'ort-dbscan'
            # raise RuntimeError(
            #     "Requested clustering of %d orientations, "
            #     + "which would be too slow!" % qfib_r.shape[1]
            # )

        logger.info(
            "Feeding %d orientations above %.1f%% to clustering",
            num_ors,
            100 * min_compl,
        )

        if algorithm == 'dbscan' and not have_sklearn:
            algorithm = 'fclusterdata'
            logger.warning("sklearn >= 0.14 required for dbscan; using fclusterdata")

        if algorithm in ['dbscan', 'ort-dbscan', 'sph-dbscan']:
            # munge min_samples according to options
            if (
                min_samples is None
                or cfg.find_orientations.use_quaternion_grid is not None
            ):
                min_samples = 1

            if algorithm == 'sph-dbscan':
                logger.info("using spherical DBSCAN")
                # compute distance matrix
                pdist = pairwise_distances(qfib_r.T, metric=quat_distance, n_jobs=1)

                # run dbscan
                core_samples, labels = dbscan(
                    pdist,
                    eps=np.radians(cl_radius),
                    min_samples=min_samples,
                    metric='precomputed',
                    n_jobs=ncpus,
                )
            else:
                if algorithm == 'ort-dbscan':
                    logger.debug("using euclidean orthographic DBSCAN")
                    pts = qfib_r[1:, :].T
                    eps = 0.25 * np.radians(cl_radius)
                else:
                    logger.debug("using euclidean DBSCAN")
                    pts = qfib_r.T
                    eps = 0.5 * np.radians(cl_radius)

                # run dbscan
                core_samples, labels = dbscan(
                    pts,
                    eps=eps,
                    min_samples=min_samples,
                    metric='minkowski',
                    p=2,
                    n_jobs=ncpus,
                )

            # extract cluster labels
            cl = np.array(labels, dtype=int)  # convert to array
            noise_points = cl == -1  # index for marking noise
            cl += 1  # move index to 1-based instead of 0
            cl[noise_points] = -1  # re-mark noise as -1
            logger.info("dbscan found %d noise points", sum(noise_points))
        elif algorithm == 'fclusterdata':
            logger.info("using spherical fclusetrdata")
            cl = cluster.hierarchy.fclusterdata(
                qfib_r.T,
                np.radians(cl_radius),
                criterion='distance',
                metric=quat_distance,
            )
        else:
            raise RuntimeError("Clustering algorithm %s not recognized" % algorithm)

        # extract number of clusters
        if np.any(cl == -1):
            nblobs = len(np.unique(cl)) - 1
        else:
            nblobs = len(np.unique(cl))

        """ PERFORM AVERAGING TO GET CLUSTER CENTROIDS """
        qbar = np.zeros((4, nblobs))
        for i in range(nblobs):
            npts = sum(cl == i + 1)
            qbar[:, i] = rot.quatAverageCluster(qfib_r[:, cl == i + 1], qsym).flatten()

    if algorithm in ('dbscan', 'ort-dbscan') and qbar.size / 4 > 1:
        logger.info("\tchecking for duplicate orientations...")
        cl = cluster.hierarchy.fclusterdata(
            qbar.T,
            np.radians(cl_radius),
            criterion='distance',
            metric=quat_distance,
        )
        nblobs_new = len(np.unique(cl))
        if nblobs_new < nblobs:
            logger.info(
                "\tfound %d duplicates within %f degrees",
                nblobs - nblobs_new,
                cl_radius,
            )
            tmp = np.zeros((4, nblobs_new))
            for i in range(nblobs_new):
                npts = sum(cl == i + 1)
                tmp[:, i] = rot.quatAverageCluster(
                    qbar[:, cl == i + 1].reshape(4, npts), qsym
                ).flatten()
            qbar = tmp

    logger.info("clustering took %f seconds", timeit.default_timer() - start)
    logger.info(
        "Found %d orientation clusters with >=%.1f%% completeness"
        " and %2f misorientation",
        qbar.size / 4,
        100.0 * min_compl,
        cl_radius,
    )

    return np.atleast_2d(qbar), cl


def merge_orientations_by_misorientation(
    compl: NDArray[np.float64],
    qfib: NDArray[np.float64],
    qsym: NDArray[np.float64],
    compl_thresh: float,
    radius: float,
):
    valid_idx = np.where(np.asarray(compl, dtype=float) >= float(compl_thresh))[0]
    if valid_idx.size == 0:
        return np.empty((4, 0)), np.full(qfib.shape[1], -1, dtype=int)

    merge_radius = np.radians(radius)
    parent = np.arange(valid_idx.size, dtype=int)

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(idx_0: int, idx_1: int) -> None:
        root_0 = find(idx_0)
        root_1 = find(idx_1)
        if root_0 != root_1:
            parent[root_1] = root_0

    for i in range(valid_idx.size):
        quat_i = qfib[:, valid_idx[i]]
        for j in range(i + 1, valid_idx.size):
            quat_j = qfib[:, valid_idx[j]]
            if xfcapi.quat_distance(quat_i, quat_j, qsym) <= merge_radius:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for local_idx, global_idx in enumerate(valid_idx):
        root = find(local_idx)
        groups.setdefault(root, []).append(int(global_idx))

    ordered_groups = sorted(
        groups.values(),
        key=lambda members: np.max(compl[members]),
        reverse=True,
    )

    qbar = np.zeros((4, len(ordered_groups)))
    labels = np.full(qfib.shape[1], -1, dtype=int)
    for out_idx, members in enumerate(ordered_groups, start=1):
        member_quats = qfib[:, members]
        if len(members) == 1:
            qbar[:, out_idx - 1] = member_quats[:, 0]
        else:
            qbar[:, out_idx - 1] = rot.quatAverageCluster(
                member_quats,
                qsym,
            ).flatten()
        labels[members] = out_idx

    logger.info(
        "merged %d scored orientations into %d grains using %.3f deg misorientation",
        valid_idx.size,
        qbar.shape[1],
        radius,
    )
    return qbar, labels


# TODO: Remove image_series from this function signature.
# TODO: Remove pd from this function signature.
def load_eta_ome_maps(
    cfg: root.RootConfig,
    pd,
    image_series,
    hkls: Optional[NDArray[np.float64] | list[int]] = None,
    clean: bool = False,
):
    """
    Load the eta-ome maps specified by the config and CLI flags. If the
    maps file exists, it will return those values. If the file does not exist,
    it will generate them using the passed HKLs (if not None) or the HKLs
    specified in the config file (if passed HKLs are Noe).

    Parameters
    ----------
    cfg: Config instance
        root config file for this problem
    pd: Not used
    image_series: Not used
    hkls: list, default = None
        list of HKLs used to generate the eta-omega maps
    clean: bool, default = False
        flag indicating whether (if True) to overwrite existing maps file

    Returns
    -------
    list:
        list of eta-omega map arrays

    """
    filename = cfg.find_orientations.orientation_maps.file
    if clean:
        logger.info('clean option specified; recomputing eta/ome orientation maps')
        res = generate_eta_ome_maps(cfg, hkls=hkls)
    else:
        try:
            res = EtaOmeMaps(str(filename))
            pd = res.planeData
            logger.info(f'loaded eta/ome orientation maps from {filename}')
            shkls = pd.getHKLs(*res.iHKLList, asStr=True)
            logger.info(
                'hkls used to generate orientation maps: %s',
                [f'[{i}]' for i in shkls],
            )
        except (AttributeError, IOError):
            logger.warning(
                f"specified maps file '{filename}' not found "
                f"and clean option not specified; "
                f"recomputing eta/ome orientation maps"
            )
            res = generate_eta_ome_maps(cfg, hkls=hkls)

        filter_maps_if_requested(res, cfg)
    return res


def filter_maps_if_requested(eta_ome, cfg: root.RootConfig):
    # !!! current logic:
    #  if False/None don't do anything
    #  if True, only do median subtraction
    #  if scalar, do median + LoG filter with that many pixels std dev
    filter_maps = cfg.find_orientations.orientation_maps.filter_maps
    if filter_maps:
        if not isinstance(filter_maps, bool):
            sigm = const.fwhm_to_sigma * filter_maps
            logger.info("filtering eta/ome maps incl LoG with %.2f std dev", sigm)
            _filter_eta_ome_maps(eta_ome, filter_stdev=sigm)
        else:
            logger.info("filtering eta/ome maps")
            _filter_eta_ome_maps(eta_ome)


def generate_eta_ome_maps(
    cfg: root.RootConfig,
    hkls: Optional[NDArray[np.float64] | list[int]] = None,
    save: bool = True,
):
    """
    Generates the eta-omega maps specified in the input config.

    Parameters
    ----------
    cfg : hexrd.core.config.root.RootConfig
        A hexrd far-field HEDM config instance.
    hkls : array_like, optional
        If not None, an override for the hkls used to generate maps. This can
        be either a list of unique hklIDs, or a list of [h, k, l] vectors.
        The default is None.
    save : bool, optional
        If True, write map archive to npz format according to path spec in cfg.
        The default is True.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    eta_ome : TYPE
        DESCRIPTION.

    """
    # extract PlaneData from config and set active hkls
    plane_data = cfg.material.plane_data

    # all active hkl ids masked by exclusions
    active_hklIDs = plane_data.getHKLID(plane_data.hkls, master=True)

    # need the below
    use_all = False

    # handle optional override
    if hkls:
        # overriding hkls are specified
        hkls = np.asarray(hkls)

        # if input is 2-d list of hkls, convert to hklIDs
        if hkls.ndim == 2:
            hkls = plane_data.getHKLID(hkls.tolist(), master=True)
    else:
        # handle logic for active hkl spec in config
        # !!!: default to all hkls defined for material,
        #      override with hkls from config, if specified;
        temp = np.asarray(cfg.find_orientations.orientation_maps.active_hkls)
        if temp.ndim == 0:
            # !!! this is only possible if active_hkls is None
            use_all = True
        elif temp.ndim == 1:
            # we have hklIDs
            hkls = temp
        elif temp.ndim == 2:
            # we have actual hkls
            hkls = plane_data.getHKLID(temp.tolist(), master=True)
        else:
            raise RuntimeError(
                'active_hkls spec must be 1-d or 2-d, not %d-d' % temp.ndim
            )

    # apply some checks to active_hkls specificaton
    if not use_all:
        # !!! hkls --> list of hklIDs now
        # catch duplicates
        assert len(np.unique(hkls)) == len(hkls), "duplicate hkls specified!"

        # catch excluded hkls
        excluded = np.zeros_like(hkls, dtype=bool)
        for i, hkl in enumerate(hkls):
            if hkl not in active_hklIDs:
                excluded[i] = True
        if np.any(excluded):
            raise RuntimeError(
                "The following requested hkls are marked as excluded: "
                + f"{hkls[excluded]}"
            )

        # ok, now re-assign active_hklIDs
        active_hklIDs = hkls

    # logging output
    shkls = plane_data.getHKLs(*active_hklIDs, asStr=True)
    logger.debug("building eta_ome maps using hkls: %s", [f'[{i}]' for i in shkls])

    # grad imageseries dict from cfg
    imsd = cfg.image_series

    # handle omega period
    ome_period, _ = _process_omegas(imsd)

    start = timeit.default_timer()

    # make eta_ome maps
    eta_ome = instrument.GenerateEtaOmeMaps(
        imsd,
        cfg.instrument.hedm,
        plane_data,
        active_hkls=active_hklIDs,
        eta_step=cfg.find_orientations.orientation_maps.eta_step,
        threshold=cfg.find_orientations.orientation_maps.threshold,
        ome_period=ome_period,
    )

    logger.debug("\t\t...took %f seconds", timeit.default_timer() - start)

    if save:
        # save maps
        fn = cfg.find_orientations.orientation_maps.file
        eta_ome.save(fn)

        logger.info(f'saved eta/ome orientation maps to "{fn}"')

    return eta_ome


def _filter_eta_ome_maps(eta_ome, filter_stdev=False):
    """
    Apply median and gauss-laplace filtering to remove streak artifacts.

    Parameters
    ----------
    eta_ome : TYPE
        DESCRIPTION.
    filter_stdev : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    eta_ome : TYPE
        DESCRIPTION.

    """
    gl_filter = ndimage.filters.gaussian_laplace
    for pf in eta_ome.dataStore:
        # first compoute row-wise median over omega channel
        ome_median = np.tile(np.nanmedian(pf, axis=0), (len(pf), 1))

        # subtract
        # !!! this changes the reference obj, but fitlering does not
        pf -= ome_median

        # filter
        # !!! False/None: only row median subtraction
        #     True: use default
        #     scalar: stdev for filter in pixels
        # ??? simplify this behavior
        if filter_stdev:
            if isinstance(filter_stdev, bool):
                filter_stdev = filter_stdev_DFLT
            pf[:] = -gl_filter(pf, filter_stdev)


def create_clustering_parameters(cfg, eta_ome):
    """
    Compute min samples and mean reflections per grain for clustering

    Parameters
    ----------
    cfg : TYPE
        DESCRIPTION.
    eta_ome : TYPE
        DESCRIPTION.

    Returns
    -------
    Tuple of (min_samples, mean_rpg)

    """

    compl_thresh = cfg.find_orientations.clustering.completeness
    percentile = cfg.find_orientations.seed_search.reflection_statistics_percentile
    stats = compute_reflection_statistics(cfg, eta_ome)
    seed_refl_per_grain = stats.seed_reflections_per_grain(
        cfg.find_orientations.seed_search.friedel_pairing
    )

    min_samples = max(
        int(
            np.floor(
                0.5
                * compl_thresh
                * _clipped_percentile(seed_refl_per_grain, percentile)
            )
        ),
        2,
    )
    mean_rpg = int(np.round(np.average(stats.active_reflections_per_grain)))

    return min_samples, mean_rpg


def find_orientations(
    cfg: root.RootConfig,
    hkls: Optional[NDArray | list[int]] = None,
    clean: bool = False,
    profile: bool = False,
    use_direct_testing: bool = False,
):
    """


    Parameters
    ----------
    cfg : TYPE
        DESCRIPTION.
    hkls : TYPE, optional
        DESCRIPTION. The default is None.
    clean : TYPE, optional
        DESCRIPTION. The default is False.
    profile : TYPE, optional
        DESCRIPTION. The default is False.
    use_direct_search : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    # grab objects from config
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    plane_data = cfg.material.plane_data
    imsd = cfg.image_series
    instr = cfg.instrument.hedm
    eta_ranges = cfg.find_orientations.eta.range

    # tolerances
    tth_tol = plane_data.tThWidth
    eta_tol = np.radians(cfg.find_orientations.eta.tolerance)
    ome_tol = np.radians(cfg.find_orientations.omega.tolerance)

    # handle omega period
    ome_period, _ = _process_omegas(imsd)

    # for multiprocessing
    ncpus = cfg.multiprocessing

    # thresholds
    image_threshold = cfg.find_orientations.orientation_maps.threshold
    on_map_threshold = cfg.find_orientations.threshold
    compl_thresh = cfg.find_orientations.clustering.completeness

    # clustering
    cl_algorithm = cfg.find_orientations.clustering.algorithm
    cl_radius = cfg.find_orientations.clustering.radius

    # =========================================================================
    # ORIENTATION SCORING
    # =========================================================================
    do_grid_search = cfg.find_orientations.use_quaternion_grid is not None

    if use_direct_testing:
        npdiv_DFLT = 2
        params = dict(
            plane_data=plane_data,
            instrument=instr,
            imgser_dict=imsd,
            tth_tol=tth_tol,
            eta_tol=eta_tol,
            ome_tol=ome_tol,
            eta_ranges=np.radians(eta_ranges),
            ome_period=np.radians(ome_period),
            npdiv=npdiv_DFLT,
            threshold=image_threshold,
        )

        logger.info("\tusing direct search on %d processes", ncpus)

        # handle search space
        if cfg.find_orientations.use_quaternion_grid is None:
            # doing seeded search
            logger.debug("Will perform seeded search")
            logger.debug(
                "\tgenerating search quaternion list using %d processes", ncpus
            )
            start = timeit.default_timer()

            # need maps
            eta_ome = load_eta_ome_maps(cfg, plane_data, imsd, hkls=hkls, clean=clean)

            # generate trial orientations
            qfib = generate_orientation_candidates(cfg, eta_ome)

            logger.info("\t\t...took %f seconds", timeit.default_timer() - start)
        else:
            # doing grid search
            try:
                qfib = np.load(cfg.find_orientations.use_quaternion_grid)
            except IOError:
                raise RuntimeError(
                    "specified quaternion grid file '%s' not found!"
                    % cfg.find_orientations.use_quaternion_grid
                )

        if qfib.size == 0:
            raise RuntimeError("seed search did not generate any orientation candidates")

        # execute direct search
        nworkers = _pool_worker_count(qfib.shape[1], ncpus)
        if nworkers > 1 and _spawn_pool_is_expensive():
            logger.info(
                "\tusing serial direct orientation testing on spawn multiprocessing context"
            )
            nworkers = 1
        if nworkers > 1:
            chunksize = _pool_chunksize(qfib.shape[1], nworkers)
            with const.mp_context.Pool(
                nworkers,
                indexer.test_orientation_FF_init,
                (params,),
            ) as pool:
                completeness = pool.map(
                    indexer.test_orientation_FF_reduced,
                    qfib.T,
                    chunksize=chunksize,
                )
        else:
            indexer.test_orientation_FF_init(params)
            completeness = list(map(indexer.test_orientation_FF_reduced, qfib.T))
    else:
        logger.debug("\tusing map search with paintGrid on %d processes", ncpus)

        start = timeit.default_timer()

        # handle eta-ome maps
        eta_ome = load_eta_ome_maps(cfg, plane_data, imsd, hkls=hkls, clean=clean)

        # handle search space
        if cfg.find_orientations.use_quaternion_grid is None:
            # doing seeded search
            logger.debug(
                "\tgenerating search quaternion list using %d processes", ncpus
            )
            start = timeit.default_timer()

            qfib = generate_orientation_candidates(cfg, eta_ome)
            logger.info("\t\t...took %f seconds", timeit.default_timer() - start)
        else:
            # doing grid search
            try:
                qfib = np.load(cfg.find_orientations.use_quaternion_grid)
            except IOError:
                raise RuntimeError(
                    "specified quaternion grid file '%s' not found!"
                    % cfg.find_orientations.use_quaternion_grid
                )
        if qfib.size == 0:
            raise RuntimeError("seed search did not generate any orientation candidates")

        # do map-based indexing
        start = timeit.default_timer()

        logger.info("will test %d quaternions using %d processes", qfib.shape[1], ncpus)

        completeness = indexer.paintGrid(
            qfib,
            eta_ome,
            etaRange=np.radians(cfg.find_orientations.eta.range),
            omeTol=np.radians(cfg.find_orientations.omega.tolerance),
            etaTol=np.radians(cfg.find_orientations.eta.tolerance),
            omePeriod=np.radians(cfg.find_orientations.omega.period),
            threshold=on_map_threshold,
            doMultiProc=ncpus > 1,
            nCPUs=ncpus,
        )
        logger.info("\t\t...took %f seconds", timeit.default_timer() - start)
    completeness = np.array(completeness)

    logger.info(
        "\tSaving %d scored orientations with max completeness %f%%",
        qfib.shape[1],
        100 * np.max(completeness),
    )

    results = {}
    results['scored_orientations'] = {
        'test_quaternions': qfib,
        'score': completeness,
    }

    # =========================================================================
    # CLUSTERING AND GRAINS OUTPUT
    # =========================================================================

    logger.debug("\trunning clustering using '%s'", cl_algorithm)

    start = timeit.default_timer()

    exact_merge_sparse_search = (
        not do_grid_search
        and cfg.find_orientations.use_quaternion_grid is None
        and cfg.find_orientations.seed_search.candidate_generator
        in ('pairwise-greedy', 'pairwise-consensus')
    )

    sparse_candidate_search = (
        not do_grid_search
        and cfg.find_orientations.use_quaternion_grid is None
        and cfg.find_orientations.seed_search.candidate_generator
        in ('pairwise', 'pairwise-greedy', 'pairwise-consensus')
    )

    if exact_merge_sparse_search:
        logger.info(
            "\tmerging sparse candidates using exact quaternion misorientation"
        )
        qbar, cl = merge_orientations_by_misorientation(
            completeness,
            qfib,
            plane_data.q_sym,
            compl_thresh,
            cl_radius,
        )
        logger.info("\t\t...took %f seconds", (timeit.default_timer() - start))
        logger.info("\tfound %d grains", qbar.shape[1])
        results['qbar'] = qbar
        return results

    if do_grid_search or sparse_candidate_search:
        min_samples = 1
        mean_rpg = 1
    else:
        min_samples, mean_rpg = create_clustering_parameters(cfg, eta_ome)

    logger.info("\tmean reflections per grain: %d", mean_rpg)
    logger.info("\tneighborhood size: %d", min_samples)

    qbar, cl = run_cluster(
        completeness,
        qfib,
        plane_data.q_sym,
        cfg,
        min_samples=min_samples,
        compl_thresh=compl_thresh,
        radius=cl_radius,
    )

    logger.info("\t\t...took %f seconds", (timeit.default_timer() - start))
    logger.info("\tfound %d grains", qbar.shape[1])

    results['qbar'] = qbar

    return results
