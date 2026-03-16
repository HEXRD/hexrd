from dataclasses import dataclass, field
from math import gcd
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
activeHklStatisticsMP = None


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
    fiber_family_id: tuple[int, ...] | None = None
    intensity: float = 0.0
    support: int = 1


@dataclass
class SeedPeak:
    eta: float
    ome: float
    intensity: float
    support: int = 1


@dataclass
class MetaSeedReflection:
    seed_index: int
    hkl_id: int
    hkl: NDArray[np.float64]
    tth: float
    eta: float
    ome: float
    gvec_s: NDArray[np.float64]
    fiber_family_id: tuple[int, ...] | None = None
    intensity: float = 0.0
    support: int = 1
    weight: int = 1
    friedel_status: str = 'unpaired'
    mate_expected: bool = False


@dataclass
class SeedPeakGroup:
    seed_index: int
    hkl_id: int
    hkl: NDArray[np.float64]
    tth: float
    peaks: list[SeedPeak]
    fiber_family_id: tuple[int, ...] | None = None


@dataclass
class ReflectionStatistics:
    sample_count: int
    active_reflections_per_grain: NDArray[np.int64]
    seed_reflections_raw_per_grain: NDArray[np.int64]
    seed_reflections_reduced_per_grain: NDArray[np.int64]
    seed_hkls_per_grain: NDArray[np.int64]
    seed_reflections_raw_by_hkl: dict[int, NDArray[np.int64]]
    seed_reflections_reduced_by_hkl: dict[int, NDArray[np.int64]]
    seed_family_ids: tuple[tuple[int, ...], ...] = ()
    seed_family_visibility_prob: dict[tuple[int, ...], float] = field(default_factory=dict)
    seed_family_pair_visibility_prob: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = field(default_factory=dict)
    seed_family_eta_reliability: dict[tuple[int, ...], float] = field(default_factory=dict)

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


def _fiber_family_key(hkl) -> tuple[int, ...]:
    hkl = np.rint(np.asarray(hkl, dtype=float)).astype(int).reshape(-1)
    nonzero = np.flatnonzero(hkl)
    if nonzero.size == 0:
        return tuple(int(x) for x in hkl)

    divisor = 0
    for value in np.abs(hkl[nonzero]):
        divisor = gcd(divisor, int(value))
    if divisor <= 0:
        divisor = 1

    primitive = (hkl // divisor).astype(int)
    first_nonzero = primitive[nonzero[0]]
    if first_nonzero < 0:
        primitive = -primitive

    return tuple(int(x) for x in primitive.tolist())


def _reflection_family_id(reflection) -> tuple[int, ...]:
    family_id = getattr(reflection, 'fiber_family_id', None)
    if family_id is not None:
        return tuple(int(x) for x in family_id)

    return _fiber_family_key(reflection.hkl)


def _eta_reliability(eta: float) -> float:
    # HEDM reflections are most stable near the eta horizon (0, pi) and
    # least stable near the rotation-axis projection (±pi/2).
    return float(abs(np.cos(float(eta))) ** 2)


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


def _periodic_angle_in_ranges(angle, ranges) -> bool:
    if np.isnan(angle):
        return False

    twopi = 2.0 * np.pi
    angle = float(np.mod(angle, twopi))
    for start, stop in ranges:
        start = float(np.mod(start, twopi))
        stop = float(np.mod(stop, twopi))
        if np.isclose(start, stop):
            return True
        if start <= stop:
            if start <= angle <= stop:
                return True
        else:
            if angle >= start or angle <= stop:
                return True

    return False


def _friedel_partner_visible(
    eta: float,
    ome: float,
    eta_ranges,
    ome_ranges,
) -> bool:
    return _periodic_angle_in_ranges(eta, eta_ranges) and _periodic_angle_in_ranges(
        ome,
        ome_ranges,
    )


def _friedel_status_weight(status: str, support: int) -> int:
    support = max(int(support), 1)
    if status == 'paired_visible':
        return support + 2
    if status == 'single_occluded':
        return support + 1

    return support


def _reflection_axis_from_angles(
    tth: float,
    eta: float,
    ome: float,
    chi: float,
) -> NDArray[np.float64]:
    gvec_s = xfcapi.angles_to_gvec(
        np.atleast_2d([tth, eta, ome]),
        chi=chi,
    ).T.reshape(3)
    return mutil.unitVector(np.asarray(gvec_s, dtype=float).reshape(3, 1)).reshape(3)


def _friedel_pair_matching(
    peaks: list[SeedPeak],
    tth: float,
    chi: float,
    eta_tol: float,
    ome_tol: float,
):
    peak_count = len(peaks)
    if peak_count == 0:
        return None

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
    best_match = np.full(peak_count, -1, dtype=int)
    best_cost = np.full(peak_count, np.inf)

    if np.count_nonzero(valid) >= 2:
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
        source_ids = np.tile(np.arange(peak_count), len(periodic_shifts))
        tree = cKDTree(tiled_coords)

        search_radius = np.sqrt(2.0)
        intensity_weight = 0.05
        for i in np.where(valid)[0]:
            pred_point = np.array([pred_etas[i], pred_omes[i]], dtype=float) / scale
            candidate_ids = np.unique(
                source_ids[tree.query_ball_point(pred_point, search_radius)]
            )

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

    return {
        'etas': etas,
        'omes': omes,
        'intensities': intensities,
        'pred_etas': pred_etas,
        'pred_omes': pred_omes,
        'valid': valid,
        'best_match': best_match,
        'best_cost': best_cost,
    }


def _meta_reflections_from_peaks(
    peaks: list[SeedPeak],
    seed_index: int,
    hkl_id: int,
    hkl: NDArray[np.float64],
    fiber_family_id: tuple[int, ...] | None,
    tth: float,
    chi: float,
    eta_tol: float,
    ome_tol: float,
    eta_ranges,
    ome_ranges,
    use_friedel_pairing: bool,
) -> list[MetaSeedReflection]:
    if not peaks:
        return []

    if not use_friedel_pairing:
        reflections = []
        for peak in peaks:
            reflections.append(
                MetaSeedReflection(
                    seed_index=seed_index,
                    hkl_id=int(hkl_id),
                    hkl=np.asarray(hkl, dtype=float),
                    fiber_family_id=fiber_family_id,
                    tth=float(tth),
                    eta=float(peak.eta),
                    ome=float(peak.ome),
                    gvec_s=_reflection_axis_from_angles(tth, peak.eta, peak.ome, chi),
                    intensity=float(peak.intensity),
                    support=int(peak.support),
                    weight=int(peak.support),
                    friedel_status='unpaired',
                    mate_expected=False,
                )
            )
        return reflections

    match_data = _friedel_pair_matching(peaks, tth, chi, eta_tol, ome_tol)
    if match_data is None:
        return []

    valid = match_data['valid']
    best_match = match_data['best_match']
    best_cost = match_data['best_cost']
    pred_etas = match_data['pred_etas']
    pred_omes = match_data['pred_omes']
    intensities = match_data['intensities']

    reflections = []
    used = np.zeros(len(peaks), dtype=bool)
    order = np.argsort(best_cost)

    for i in order:
        if used[i]:
            continue

        j = best_match[i]
        if j < 0 or used[j] or best_match[j] != i:
            continue

        rep_idx = i if intensities[i] >= intensities[j] else j
        rep_peak = peaks[rep_idx]
        support = peaks[i].support + peaks[j].support
        status = 'paired_visible'
        reflections.append(
            MetaSeedReflection(
                seed_index=seed_index,
                hkl_id=int(hkl_id),
                hkl=np.asarray(hkl, dtype=float),
                fiber_family_id=fiber_family_id,
                tth=float(tth),
                eta=float(rep_peak.eta),
                ome=float(rep_peak.ome),
                gvec_s=_reflection_axis_from_angles(
                    tth,
                    rep_peak.eta,
                    rep_peak.ome,
                    chi,
                ),
                intensity=float(peaks[i].intensity + peaks[j].intensity),
                support=int(support),
                weight=_friedel_status_weight(status, support),
                friedel_status=status,
                mate_expected=True,
            )
        )
        used[i] = True
        used[j] = True

    for i, peak in enumerate(peaks):
        if used[i]:
            continue

        mate_expected = bool(valid[i]) and _friedel_partner_visible(
            pred_etas[i],
            pred_omes[i],
            eta_ranges,
            ome_ranges,
        )
        status = 'single_missing' if mate_expected else 'single_occluded'
        reflections.append(
            MetaSeedReflection(
                seed_index=seed_index,
                hkl_id=int(hkl_id),
                hkl=np.asarray(hkl, dtype=float),
                fiber_family_id=fiber_family_id,
                tth=float(tth),
                eta=float(peak.eta),
                ome=float(peak.ome),
                gvec_s=_reflection_axis_from_angles(tth, peak.eta, peak.ome, chi),
                intensity=float(peak.intensity),
                support=int(peak.support),
                weight=_friedel_status_weight(status, peak.support),
                friedel_status=status,
                mate_expected=mate_expected,
            )
        )

    return reflections


def _pair_friedel_seed_peaks(
    peaks: list[SeedPeak],
    tth: float,
    chi: float,
    eta_tol: float,
    ome_tol: float,
) -> list[SeedPeak]:
    if len(peaks) < 2:
        return peaks
    match_data = _friedel_pair_matching(peaks, tth, chi, eta_tol, ome_tol)
    if match_data is None or np.count_nonzero(match_data['valid']) < 2:
        return peaks

    best_match = match_data['best_match']
    best_cost = match_data['best_cost']
    intensities = match_data['intensities']
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


def _seed_selection_budget(cfg, candidate_pool_size: int) -> int:
    requested = int(cfg.find_orientations.seed_search.auto_select_count)
    return min(max(requested, 1), max(int(candidate_pool_size), 1))


def _plane_data_hkl_records(
    plane_data,
    active_hkl_ids,
):
    active_hkl_ids = np.asarray(active_hkl_ids, dtype=int).reshape(-1)
    active_hkls = np.asarray(plane_data.getHKLs(*active_hkl_ids).T, dtype=float)
    pd_hkl_idx = np.asarray(
        plane_data.getHKLID(active_hkls, master=False),
        dtype=int,
    )
    tth = np.asarray(plane_data.getTTh()[pd_hkl_idx], dtype=float)
    tth_width = getattr(plane_data, 'tThWidth', None)

    records = []
    for active_idx, (hkl_id, hkl, pd_idx, tth_value) in enumerate(
        zip(active_hkl_ids, active_hkls, pd_hkl_idx, tth)
    ):
        if tth_width is not None and np.isfinite(float(tth_width)):
            tth_lo = float(tth_value - 0.5 * float(tth_width))
            tth_hi = float(tth_value + 0.5 * float(tth_width))
        else:
            hkl_data = plane_data.hklDataList[int(pd_idx)]
            tth_lo = float(hkl_data['tThetaLo'])
            tth_hi = float(hkl_data['tThetaHi'])

        records.append(
            {
                'active_idx': int(active_idx),
                'hkl_id': int(hkl_id),
                'hkl': np.asarray(hkl, dtype=float),
                'family_id': _fiber_family_key(hkl),
                'pd_hkl_idx': int(pd_idx),
                'tth': float(tth_value),
                'tth_range': (float(tth_lo), float(tth_hi)),
                'order': float(np.linalg.norm(np.asarray(hkl, dtype=float))),
            }
        )

    return records


def _active_hkl_records(eta_ome):
    return _plane_data_hkl_records(
        eta_ome.planeData,
        np.asarray(eta_ome.iHKLList, dtype=int),
    )


def _merged_active_tth_groups(active_records):
    if not active_records:
        return []

    sorted_records = sorted(
        active_records,
        key=lambda record: (record['tth_range'][0], record['tth_range'][1]),
    )
    groups = []
    current_group = [sorted_records[0]]
    current_hi = sorted_records[0]['tth_range'][1]
    for record in sorted_records[1:]:
        lo, hi = record['tth_range']
        if lo <= current_hi:
            current_group.append(record)
            current_hi = max(current_hi, hi)
        else:
            groups.append(current_group)
            current_group = [record]
            current_hi = hi
    groups.append(current_group)
    return groups


def _degenerate_tth_group(
    group,
    tol=const.sqrt_epsf,
):
    if len(group) < 2:
        return False

    tth_vals = np.asarray([record['tth'] for record in group], dtype=float)
    return bool(np.max(tth_vals) - np.min(tth_vals) <= max(float(tol), const.sqrt_epsf))


def _separable_seed_family_records(active_records):
    groups = _merged_active_tth_groups(active_records)
    separable = []
    for group in groups:
        family_ids = {record['family_id'] for record in group}
        if len(group) != 1 or len(family_ids) != 1:
            continue
        separable.append(group[0].copy())

    return separable


def active_hkl_statistics_init(params):
    global activeHklStatisticsMP
    activeHklStatisticsMP = params


def active_hkl_statistics_cleanup():
    global activeHklStatisticsMP
    activeHklStatisticsMP = None


def active_hkl_statistics_reduced(grain_range):
    params = activeHklStatisticsMP
    grain_param_list = params['grain_param_list'][slice(*grain_range)]
    active_hkl_ids = params['active_hkl_ids']
    active_index_by_hkl = params['active_index_by_hkl']

    sim_results = params['instr'].simulate_rotation_series(
        params['plane_data'],
        grain_param_list,
        eta_ranges=params['eta_ranges'],
        ome_ranges=params['ome_ranges'],
        ome_period=params['ome_period'],
    )

    ngrains = grain_param_list.shape[0]
    active_counts = np.zeros(ngrains, dtype=np.int64)
    per_hkl_counts = np.zeros((len(active_hkl_ids), ngrains), dtype=np.int64)

    for sim_result in sim_results.values():
        valid_ids_by_grain = sim_result[0]
        for igrain, valid_ids in enumerate(valid_ids_by_grain):
            valid_ids = np.asarray(valid_ids, dtype=int)
            if valid_ids.size == 0:
                continue

            mask = np.isin(valid_ids, active_hkl_ids)
            visible_ids = valid_ids[mask]
            if visible_ids.size == 0:
                continue

            active_counts[igrain] += int(visible_ids.size)
            for hkl_id in visible_ids:
                per_hkl_counts[active_index_by_hkl[int(hkl_id)], igrain] += 1

    return {
        'active_reflections_per_grain': active_counts,
        'active_reflections_by_hkl': per_hkl_counts,
    }


def _compute_active_hkl_statistics(
    cfg,
    active_hkl_ids,
):
    plane_data = cfg.material.plane_data
    instr = cfg.instrument.hedm
    active_hkl_ids = np.asarray(active_hkl_ids, dtype=int).reshape(-1)
    if active_hkl_ids.size == 0:
        return {
            'sample_count': 0,
            'active_reflections_per_grain': np.array([], dtype=np.int64),
            'active_reflections_by_hkl': {},
        }

    sample_count = max(
        int(cfg.find_orientations.seed_search.reflection_statistics_samples),
        1,
    )
    sample_seed = int(cfg.find_orientations.seed_search.reflection_statistics_seed)
    eta_ranges = np.radians(cfg.find_orientations.eta.range)
    ome_period, ome_ranges = _process_omegas(cfg.image_series)

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
        'active_index_by_hkl': {
            int(hkl_id): idx for idx, hkl_id in enumerate(active_hkl_ids)
        },
        'eta_ranges': eta_ranges,
        'ome_ranges': np.radians(ome_ranges),
        'ome_period': np.radians(ome_period),
    }

    nworkers = _pool_worker_count(sample_count, cfg.multiprocessing)
    if nworkers > 1 and _spawn_pool_is_expensive():
        logger.info(
            "\tusing serial active-hkl statistics simulation on spawn multiprocessing context"
        )
        nworkers = 1

    grain_ranges = distribute_tasks(sample_count, nworkers)
    if nworkers > 1:
        with const.mp_context.Pool(
            nworkers,
            active_hkl_statistics_init,
            (params,),
        ) as pool:
            results = pool.map(active_hkl_statistics_reduced, grain_ranges)
    else:
        active_hkl_statistics_init(params)
        try:
            results = list(map(active_hkl_statistics_reduced, grain_ranges))
        finally:
            active_hkl_statistics_cleanup()

    active_reflections_per_grain = np.concatenate(
        [result['active_reflections_per_grain'] for result in results]
    )
    active_reflections_by_hkl = {
        int(hkl_id): np.concatenate(
            [result['active_reflections_by_hkl'][idx] for result in results]
        )
        for idx, hkl_id in enumerate(active_hkl_ids)
    }
    return {
        'sample_count': sample_count,
        'active_reflections_per_grain': active_reflections_per_grain,
        'active_reflections_by_hkl': active_reflections_by_hkl,
    }


def _select_seed_family_records(
    family_records,
    family_pair_priority,
    budget: int,
):
    if not family_records:
        return []

    budget = min(max(int(budget), 1), len(family_records))
    record_by_family = {record['family_id']: record for record in family_records}
    selected = []
    remaining = set(record_by_family)

    while remaining and len(selected) < budget:
        best_family = None
        best_score = -np.inf
        for family_id in remaining:
            record = record_by_family[family_id]
            score = float(record['unary_score'])
            if selected:
                pair_scores = [
                    family_pair_priority.get(
                        _family_pair_key(family_id, selected_family),
                        0.0,
                    )
                    for selected_family in selected
                ]
                if pair_scores:
                    score *= float(np.mean(pair_scores))

            if score > best_score:
                best_score = score
                best_family = family_id

        if best_family is None:
            break

        selected.append(best_family)
        remaining.remove(best_family)

    return [record_by_family[family_id] for family_id in selected]


def _auto_select_seed_hkl_indices(
    cfg,
    eta_ome,
    candidate_indices,
):
    candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
    if candidate_indices.size == 0:
        return []

    active_records = [
        record
        for record in _active_hkl_records(eta_ome)
        if int(record['active_idx']) in set(candidate_indices.tolist())
    ]
    family_records = _separable_seed_family_records(active_records)
    if not family_records:
        logger.warning(
            "\tno separable non-degenerate seed hkls were found in the active eta-omega maps"
        )
        return []

    candidate_seed_indices = [record['active_idx'] for record in family_records]
    try:
        stats = compute_reflection_statistics(
            cfg,
            eta_ome,
            seed_hkl_indices=candidate_seed_indices,
        )
    except Exception as exc:
        logger.warning(
            "\tseed auto-selection statistics failed (%s); falling back to geometry-only ranking",
            exc,
        )
        stats = None

    use_friedel_pairing = cfg.find_orientations.seed_search.friedel_pairing
    plane_data = eta_ome.planeData
    sym_hkls = plane_data.getSymHKLs()
    bmat = plane_data.latVecOps['B']
    pair_tol = np.radians(cfg.find_orientations.seed_search.pairwise_tolerance)
    epsilon = 1.0 / max(getattr(stats, 'sample_count', 1), 1)

    deduped_records = {}
    for record in family_records:
        crystal_dirs = mutil.unitVector(np.dot(bmat, sym_hkls[record['pd_hkl_idx']]))
        crystal_dirs = np.asarray(crystal_dirs, dtype=float)
        if crystal_dirs.ndim == 1:
            crystal_dirs = crystal_dirs.reshape(3, 1)

        visibility = epsilon
        support = epsilon
        eta_reliability = 1.0
        if stats is not None:
            per_hkl_counts = stats.seed_reflections_by_hkl(use_friedel_pairing)
            counts = np.asarray(per_hkl_counts.get(record['hkl_id'], []), dtype=float)
            if counts.size:
                visibility = max(float(np.mean(counts > 0.0)), epsilon)
                support = max(float(np.mean(counts)), epsilon)
            eta_reliability = max(
                float(stats.seed_family_eta_reliability.get(record['family_id'], 0.0)),
                epsilon,
            )

        order_score = 1.0 / (1.0 + max(record['order'], 0.0))
        unary_score = visibility * support * eta_reliability * order_score
        ranked = dict(record)
        ranked.update(
            {
                'visibility': float(visibility),
                'support': float(support),
                'eta_reliability': float(eta_reliability),
                'order_score': float(order_score),
                'crystal_dirs': crystal_dirs,
                'multiplicity': int(crystal_dirs.shape[1]),
                'unary_score': float(unary_score),
            }
        )
        prev = deduped_records.get(record['family_id'])
        if prev is None or (
            ranked['unary_score'] > prev['unary_score']
            or (
                np.isclose(ranked['unary_score'], prev['unary_score'])
                and ranked['tth'] < prev['tth']
            )
        ):
            deduped_records[record['family_id']] = ranked

    family_records = list(deduped_records.values())
    if not family_records:
        return []

    family_pair_priority = {}
    for i, record_i in enumerate(family_records):
        family_i = record_i['family_id']
        for j in range(i + 1, len(family_records)):
            record_j = family_records[j]
            family_j = record_j['family_id']
            pair_key = _family_pair_key(family_i, family_j)
            covis = (
                stats.seed_family_pair_visibility_prob.get(pair_key, 0.0)
                if stats is not None
                else epsilon
            )
            geometry = _geometric_family_pair_score(
                record_i['crystal_dirs'],
                record_j['crystal_dirs'],
                pair_tol,
            )
            family_pair_priority[pair_key] = float(
                max(covis, epsilon)
                * max(geometry, epsilon)
            )

    budget = _seed_selection_budget(cfg, len(family_records))
    selected_records = _select_seed_family_records(
        family_records,
        family_pair_priority,
        budget,
    )
    selected_indices = [int(record['active_idx']) for record in selected_records]

    logger.info(
        "\tauto-selected %d separable seed hkls from %d candidate maps: %s",
        len(selected_records),
        len(candidate_indices),
        [str(record['hkl']) for record in selected_records],
    )

    return selected_indices


def _active_observability_target(cfg) -> int:
    seed_goal = max(int(cfg.find_orientations.seed_search.auto_select_count), 3)
    return max(6, 2 * seed_goal)


def _auto_select_active_hkl_ids(cfg, candidate_hkl_ids=None):
    plane_data = cfg.material.plane_data
    if candidate_hkl_ids is None:
        candidate_hkl_ids = np.asarray(
            plane_data.getHKLID(plane_data.hkls, master=True),
            dtype=int,
        )
    else:
        candidate_hkl_ids = np.asarray(candidate_hkl_ids, dtype=int).reshape(-1)

    if candidate_hkl_ids.size == 0:
        return np.array([], dtype=int)

    active_records = _plane_data_hkl_records(plane_data, candidate_hkl_ids)
    stats = _compute_active_hkl_statistics(cfg, candidate_hkl_ids)
    sample_count = max(int(stats['sample_count']), 1)
    epsilon = 1.0 / sample_count

    for record in active_records:
        counts = np.asarray(
            stats['active_reflections_by_hkl'].get(record['hkl_id'], []),
            dtype=float,
        )
        if counts.size == 0:
            visibility = 0.0
            support = 0.0
        else:
            visibility = float(np.mean(counts > 0.0))
            support = float(np.mean(counts))
        record['visibility'] = visibility
        record['support'] = support
        record['order_score'] = 1.0 / (1.0 + max(record['order'], 0.0))
        record['active_score'] = (
            max(visibility, epsilon)
            * max(support, epsilon)
            * record['order_score']
        )

    groups = _merged_active_tth_groups(active_records)
    clean_records = []
    degenerate_groups = []
    for group in groups:
        family_ids = {record['family_id'] for record in group}
        if len(group) == 1 and len(family_ids) == 1:
            clean_records.append(group[0])
        elif _degenerate_tth_group(group):
            degenerate_groups.append(group)

    clean_records.sort(key=lambda record: (record['tth'], -record['active_score']))
    selected = [record for record in clean_records if record['visibility'] > 0.0]
    selected_ids = [int(record['hkl_id']) for record in selected]

    target_reflections = _active_observability_target(cfg)
    p10_active = _clipped_percentile(
        np.sum(
            np.vstack(
                [
                    stats['active_reflections_by_hkl'][hkl_id]
                    for hkl_id in selected_ids
                ]
            ),
            axis=0,
        )
        if selected_ids
        else np.zeros(sample_count, dtype=float),
        cfg.find_orientations.seed_search.reflection_statistics_percentile,
    )

    if p10_active < target_reflections and degenerate_groups:
        degenerate_groups.sort(
            key=lambda group: (
                min(record['tth'] for record in group),
                -max(record['active_score'] for record in group),
            )
        )
        for group in degenerate_groups:
            representatives = sorted(
                group,
                key=lambda record: (
                    -record['active_score'],
                    record['tth'],
                ),
            )
            if not representatives:
                continue

            record = representatives[0]
            if record['hkl_id'] in selected_ids or record['visibility'] <= 0.0:
                continue

            selected.append(record)
            selected_ids.append(int(record['hkl_id']))
            p10_active = _clipped_percentile(
                np.sum(
                    np.vstack(
                        [
                            stats['active_reflections_by_hkl'][hkl_id]
                            for hkl_id in selected_ids
                        ]
                    ),
                    axis=0,
                ),
                cfg.find_orientations.seed_search.reflection_statistics_percentile,
            )
            if p10_active >= target_reflections:
                break

    selected.sort(key=lambda record: record['tth'])
    selected_ids = np.asarray([int(record['hkl_id']) for record in selected], dtype=int)
    logger.info(
        "\tauto-selected %d active hkls (target p%.1f active reflections/grain >= %.1f, achieved %.1f): %s",
        selected_ids.size,
        float(np.clip(cfg.find_orientations.seed_search.reflection_statistics_percentile, 0.0, 100.0)),
        float(target_reflections),
        float(p10_active),
        [str(record['hkl']) for record in selected],
    )
    return selected_ids


def _resolved_orientation_map_hkl_ids(cfg, hkls=None):
    plane_data = cfg.material.plane_data
    if hkls is not None:
        hkls = np.asarray(hkls)
        if hkls.ndim == 2:
            hkls = plane_data.getHKLID(hkls.tolist(), master=True)
        return np.asarray(hkls, dtype=int)

    if cfg.find_orientations.orientation_maps.active_hkl_selection == 'auto':
        return _auto_select_active_hkl_ids(cfg)

    active_hkl_ids = np.asarray(
        plane_data.getHKLID(plane_data.hkls, master=True),
        dtype=int,
    )
    temp = np.asarray(cfg.find_orientations.orientation_maps.active_hkls)
    if temp.ndim == 0:
        return active_hkl_ids
    if temp.ndim == 1:
        return np.asarray(temp, dtype=int)
    if temp.ndim == 2:
        return np.asarray(
            plane_data.getHKLID(temp.tolist(), master=True),
            dtype=int,
        )
    raise RuntimeError(
        'active_hkls spec must be 1-d or 2-d, not %d-d' % temp.ndim
    )


def _resolved_seed_hkl_indices(cfg, eta_ome):
    manual_seed_indices = cfg.find_orientations.seed_search.hkl_seeds
    auto_select = cfg.find_orientations.seed_search.auto_select_hkls

    if manual_seed_indices is not None and not auto_select:
        return [int(idx) for idx in manual_seed_indices]

    if manual_seed_indices is None and not auto_select:
        raise RuntimeError('"find_orientations:seed_search:hkl_seeds" must be defined for seeded search')

    candidate_indices = (
        np.asarray(manual_seed_indices, dtype=int)
        if manual_seed_indices is not None
        else np.arange(len(eta_ome.iHKLList), dtype=int)
    )
    return _auto_select_seed_hkl_indices(cfg, eta_ome, candidate_indices)


def _collect_seed_peak_groups(cfg, eta_ome):
    chi = cfg.instrument.hedm.chi
    seed_hkl_ids = _resolved_seed_hkl_indices(cfg, eta_ome)
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

    peak_groups = []
    total_raw_spots = 0
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
        peak_groups.append(
            SeedPeakGroup(
                seed_index=seed_index,
                hkl_id=int(pd_hkl_ids[seed_index]),
                hkl=np.asarray(this_hkl, dtype=float),
                fiber_family_id=_fiber_family_key(this_hkl),
                tth=float(this_tth),
                peaks=seed_peaks,
            )
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
        total_raw_spots=total_raw_spots,
    )
    return peak_groups, seed_crystal_dirs, params


def _collect_seed_reflections(cfg, eta_ome):
    peak_groups, seed_crystal_dirs, params = _collect_seed_peak_groups(cfg, eta_ome)
    reflections = []
    total_reduced_spots = 0
    for peak_group in peak_groups:
        seed_peaks = peak_group.peaks
        if params['use_friedel_pairing']:
            seed_peaks = _pair_friedel_seed_peaks(
                seed_peaks,
                peak_group.tth,
                params['chi'],
                params['pair_eta_tol'],
                params['pair_ome_tol'],
            )
        total_reduced_spots += len(seed_peaks)

        for seed_peak in seed_peaks:
            gvec_s = xfcapi.angles_to_gvec(
                np.atleast_2d([peak_group.tth, seed_peak.eta, seed_peak.ome]),
                chi=params['chi'],
            ).T.reshape(3)
            reflections.append(
                SeedReflection(
                    seed_index=peak_group.seed_index,
                    hkl_id=peak_group.hkl_id,
                    hkl=peak_group.hkl,
                    fiber_family_id=peak_group.fiber_family_id,
                    tth=peak_group.tth,
                    eta=seed_peak.eta,
                    ome=seed_peak.ome,
                    gvec_s=np.asarray(gvec_s, dtype=float),
                    intensity=seed_peak.intensity,
                    support=seed_peak.support,
                )
            )

    if params['use_friedel_pairing'] and params['total_raw_spots']:
        logger.info(
            "\tFriedel pairing reduced seed spots from %d to %d",
            params['total_raw_spots'],
            total_reduced_spots,
        )

    return reflections, seed_crystal_dirs, params


def _collect_pairwise_consensus_reflections(cfg, eta_ome):
    peak_groups, seed_crystal_dirs, params = _collect_seed_peak_groups(cfg, eta_ome)
    reflections = []
    status_counts = {
        'paired_visible': 0,
        'single_occluded': 0,
        'single_missing': 0,
        'unpaired': 0,
    }

    for peak_group in peak_groups:
        meta_reflections = _meta_reflections_from_peaks(
            peak_group.peaks,
            peak_group.seed_index,
            peak_group.hkl_id,
            peak_group.hkl,
            peak_group.fiber_family_id,
            peak_group.tth,
            params['chi'],
            params['pair_eta_tol'],
            params['pair_ome_tol'],
            params['eta_ranges'],
            params['ome_ranges'],
            params['use_friedel_pairing'],
        )
        reflections.extend(meta_reflections)
        for reflection in meta_reflections:
            status_counts.setdefault(reflection.friedel_status, 0)
            status_counts[reflection.friedel_status] += 1

    if params['use_friedel_pairing'] and params['total_raw_spots']:
        logger.info(
            "\tpairwise-consensus condensed %d raw seed spots into %d Friedel-aware meta-reflections (%d paired, %d occluded singles, %d expected-missing singles)",
            params['total_raw_spots'],
            len(reflections),
            status_counts['paired_visible'],
            status_counts['single_occluded'],
            status_counts['single_missing'],
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
            if _reflection_family_id(reflection_i) == _reflection_family_id(reflection_j):
                continue

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
    family_visible = np.zeros((len(params['seed_family_ids']), ngrains), dtype=bool)
    family_eta_quality_sum = np.zeros((len(params['seed_family_ids']), ngrains), dtype=float)
    family_eta_quality_count = np.zeros((len(params['seed_family_ids']), ngrains), dtype=np.int64)
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
                family_id = params['seed_family_id_by_hkl'][hkl_id]
                family_idx = params['seed_family_index_by_id'][family_id]
                per_seed_raw[seed_idx, igrain] += 1
                family_eta_quality_sum[family_idx, igrain] += _eta_reliability(angs[1])
                family_eta_quality_count[family_idx, igrain] += 1
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
        seed_hkl_counts[igrain] = len(
            {
                params['seed_family_id_by_hkl'][int(hkl_id)]
                for hkl_id in predicted_by_hkl
            }
        )
        for hkl_id in predicted_by_hkl:
            family_id = params['seed_family_id_by_hkl'][int(hkl_id)]
            family_visible[params['seed_family_index_by_id'][family_id], igrain] = True
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
        'seed_family_visible': family_visible,
        'seed_family_eta_quality_sum': family_eta_quality_sum,
        'seed_family_eta_quality_count': family_eta_quality_count,
    }


def compute_reflection_statistics(
    cfg,
    eta_ome,
    seed_hkl_indices=None,
) -> ReflectionStatistics:
    plane_data = cfg.material.plane_data
    instr = cfg.instrument.hedm
    active_hkl_ids = np.asarray(eta_ome.iHKLList, dtype=int)
    if seed_hkl_indices is None:
        seed_hkl_indices = np.asarray(
            _resolved_seed_hkl_indices(cfg, eta_ome),
            dtype=int,
        )
    else:
        seed_hkl_indices = np.asarray(seed_hkl_indices, dtype=int)
    seed_hkl_ids = np.asarray(active_hkl_ids[seed_hkl_indices], dtype=int)
    seed_family_ids = tuple(
        dict.fromkeys(
            _fiber_family_key(hkl)
            for hkl in plane_data.getHKLs(*seed_hkl_ids).T
        )
    )
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
        'seed_family_id_by_hkl': {
            int(hkl_id): _fiber_family_key(hkl)
            for hkl_id, hkl in zip(
                seed_hkl_ids,
                plane_data.getHKLs(*seed_hkl_ids).T,
            )
        },
        'seed_family_ids': seed_family_ids,
        'seed_family_index_by_id': {
            family_id: idx for idx, family_id in enumerate(seed_family_ids)
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
    seed_family_visible = np.concatenate(
        [result['seed_family_visible'] for result in results],
        axis=1,
    )
    seed_family_visibility_prob = {
        family_id: float(np.mean(seed_family_visible[idx]))
        for idx, family_id in enumerate(seed_family_ids)
    }
    seed_family_pair_visibility_prob = {}
    for i, family_i in enumerate(seed_family_ids):
        for j in range(i + 1, len(seed_family_ids)):
            family_j = seed_family_ids[j]
            pair_key = tuple(sorted((family_i, family_j)))
            seed_family_pair_visibility_prob[pair_key] = float(
                np.mean(seed_family_visible[i] & seed_family_visible[j])
            )
    seed_family_eta_quality_sum = np.sum(
        np.stack([result['seed_family_eta_quality_sum'] for result in results], axis=0),
        axis=0,
    )
    seed_family_eta_quality_count = np.sum(
        np.stack([result['seed_family_eta_quality_count'] for result in results], axis=0),
        axis=0,
    )
    seed_family_eta_reliability = {}
    for idx, family_id in enumerate(seed_family_ids):
        total_count = int(np.sum(seed_family_eta_quality_count[idx]))
        if total_count <= 0:
            seed_family_eta_reliability[family_id] = 0.0
        else:
            seed_family_eta_reliability[family_id] = float(
                np.sum(seed_family_eta_quality_sum[idx]) / total_count
            )

    stats = ReflectionStatistics(
        sample_count=sample_count,
        active_reflections_per_grain=active_reflections_per_grain,
        seed_reflections_raw_per_grain=seed_reflections_raw_per_grain,
        seed_reflections_reduced_per_grain=seed_reflections_reduced_per_grain,
        seed_hkls_per_grain=seed_hkls_per_grain,
        seed_reflections_raw_by_hkl=seed_reflections_raw_by_hkl,
        seed_reflections_reduced_by_hkl=seed_reflections_reduced_by_hkl,
        seed_family_ids=seed_family_ids,
        seed_family_visibility_prob=seed_family_visibility_prob,
        seed_family_pair_visibility_prob=seed_family_pair_visibility_prob,
        seed_family_eta_reliability=seed_family_eta_reliability,
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
    if stats.seed_family_eta_reliability:
        logger.info(
            "\tseed-family eta reliability range: %.3f-%.3f",
            float(min(stats.seed_family_eta_reliability.values())),
            float(max(stats.seed_family_eta_reliability.values())),
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
    unique_hkl_count = len(
        {_reflection_family_id(reflection) for reflection in reflections}
    )

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


def _family_pair_key(
    family_a: tuple[int, ...],
    family_b: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(sorted((tuple(family_a), tuple(family_b))))


def _geometric_family_pair_score(
    dirs_a: NDArray[np.float64],
    dirs_b: NDArray[np.float64],
    pair_tol: float,
) -> float:
    pair_tol = max(float(pair_tol), np.radians(0.25))
    dot = np.clip(np.abs(np.dot(dirs_a.T, dirs_b)), -1.0, 1.0)
    angles = np.sort(np.ravel(np.arccos(dot)))
    if angles.size == 0:
        return 0.0

    unique_angles = [float(angles[0])]
    merge_tol = 0.5 * pair_tol
    for angle in angles[1:]:
        if abs(float(angle) - unique_angles[-1]) > merge_tol:
            unique_angles.append(float(angle))

    unique_count = len(unique_angles)
    if unique_count <= 0:
        return 0.0

    score = 1.0 / unique_count
    if unique_count > 1:
        min_sep = min(np.diff(unique_angles))
        score *= 0.5 + 0.5 * min(1.0, float(min_sep) / pair_tol)

    return float(score)


def _build_family_pair_priority_table(
    reflections,
    seed_crystal_dirs,
    stats: ReflectionStatistics,
    pair_tol: float,
):
    family_dirs = {}
    family_best_weight = {}
    for reflection in reflections:
        family_id = _reflection_family_id(reflection)
        family_dirs.setdefault(family_id, seed_crystal_dirs[reflection.seed_index])
        family_best_weight[family_id] = max(
            family_best_weight.get(family_id, 0),
            int(getattr(reflection, 'weight', reflection.support)),
        )

    family_pair_priority = {}
    family_partner_rankings = {}
    epsilon = 1.0 / max(stats.sample_count, 1)
    family_ids = list(family_dirs)
    for family_a in family_ids:
        partner_scores = []
        for family_b in family_ids:
            if family_a == family_b:
                continue

            pair_key = _family_pair_key(family_a, family_b)
            covis = stats.seed_family_pair_visibility_prob.get(pair_key, 0.0)
            vis_a = stats.seed_family_visibility_prob.get(family_a, 0.0)
            vis_b = stats.seed_family_visibility_prob.get(family_b, 0.0)
            eta_a = stats.seed_family_eta_reliability.get(family_a, 0.0)
            eta_b = stats.seed_family_eta_reliability.get(family_b, 0.0)
            geometry = _geometric_family_pair_score(
                family_dirs[family_a],
                family_dirs[family_b],
                pair_tol,
            )
            score = (
                max(covis, epsilon)
                * max(vis_a, epsilon)
                * max(vis_b, epsilon)
                * max(eta_a, epsilon)
                * max(eta_b, epsilon)
                * geometry
            )
            family_pair_priority[(family_a, family_b)] = float(score)
            partner_scores.append(
                (
                    float(score),
                    family_best_weight.get(family_b, 0),
                    family_b,
                )
            )

        partner_scores.sort(reverse=True)
        family_partner_rankings[family_a] = [item[2] for item in partner_scores]

    return family_pair_priority, family_partner_rankings


def _interleaved_partner_order(
    candidate_indices,
    reflections,
    anchor_family_id,
    family_partner_rankings,
    max_partners,
):
    by_family = {}
    for idx in candidate_indices:
        family_id = _reflection_family_id(reflections[idx])
        if family_id == anchor_family_id:
            continue
        by_family.setdefault(family_id, []).append(idx)

    if not by_family:
        return []

    ranked_families = list(family_partner_rankings.get(anchor_family_id, ()))
    ranked_families.extend(
        family_id for family_id in by_family if family_id not in ranked_families
    )

    ordered = []
    while ranked_families and (max_partners <= 0 or len(ordered) < max_partners):
        progress = False
        for family_id in ranked_families:
            queue = by_family.get(family_id)
            if not queue:
                continue
            ordered.append(queue.pop(0))
            progress = True
            if max_partners > 0 and len(ordered) >= max_partners:
                break

        if not progress:
            break

    return ordered


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
        reflection_weight = int(getattr(reflection, 'weight', reflection.support))
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
            support_weight += reflection_weight
            seed_ids.add(reflections[idx].seed_index)
            hkl_ids.add(_reflection_family_id(reflections[idx]))
            if claim_tol > const.sqrt_epsf:
                proximity_score += reflection_weight * max(
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
    family_partner_rankings = params['family_partner_rankings']

    proposals = []
    pair_tests = 0
    start, stop = anchor_range
    for anchor_pos in range(start, stop):
        anchor_idx = order[anchor_pos]
        anchor = reflections[anchor_idx]
        anchor_family_id = _reflection_family_id(anchor)
        partner_order = _interleaved_partner_order(
            order[anchor_pos + 1 :],
            reflections,
            anchor_family_id,
            family_partner_rankings,
            max_partners,
        )
        if not partner_order:
            continue

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
    stats: ReflectionStatistics,
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
            -getattr(reflections[idx], 'weight', reflections[idx].support),
            -reflections[idx].support,
            -reflections[idx].intensity,
            reflections[idx].seed_index,
            reflections[idx].hkl_id,
        ),
    )
    claim_tol = pair_tol
    local_keep = min(max(4, max_candidates // max(len(order), 1) + 1), 8)
    family_pair_priority, family_partner_rankings = _build_family_pair_priority_table(
        reflections,
        seed_crystal_dirs,
        stats,
        pair_tol,
    )

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
        'family_pair_priority': family_pair_priority,
        'family_partner_rankings': family_partner_rankings,
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

    reflections, seed_crystal_dirs, params = _collect_pairwise_consensus_reflections(
        cfg,
        eta_ome,
    )
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
            stats,
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
                if (
                    idx != anchor_idx
                    and active_mask[idx]
                    and _reflection_family_id(reflections[idx])
                    != _reflection_family_id(anchor)
                )
            ],
            key=lambda idx: (
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
    desired_hkls = _resolved_orientation_map_hkl_ids(cfg, hkls=hkls)
    if clean:
        logger.info('clean option specified; recomputing eta/ome orientation maps')
        res = generate_eta_ome_maps(cfg, hkls=hkls)
    else:
        try:
            res = EtaOmeMaps(str(filename))
            pd = res.planeData
            if not np.array_equal(
                np.asarray(res.iHKLList, dtype=int),
                np.asarray(desired_hkls, dtype=int),
            ):
                logger.warning(
                    "loaded eta/ome maps do not match requested active hkls; recomputing maps"
                )
                res = generate_eta_ome_maps(cfg, hkls=hkls)
            else:
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
    allowed_hkl_ids = np.asarray(
        plane_data.getHKLID(plane_data.hkls, master=True),
        dtype=int,
    )
    active_hklIDs = np.asarray(
        _resolved_orientation_map_hkl_ids(cfg, hkls=hkls),
        dtype=int,
    )

    # catch duplicates
    assert len(np.unique(active_hklIDs)) == len(active_hklIDs), "duplicate hkls specified!"

    # catch excluded hkls
    excluded = np.array([hkl not in allowed_hkl_ids for hkl in active_hklIDs], dtype=bool)
    if np.any(excluded):
        raise RuntimeError(
            "The following requested hkls are marked as excluded: "
            + f"{active_hklIDs[excluded]}"
        )

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
