#!/usr/bin/env python3
import argparse
import copy
import csv
import logging
import os
import sys
import tempfile
import timeit
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault('ACK_DEPRECATED', 'true')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba-cache')
warnings.filterwarnings('ignore', message='omega period specification is deprecated')
logging.disable(logging.CRITICAL)

from hexrd.core import matrixutil as mutil
from hexrd.core import rotations as rot
from hexrd.core.valunits import valWUnit
from hexrd.hedm import config, indexer
from hexrd.hedm.findorientations import (
    _collect_seed_reflections,
    find_orientations,
    generate_orientation_candidates,
)
from hexrd.hedm.xrdutil.utils import EtaOmeMaps, simulateOmeEtaMaps


MAX_RAW_PAIRS = None


@dataclass
class SyntheticEtaOmeMaps:
    dataStore: np.ndarray
    planeData: object
    iHKLList: np.ndarray
    etaEdges: np.ndarray
    omeEdges: np.ndarray
    etas: np.ndarray
    omegas: np.ndarray

    def save(self, filename: str | Path) -> None:
        args = np.array(self.planeData.getParams(), dtype=object)[:4]
        args[2] = valWUnit('wavelength', 'length', args[2], 'angstrom')
        hkls = np.vstack([i['hkl'] for i in self.planeData.hklDataList]).T
        save_dict = {
            'dataStore': self.dataStore,
            'etas': self.etas,
            'etaEdges': self.etaEdges,
            'iHKLList': self.iHKLList,
            'omegas': self.omegas,
            'omeEdges': self.omeEdges,
            'planeData_args': args,
            'planeData_hkls': hkls,
            'planeData_excl': self.planeData.exclusions,
        }
        np.savez_compressed(filename, **save_dict)


def subsample_edges(edges: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return np.asarray(edges, dtype=float)

    reduced = np.asarray(edges[::stride], dtype=float)
    if reduced[-1] != edges[-1]:
        reduced = np.r_[reduced, edges[-1]]
    return reduced


def subsample_template_maps(template: EtaOmeMaps, omega_stride: int, eta_stride: int):
    if omega_stride <= 1 and eta_stride <= 1:
        return template

    data = np.asarray(template.dataStore)[:, ::omega_stride, ::eta_stride]
    ome_edges = subsample_edges(np.asarray(template.omeEdges), omega_stride)
    eta_edges = subsample_edges(np.asarray(template.etaEdges), eta_stride)
    omegas = 0.5 * (ome_edges[:-1] + ome_edges[1:])
    etas = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    return SyntheticEtaOmeMaps(
        dataStore=data,
        planeData=copy.deepcopy(template.planeData),
        iHKLList=np.asarray(template.iHKLList, dtype=int),
        etaEdges=eta_edges,
        omeEdges=ome_edges,
        etas=etas,
        omegas=omegas,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark find-orientations scaling on synthetic eta-omega maps."
    )
    parser.add_argument('--config', required=True, help='Path to HEXRD config file')
    parser.add_argument(
        '--template-maps',
        required=True,
        help='Existing eta-omega maps file used as a template for edges and NaN mask',
    )
    parser.add_argument(
        '--counts',
        nargs='+',
        type=int,
        default=[25, 50, 100, 300, 600, 1000],
        help='Grain counts to simulate',
    )
    parser.add_argument(
        '--generators',
        nargs='+',
        default=['discrete-fibers', 'pairwise', 'pairwise-greedy'],
        choices=['discrete-fibers', 'pairwise', 'pairwise-greedy'],
        help='Candidate generators to benchmark',
    )
    parser.add_argument('--repeats', type=int, default=1, help='Runs per grain count')
    parser.add_argument('--seed', type=int, default=0, help='Base RNG seed')
    parser.add_argument(
        '--peak-scale',
        type=float,
        default=10.0,
        help='Baseline intensity contribution per grain reflection',
    )
    parser.add_argument(
        '--intensity-sigma',
        type=float,
        default=0.35,
        help='Lognormal sigma for per-grain intensity variation',
    )
    parser.add_argument(
        '--noise-scale',
        type=float,
        default=0.0,
        help='Standard deviation of additive white Gaussian background noise',
    )
    parser.add_argument(
        '--dilation-probability',
        type=float,
        default=0.0,
        help='Probability that a grain receives binary dilation',
    )
    parser.add_argument(
        '--max-dilation',
        type=int,
        default=0,
        help='Maximum binary dilation iterations applied per grain',
    )
    parser.add_argument(
        '--multiprocessing',
        type=int,
        default=1,
        help='Number of CPUs to use during indexing',
    )
    parser.add_argument(
        '--omega-stride',
        type=int,
        default=1,
        help='Subsample factor for omega bins in the template map',
    )
    parser.add_argument(
        '--eta-stride',
        type=int,
        default=1,
        help='Subsample factor for eta bins in the template map',
    )
    parser.add_argument(
        '--truth-tolerance',
        type=float,
        default=1.0,
        help='Misorientation tolerance in degrees for truth matching',
    )
    parser.add_argument(
        '--max-raw-pairs',
        type=int,
        default=500000,
        help='Skip exhaustive pairwise when estimated seed-reflection pairs exceed this limit',
    )
    parser.add_argument(
        '--friedel-pairing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable Friedel reduction during seeded search',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/tmp/find_orientations_scaling.csv'),
        help='CSV output path',
    )
    return parser.parse_args()


def random_exp_maps(num_grains: int, rng: np.random.Generator) -> np.ndarray:
    rand_q = mutil.unitVector(rng.standard_normal((4, num_grains)))
    rand_e = np.tile(2.0 * np.arccos(rand_q[0, :]), (3, 1)) * mutil.unitVector(
        rand_q[1:, :]
    )
    return np.asarray(rand_e, dtype=float)


def restrict_plane_data_to_active_hkls(template: EtaOmeMaps):
    plane_data = copy.deepcopy(template.planeData)
    active_hkls = np.asarray(template.iHKLList, dtype=int)
    exclusions = np.ones(plane_data.getNhklRef(), dtype=bool)
    exclusions[active_hkls] = False
    plane_data.exclusions = exclusions
    return plane_data


def simulate_synthetic_eta_ome_maps(
    template: EtaOmeMaps,
    exp_maps: np.ndarray,
    chi: float,
    rng: np.random.Generator,
    peak_scale: float,
    intensity_sigma: float,
    noise_scale: float,
    dilation_probability: float,
    max_dilation: int,
) -> SyntheticEtaOmeMaps:
    plane_data = restrict_plane_data_to_active_hkls(template)
    template_data = np.asarray(template.dataStore, dtype=float)
    nan_mask = np.isnan(template_data)
    data = np.zeros_like(template_data, dtype=float)

    ome_edges_deg = np.degrees(template.omeEdges)
    eta_edges_deg = np.degrees(template.etaEdges)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for grain_idx in range(exp_maps.shape[1]):
            grain_map = simulateOmeEtaMaps(
                ome_edges_deg,
                eta_edges_deg,
                plane_data,
                exp_maps[:, grain_idx].reshape(3, 1),
                chi=chi,
            ).astype(float)

            if max_dilation > 0 and rng.random() < dilation_probability:
                radius = int(rng.integers(1, max_dilation + 1))
                grain_map = ndimage.binary_dilation(
                    grain_map > 0,
                    iterations=radius,
                ).astype(float)

            intensity = peak_scale
            if intensity_sigma > 0:
                intensity *= float(rng.lognormal(mean=0.0, sigma=intensity_sigma))

            data += intensity * grain_map

    valid_mask = ~nan_mask
    if noise_scale > 0:
        noise = noise_scale * rng.standard_normal(data.shape)
        data[valid_mask] += noise[valid_mask]
        data[valid_mask] = np.clip(data[valid_mask], 0.0, None)

    data[nan_mask] = np.nan
    return SyntheticEtaOmeMaps(
        dataStore=data,
        planeData=plane_data,
        iHKLList=np.asarray(template.iHKLList, dtype=int),
        etaEdges=np.asarray(template.etaEdges, dtype=float),
        omeEdges=np.asarray(template.omeEdges, dtype=float),
        etas=np.asarray(template.etas, dtype=float),
        omegas=np.asarray(template.omegas, dtype=float),
    )


def load_config(config_path: Path):
    cfg = config.open(config_path)[0]
    cfg.working_dir = str(config_path.parent)
    return cfg


def apply_run_settings(cfg, maps_path: Path, generator: str, multiprocessing: int, friedel_pairing: bool):
    cfg._cfg['multiprocessing'] = multiprocessing
    cfg._cfg['find_orientations']['orientation_maps']['file'] = str(maps_path)
    cfg._cfg['find_orientations'].get('omega', {}).pop('period', None)
    seed_cfg = cfg._cfg['find_orientations'].setdefault('seed_search', {})
    seed_cfg['candidate_generator'] = generator
    seed_cfg['friedel_pairing'] = friedel_pairing


def match_found_orientations(found_qbar, exp_maps, qsym, tol_deg):
    if found_qbar.size == 0:
        return 0, np.nan

    true_quats = rot.quatOfExpMap(exp_maps)
    if true_quats.ndim == 1:
        true_quats = true_quats.reshape(4, 1)

    cost = np.zeros((found_qbar.shape[1], true_quats.shape[1]), dtype=float)
    for i in range(found_qbar.shape[1]):
        cost[i, :] = rot.misorientation(
            found_qbar[:, i].reshape(4, 1),
            true_quats,
            (qsym,),
        )[0]

    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = cost[row_ind, col_ind]
    matched = int(np.count_nonzero(assigned <= np.radians(tol_deg)))
    mean_deg = float(np.degrees(np.mean(assigned))) if assigned.size else np.nan
    return matched, mean_deg


def benchmark_generator(cfg_path: Path, maps_path: Path, generator: str, multiprocessing: int, friedel_pairing: bool, exp_maps: np.ndarray, truth_tol_deg: float):
    cfg = load_config(cfg_path)
    apply_run_settings(cfg, maps_path, generator, multiprocessing, friedel_pairing)

    eta_ome = EtaOmeMaps(str(maps_path))
    estimated_seed_reflections = np.nan
    estimated_pair_count = np.nan
    if generator == 'pairwise':
        reflections, _, _ = _collect_seed_reflections(cfg, eta_ome)
        estimated_seed_reflections = int(len(reflections))
        estimated_pair_count = int(len(reflections) * (len(reflections) - 1) // 2)
        if (
            MAX_RAW_PAIRS is not None
            and estimated_pair_count > MAX_RAW_PAIRS
        ):
            return {
                'estimated_seed_reflections': estimated_seed_reflections,
                'estimated_pair_count': estimated_pair_count,
                'candidate_time_s': 0.0,
                'num_candidates': 0,
                'score_time_s': 0.0,
                'max_completeness': np.nan,
                'num_above_threshold': 0,
                'full_total_time_s': np.nan,
                'num_grains': 0,
                'matched_grains': 0,
                'mean_truth_misorientation_deg': np.nan,
                'error': (
                    f'skipped: estimated raw pairs {estimated_pair_count} '
                    f'exceed limit {MAX_RAW_PAIRS}'
                ),
            }

    start = timeit.default_timer()
    qfib = generate_orientation_candidates(cfg, eta_ome)
    candidate_time = timeit.default_timer() - start

    if qfib.size == 0:
        return {
            'estimated_seed_reflections': estimated_seed_reflections,
            'estimated_pair_count': estimated_pair_count,
            'candidate_time_s': candidate_time,
            'num_candidates': 0,
            'score_time_s': 0.0,
            'max_completeness': np.nan,
            'num_above_threshold': 0,
            'full_total_time_s': np.nan,
            'num_grains': 0,
            'matched_grains': 0,
            'mean_truth_misorientation_deg': np.nan,
            'error': 'no candidates',
        }

    start = timeit.default_timer()
    completeness = np.array(
        indexer.paintGrid(
            qfib,
            eta_ome,
            etaRange=np.radians(cfg.find_orientations.eta.range),
            omeTol=np.radians(cfg.find_orientations.omega.tolerance),
            etaTol=np.radians(cfg.find_orientations.eta.tolerance),
            omePeriod=np.radians(cfg.find_orientations.omega.period),
            threshold=cfg.find_orientations.threshold,
            doMultiProc=multiprocessing > 1,
            nCPUs=multiprocessing,
        )
    )
    score_time = timeit.default_timer() - start

    try:
        start = timeit.default_timer()
        results = find_orientations(cfg)
        full_total_time = timeit.default_timer() - start
        qbar = results['qbar']
        matched_grains, mean_misorientation = match_found_orientations(
            qbar,
            exp_maps,
            eta_ome.planeData.q_sym,
            truth_tol_deg,
        )
        error = ''
    except Exception as exc:  # pragma: no cover - benchmark reporting path
        full_total_time = np.nan
        qbar = np.empty((4, 0))
        matched_grains = 0
        mean_misorientation = np.nan
        error = str(exc)

    return {
        'estimated_seed_reflections': estimated_seed_reflections,
        'estimated_pair_count': estimated_pair_count,
        'candidate_time_s': candidate_time,
        'num_candidates': int(qfib.shape[1]),
        'score_time_s': score_time,
        'max_completeness': float(np.max(completeness)),
        'num_above_threshold': int(
            np.count_nonzero(completeness >= cfg.find_orientations.clustering.completeness)
        ),
        'full_total_time_s': full_total_time,
        'num_grains': int(qbar.shape[1]),
        'matched_grains': matched_grains,
        'mean_truth_misorientation_deg': mean_misorientation,
        'error': error,
    }


def main():
    global MAX_RAW_PAIRS
    args = parse_args()
    MAX_RAW_PAIRS = args.max_raw_pairs
    cfg_path = Path(args.config).resolve()
    template_maps_path = Path(args.template_maps).resolve()
    template = subsample_template_maps(
        EtaOmeMaps(str(template_maps_path)),
        args.omega_stride,
        args.eta_stride,
    )
    base_cfg = load_config(cfg_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    csv_fp = args.output.open('w', newline='')
    writer = None
    with tempfile.TemporaryDirectory(prefix='hexrd-synth-eta-ome-') as tmpdir:
        tmpdir_path = Path(tmpdir)
        for repeat in range(args.repeats):
            for grain_count in args.counts:
                rng = np.random.default_rng(args.seed + 1000 * repeat + grain_count)
                exp_maps = random_exp_maps(grain_count, rng)
                synthetic_maps = simulate_synthetic_eta_ome_maps(
                    template,
                    exp_maps,
                    base_cfg.instrument.hedm.chi,
                    rng,
                    peak_scale=args.peak_scale,
                    intensity_sigma=args.intensity_sigma,
                    noise_scale=args.noise_scale,
                    dilation_probability=args.dilation_probability,
                    max_dilation=args.max_dilation,
                )
                maps_path = tmpdir_path / f'synthetic_eta_ome_{grain_count}_{repeat}.npz'
                synthetic_maps.save(maps_path)

                for generator in args.generators:
                    result = benchmark_generator(
                        cfg_path,
                        maps_path,
                        generator,
                        args.multiprocessing,
                        args.friedel_pairing,
                        exp_maps,
                        args.truth_tolerance,
                    )
                    row = {
                        'grain_count': grain_count,
                        'repeat': repeat,
                        'generator': generator,
                        'friedel_pairing': args.friedel_pairing,
                        'peak_scale': args.peak_scale,
                        'intensity_sigma': args.intensity_sigma,
                        'noise_scale': args.noise_scale,
                        'dilation_probability': args.dilation_probability,
                        'max_dilation': args.max_dilation,
                        'omega_stride': args.omega_stride,
                        'eta_stride': args.eta_stride,
                    }
                    row.update(result)
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(csv_fp, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    csv_fp.flush()
                    print(row, flush=True)

    csv_fp.close()

    print(f'wrote {len(rows)} benchmark rows to {args.output}')


if __name__ == '__main__':
    main()
