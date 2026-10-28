#!/usr/bin/env python3
"""Build an experimental detector-space catalog and audit master measurements.

This is a diagnostic, not a fitting backend. Reference pickles must be trusted
local outputs from benchmark_fit_grains.py. Nearest measured-centroid matching
is deliberately an optimistic test; it does not establish HKL assignments.
"""

import argparse
from dataclasses import astuple
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexrd.hedm.centroids import iter_centroids
from hexrd.hedm.experiment import HedmExperiment


def audit_measurements(catalog, measured, scale, radius=3.0):
    """Compare nearest catalog x/y/omega with measurements, using periodic omega."""
    if not len(measured) or not len(catalog):
        return dict(measurements=len(measured), matched=0, exactly_equal=0)
    catalog = np.asarray(catalog, dtype=float).copy()
    measured = np.asarray(measured, dtype=float).copy()
    catalog[:, 2] = (catalog[:, 2] + 180.0) % 360.0 - 180.0
    measured[:, 2] = (measured[:, 2] + 180.0) % 360.0 - 180.0
    extended = np.concatenate([catalog + [0, 0, offset] for offset in (-360, 0, 360)])
    distance, index = cKDTree(extended / scale).query(measured / scale)
    keep = distance <= radius
    difference = extended[index[keep]] - measured[keep]
    equal = np.all(difference == 0, axis=1)
    return dict(
        measurements=len(measured),
        matched=int(keep.sum()),
        exactly_equal=int(equal.sum()),
        median_abs_difference_xy_mm_omega_deg=(
            np.median(abs(difference), axis=0).tolist() if len(difference) else None
        ),
        max_abs_difference_xy_mm_omega_deg=(
            np.max(abs(difference), axis=0).tolist() if len(difference) else None
        ),
        match_radius_in_pixel_frame_units=radius,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--study', type=int, default=None)
    args = p.parse_args()
    experiment = HedmExperiment(str(args.config.resolve()), study=args.study)
    with args.reference.open('rb') as stream:
        reference_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        _, spots = pickle.load(stream)
    report = dict(
        experimental=True,
        enabled_for_fitting=False,
        reference_sha256=reference_sha256,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        matching='Nearest master measurement, diagnostic only; not HKL assignment',
        panels={},
    )
    arrays = {}
    for name, series in experiment.image_series.items():
        started = time.perf_counter()
        peaks = list(
            iter_centroids(
                (series[i] for i in range(len(series))),
                series.omega,
                experiment.fit_grains.threshold,
            )
        )
        elapsed = time.perf_counter() - started
        # Preserve compact numeric arrays; no Python objects in the NPZ archive.
        rows = np.array(
            [astuple(peak)[:6] + peak.bounds for peak in peaks], dtype=float
        )
        rows = rows.reshape(-1, 12)
        arrays[name] = rows
        panel = experiment.instrument.detectors[name]
        xy = panel.pixelToCart(rows[:, :2])
        catalog = np.column_stack((xy, rows[:, 2]))
        measured = []
        for signals, grain_panels in spots.values():
            for record in grain_panels.get(name, []):
                if record[0] >= 0:
                    measured.append([*record[7], np.degrees(record[6][2])])
        measured = np.asarray(measured).reshape(-1, 3)
        delta = np.median(abs(np.diff(series.omega, axis=1)))
        audit = audit_measurements(
            catalog, measured, [panel.pixel_size_col, panel.pixel_size_row, delta]
        )
        report['panels'][name] = dict(
            peaks=len(peaks),
            preprocessing_seconds=elapsed,
            catalog_bytes=rows.nbytes,
            **audit,
        )
        print(name, report['panels'][name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    report['passes_exact_measurement_gate'] = all(
        item['exactly_equal'] == item['measurements'] and item['measurements'] > 0
        for item in report['panels'].values()
    )
    report['fit_equivalence_established'] = False
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
