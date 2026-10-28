#!/usr/bin/env python3
"""Measure a centroid-only replay of an unchanged fit, with an exact-result gate.

Research only: recording runs the full workflow, including intermediate fits.
This does NOT make the first run image-independent. A reusable production
cache would additionally need complete data/material/geometry provenance.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexrd.hedm.experiment import HedmExperiment
from hexrd.hedm.fitgrains import fit_grains
from scripts.benchmark_fit_grains import assert_identical


class RequestReplay:
    """In-memory trace bound to an unchanged experiment; fail on new requests."""

    def __init__(self, original):
        self.original = original
        self.records = []
        self.position = 0
        self.scope = None
        self.replaying = False

    def __call__(self, plane_data, grain_params, images, **options):
        if options.get('filename') is not None:
            raise ValueError('Replay exploration requires spot writing disabled')
        scope = (id(plane_data), id(images))
        if self.scope is None:
            self.scope = scope
        if scope != self.scope:
            raise ValueError('Centroid replay experiment scope changed')
        key = pickle.dumps((np.asarray(grain_params), options), protocol=5)
        if not self.replaying:
            result = self.original(plane_data, grain_params, images, **options)
            self.records.append((key, copy.deepcopy(result)))
            return result
        if self.position >= len(self.records):
            raise ValueError('Unrecorded centroid request')
        expected, result = self.records[self.position]
        if key != expected:
            raise ValueError('Centroid request changed: refusing approximate reuse')
        self.position += 1
        return copy.deepcopy(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--grains', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=10)
    args = p.parse_args()
    if args.limit < 1:
        p.error('--limit must be positive')
    experiment = HedmExperiment(str(args.config.resolve()))
    experiment.max_workers = 1
    grains = np.loadtxt(args.grains, ndmin=2)[: args.limit]
    instrument = experiment.instrument
    material = experiment.get_active_material()
    original = instrument.pull_spots
    trace = RequestReplay(original)
    instrument.pull_spots = trace
    times = []
    try:
        start = time.perf_counter()
        recorded = fit_grains(
            experiment,
            grains.copy(),
            material,
            write_spots_files=False,
            return_pull_spots_data=True,
        )
        times.append(time.perf_counter() - start)
        trace.replaying = True
        start = time.perf_counter()
        replayed = fit_grains(
            experiment,
            grains.copy(),
            material,
            write_spots_files=False,
            return_pull_spots_data=True,
        )
        times.append(time.perf_counter() - start)
        assert trace.position == len(trace.records), 'Unconsumed centroid requests'
    finally:
        instrument.pull_spots = original
    assert_identical(recorded, replayed)
    with args.reference.open('rb') as f:
        reference_sha256 = hashlib.file_digest(f, 'sha256').hexdigest()
        f.seek(0)
        reference = pickle.load(f)
    ids = set(int(row[0]) for row in grains)
    reference = (
        [r for r in reference[0] if r[0] in ids],
        {k: v for k, v in reference[1].items() if k in ids},
    )
    assert_identical(reference, replayed)
    report = dict(
        experimental=True,
        enabled_for_fitting=False,
        grains=len(grains),
        extraction_requests=len(trace.records),
        replay_image_extractions=0,
        recording_seconds=times[0],
        replay_seconds=times[1],
        identical_fits_and_spots_to_master=True,
        reference_sha256=reference_sha256,
        scope='Unchanged experiment and identical requests in the same process only',
        limitation='Recording requires the full extraction/fit sequence. Not a first-run speedup.',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
