# Fit-grains dashboard and centroid experiments

October 28, 2026. Reference: local master
`ca51e7d5d278bafb6a007dd284d9867714b19465`, built with its own native extensions.

## Running the dashboard

```bash
hexrd fit-grains configuration.yml
```

Interactive terminals now open the same Textual presentation used by
find-orientations, with a fitting-specific view:

- Configuration, material, shared image loading, grain fitting, and result writing.
- Grain completion count, throughput/ETA, elapsed time, and process-tree RAM.
- The latest 32 grain updates: extraction, fitting, refitting, tolerance pass,
  completeness, and chi-square. Insufficient reflections are shown explicitly.
- Stage inspection, an event log, cancellation, and a persistent final summary.

Use `f` to follow the active stage, `l` to toggle the log, space to pause log
scrolling, and `q` to cancel. After work finishes, `q` closes the dashboard.
Cancellation terminates and joins fitting workers before completion is shown.
The data table has more room than the log; hiding the log provides additional
space on narrow terminals.

`--no-tui` uses plain logs and exits automatically, `--quiet` suppresses terminal
progress, and redirected/noninteractive runs use the batch path. Existing
Python callers do not launch an application; the CLI owns the TUI. If fitting
needs to run find-orientations first, its stages appear in the same dashboard.

Both views use `WorkflowDashboard` and the same progress/context machinery.
Bounded worker messages carry presentation data only; the parent owns UI state.
Intermediate progress messages may be dropped when the queue is full, rather
than blocking scientific workers. Final counts are reconciled from results.
Result ordering and the numerical fitting/extraction code are preserved.
Workers detach inherited dashboard handlers and contexts after forking.

The workflow runs in an explicitly owned thread with a copied application
context. This also resolves the previously reproduced asyncio executor shutdown
hang: the former dashboard shutdown tests now pass, along with the new fitting
UI tests. Configuration failures restore logging handlers and progress context.

## Exact-master acceptance

With dashboard reporting enabled, all **1,690 Inconel fits and all returned
spot data agree exactly with master**, including array shapes/dtypes, ordering,
and matching NaNs. No numerical tolerance is used. A real Textual run of the
CLI fitted three Inconel grains, wrote results/spot files, and matched master's
fit results exactly. The nine example runs also match master for fits and spots:
six single-GE ruby series, composite Dexelas, ROI Dexelas, and alpha-zirconium.

The regression selection passes **492 tests**, including the previous dashboard
shutdown tests, fit UI completion/cancellation, worker cleanup, and the existing
scientific/integration tests. Its single warning is the expected simulated HDF5
close failure. Machine-readable checks are in
[fit-grains-dashboard-results.json](fit-grains-dashboard-results.json).

## What the centroid experiment establishes

The experimental streaming implementation is in `hexrd/hedm/centroids.py`.
It thresholds sparse or dense frames, labels eight-connected components in each
frame, and joins neighboring foreground pixels across contiguous omega frames
to obtain 26-connected components. It retains sparse pixel coordinates from
adjacent frames and statistics for active components; it does not allocate a
dense detector-by-rotation volume. Components can branch and merge. Gaps or
reversals in omega terminate them.

The resulting records contain detector row/column and physical omega centroids,
integrated/max intensity, pixel count, and bounds. They do not yet implement
background estimation, touching-peak splitting, uncertainty estimates, or joining
the first and last frames of a periodic scan. The audit uses an unmasked detector
catalog; the builder accepts an explicit usable-pixel mask. These are research
tools and are **not enabled as a fit-grains backend**.

On Inconel, preprocessing took **6.69 seconds** and produced **357,527 records**
occupying **34.3 MB** of numeric fields before compression. The audit compares
against all **467,819 valid final-pass master spot records**, matching the nearest
catalog position within a radius of three in pixel/frame-scaled X/Y/omega space.
It matches against master's *measured* positions, so this is an optimistic
measurement diagnostic, not independent reflection/HKL assignment.

| Measurement audit | Result |
| --- | ---: |
| Master measurements with a nearby catalog entry | 465,393 |
| Exactly equal X/Y/omega triples among those matches | **0** |
| Median absolute X difference | 0.002444 mm |
| Median absolute Y difference | 0.002612 mm |
| Median absolute omega difference | 0.000123 degrees |

This fails the exact-measurement gate. It does not establish equivalence of the
resulting grain fits. No tolerance was substituted to accept the prototype.
Matching an entry does not prove correct peak identity or resolve shared/merged
peaks; the same component can be near multiple reference measurements.

The reason is structural: master's measurement is a function of the raw pixels,
grain prediction, angular resampling grid, window, threshold, and geometry.
Its connected components are formed inside that resampled window. The selected
component, repeated samples of detector pixels, clipped support, and floating
point centroid calculation can change as the prediction/window changes. A single
detector-space centroid discards information needed to reproduce those queries.
Even a well-isolated physical peak need not give an identical measurement.

## An exact centroid-only replay

A second experiment records the complete reflection measurements for each
extraction request, including every tolerance pass, then repeats fitting from
the same initial grains using those measurements. The replay refuses a request
if its grain parameters or extraction options differ, even by one floating point
step, and checks the experiment object identities. Spot writing is disabled.

For **10 Inconel grains**, all **30 extraction requests** were replayed without
calling the image extractor. Both the fitted results and returned spots match
master exactly. Recording took **1.63 seconds**; the replay took **0.30 seconds**
in this individual serial run. This demonstrates that fitting can consume
precomputed measurements exactly when the measurement requests are identical.
**Recording includes the original extraction and intermediate fits: this is not
a first-run speedup or a general centroid catalog.** The trace is an in-memory
experiment scoped to an unchanged process, not a validated persistent cache.

A production implementation meeting the strict requirement has two viable
starting points:

1. A request-keyed measurement cache. Persist all extraction inputs and complete
   image/material/calibration provenance; reuse records only for identical
   requests. Validate every pass and final result against master. Changed
   predictions, tolerances, thresholds, processing, or geometry invalidate it.
2. A component catalog retaining pixel support as well as summary statistics.
   Use it to locate candidate signal cheaply, then reproduce the original
   resampling, window selection, and arithmetic whenever an equivalent
   measurement cannot be proved. A global centroid alone is insufficient.

The first can accelerate repeat fits; the second may improve first-run work but
still needs exact window-dependent evaluation. A truly independent centroid-only
measurement model would require a separately agreed scientific acceptance rule;
it cannot presently claim exact equivalence to master.

## Reproduce

Build master independently, then use the benchmark tool from this checkout:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

env/bin/python scripts/benchmark_fit_grains.py run \
  --source /path/to/master --config "$PWD/inconel/fitting_config.yml" \
  --grains "$PWD/inconel/inconel_test/grains.out" --workers 10 \
  --output /tmp/fit-master

env/bin/python scripts/benchmark_fit_grains.py examples \
  --reference-source /path/to/master --source "$PWD" \
  --examples "$PWD/examples" --output-dir /tmp/fit-examples

env/bin/python scripts/explore_fit_grains_centroids.py \
  --config "$PWD/inconel/fitting_config.yml" \
  --reference /tmp/fit-master.pkl --output /tmp/centroid-catalog

env/bin/python scripts/explore_fit_grains_centroid_replay.py \
  --config "$PWD/inconel/fitting_config.yml" \
  --grains "$PWD/inconel/inconel_test/grains.out" \
  --reference /tmp/fit-master.pkl --limit 10 --output /tmp/centroid-replay.json
```

Reference pickles must be trusted local benchmark outputs. The catalog audit
writes numeric NPZ records and a JSON report; the replay exits with an error if
exact-master comparison fails. Measurements are individual runs on a shared
machine. Artifacts for this investigation, including live/completed TUI SVG
screenshots, are in `/tmp/hexrd-fit-dashboard`.

```bash
HEXRD_EXAMPLE_REPO_PATH="$PWD/examples" env/bin/python -m pytest \
  tests/test_npz.py tests/test_experiment.py tests/test_fit-grains.py \
  tests/test_benchmark_fit_grains.py tests/test_fit_grains_buffers.py \
  tests/test_centroids.py tests/test_fit_grains_dashboard.py tests/test_dashboard.py \
  tests/core/imageseries tests/core/instrument \
  tests/core/fitting/test_fitting_grains.py tests/test_find_orientations.py -q
```
