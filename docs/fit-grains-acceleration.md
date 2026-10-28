# Further fit-grains acceleration

Measured October 28, 2026. The starting point is a frozen copy of the working
branch immediately before this investigation, including the earlier native
extraction, memory improvements, and Textual dashboard. The numerical reference
is separately built master, `ca51e7d5d278bafb6a007dd284d9867714b19465`.

## Results

The full Inconel workload uses all **1,690 input grains**, the configured three
tolerance passes and refit, and **10 workers**. No solver settings, finite
difference steps, thresholds, tolerances, or reflection-selection rules change.

| Workload | Starting implementation | Current | Speedup |
| --- | ---: | ---: | ---: |
| Fit with the usual text spot output | 41.72 s | **7.38 s** | **5.66×** |
| Fit without spot-file output | 30.57 s | **5.94 s** | **5.14×** |
| Peak aggregate PSS, with spot output | 0.711 GB | **0.625 GB** | **12.2% less** |

Times are medians of three runs in one process for each implementation. Raw
seconds, in run order:

- With files, before: 36.9444, 41.7244, 42.6960; after: 7.5492, 7.3769, 7.3064.
- Without files, before: 27.7497, 30.5729, 32.7524; after: 6.1857, 5.7830, 5.9443.

The first measured run includes lazy image loading; subsequent runs reuse loaded
images. Comparing first runs gives **4.89× with files** and **4.49× without**.
The headline numbers describe the repeated-run median, not a guaranteed speedup
for every invocation or dataset. These are individual measurements on a shared
development machine.

Timing covers `fit_grains()`, including material/image loading, workers, IPC,
refinement, and requested spot output. It excludes package imports, experiment
configuration construction, benchmark-pickle serialization, and the CLI's final
grain-table export. Both trees use Python 3.12.7, NumPy 2.3.5, SciPy 1.17.1,
`OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

Memory is sampled every 0.1 seconds across the parent and descendants, covering
experiment setup and all repeats. GB means decimal gigabytes. PSS apportions
shared pages; summed RSS double-counts them. Peak USS with files is essentially
unchanged, 0.454 GB in both implementations. The
[machine-readable report](fit-grains-acceleration-results.json) retains the full
measurements, source/input hashes, and validation records.

## Implementation

### Exact sparse peak extraction

A profile of the first 100 Inconel grains found about 551 million sampled
voxels, of which only 10.6 million exceeded the threshold. The new integer path
inverts the original spatial sampling map once per patch, then visits foreground
source pixels in each omega frame. It coalesces duplicate CSR entries before
thresholding and preserves repeated samples, clipping, ROI offsets, frame order,
and intensity dtypes.

Sorted foreground samples retain their original polar voxel coordinates.
Connected foreground runs replace per-voxel neighbor checks. Components retain
26-neighbor connectivity, first-occurrence ordering, and nearest-component tie
behavior. Integer moments are used only within conservative bounds that make
all required floating point sums exact. Local-moment division followed by the
bounding-box offset preserves the original centroid rounding. Floating images,
negative thresholds, oversized mappings, and unsafe moments retain the dense
reference-order path.

### Native geometry and objective evaluation

Angular samples are separable by axis. Native grid generation and pixel
classification avoid per-patch Python meshes and repeated NumPy dispatch.
For flat panels with an axial beam, a guarded separable projection computes
pixel indices directly; AVX2 is selected at runtime where supported, with a
portable scalar path. Tilted/general-beam geometry uses the general projection.
Boundary-sensitive patches use the original transform chain. Continuous measured
coordinates still use the reference scalar calculation.

The fit-local native objective preserves NumPy's three BLAS matrix products,
normalization order, omega-branch selection, and projection arithmetic. It caches
fixed HKL products and omits unobserved tth/eta calculations when solving omegas.
A bounded four-entry geometry cache uses exact parameter/wavelength keys to reuse
geometry for finite-difference steps that change only grain position. Energy
corrections remain part of the key. Unsupported dtypes, explicit omega periods,
simulation output, and infeasible cases retain the Python path.

Measured omega mapping and XY evaluation are batched once per detector/pass.
Smaller multiprocessing chunks improve load balance and bound retained IPC data.
All result ordering remains unchanged. The Textual dashboard continues to show
per-grain extraction, fitting, refit, and completion updates.

### Spot output

Spot table format strings are constructed once. Packed records are formatted in
batches with the same fields, precision, spacing, missing values, and headers.

Fitting now publishes each grain's last complete extraction once, instead of
writing and overwriting the same file during every tolerance pass. Early exits
and fitting errors also flush the last complete extraction. The stored records
still precede the last fit/refit, exactly as in master. Files consequently become
available later during processing, while completed-file names and contents remain
identical. Direct calls to `pull_spots(filename=...)` still write immediately.

## Acceptance checks

- All **1,690 Inconel fits and returned spot data match master exactly**, with
  no numerical tolerance, including dtype/shape, ordering, and matching NaNs.
  The full validation run also checks all 1,690 dashboard completion updates.
- All **nine examples** match master: six single-GE ruby series, composite
  Dexelas, ROI Dexelas, and alpha-zirconium with discontinuous omega wedges.
- All **1,690 text spot files** are byte-identical to the frozen starting
  implementation. Five Inconel files and all **24 ROI files** additionally
  match master byte for byte.
- A real CLI workflow in Textual fits three grains, writes results, and matches
  master's fit results exactly.
- **537 regression tests pass**, including native objective tests with reversed
  detector dictionaries, tilted beams, energy gradients, distortion, partial
  fitting flags, SIMD tails, and position-only geometry reuse; sparse/run
  connectivity; ROI/duplicate pixels; output formatting; and cleanup on errors
  and cancellation. The single warning is the intentional HDF5-close failure
  test.

Measured centroids retain the original grain- and window-dependent definition.
The independent global centroid catalog remains experimental.

## Reproduce

Build the updated native extension before running this checkout. Use independent
source trees and native binaries for the starting implementation and master.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

env/bin/python scripts/benchmark_fit_grains.py run \
  --source /path/to/starting-tree --config "$PWD/inconel/fitting_config.yml" \
  --grains "$PWD/inconel/inconel_test/grains.out" --workers 10 \
  --write-spots --no-return-spots --memory --repeat 3 --output /tmp/fit-before

env/bin/python scripts/benchmark_fit_grains.py run \
  --source "$PWD" --config "$PWD/inconel/fitting_config.yml" \
  --grains "$PWD/inconel/inconel_test/grains.out" --workers 10 \
  --write-spots --no-return-spots --memory --repeat 3 --output /tmp/fit-after

env/bin/python scripts/benchmark_fit_grains.py compare \
  /tmp/fit-before.pkl /tmp/fit-after.pkl
```

Omit `--no-return-spots` for the complete fit/spot comparison against master.
Use `examples --reference-source /path/to/master --source "$PWD" --examples
"$PWD/examples" --output-dir /tmp/fit-examples` to compare the nine examples.
The benchmark isolates analysis output from the input datasets. Trusted local
pickles and raw logs from this investigation are in `/tmp/hexrd-fit-stage3`.

```bash
HEXRD_EXAMPLE_REPO_PATH="$PWD/examples" env/bin/python -m pytest \
  tests/test_npz.py tests/test_experiment.py tests/test_fit-grains.py \
  tests/test_benchmark_fit_grains.py tests/test_fit_grains_buffers.py \
  tests/test_centroids.py tests/test_fit_grains_dashboard.py tests/test_dashboard.py \
  tests/core/imageseries tests/core/instrument \
  tests/core/fitting/test_fitting_grains.py \
  tests/core/fitting/test_native_grain_objective.py tests/test_find_orientations.py -q
```
