# Fit-grains performance investigation

Measured on October 27, 2026, using the local `inconel` and `examples`
directories. The reference is local master
`ca51e7d5d278bafb6a007dd284d9867714b19465`; the starting branch is
`925967d2976c928de8fe9404aa4569c2936b0d2b`.

## Results

The full Inconel workload uses all **1,690 grains**, three tolerance passes,
refitting, and **10 workers**. Returning every extracted spot for exact
validation takes **30.00 seconds**, compared with **257.09 seconds on master**:
**8.57× faster**. The first implementation took 127.67 seconds under the same
validation settings.

For normal fitting, without retaining the returned spot tables, the new code
is **4.63× faster than the first implementation**, with **34% less peak PSS**.
The ROI example uses **54% less peak PSS**. PSS apportions shared pages across
processes; USS counts private pages. Summed RSS counts shared pages repeatedly
and is a less useful measure of this multiprocessing workload.

| Normal fitting workload | First implementation | Current | Improvement |
| --- | ---: | ---: | ---: |
| Inconel, 1,690 grains, 10 workers: fit time | 131.99 s | **28.52 s** | 4.63× |
| Inconel: peak aggregate PSS | 0.958 GB | **0.630 GB** | 34% less |
| Inconel: peak aggregate USS | 0.620 GB | **0.314 GB** | 49% less |
| Dexelas ROI, 3 grains, 3 workers: fit time | 18.00 s | **5.74 s** | 3.13× |
| Dexelas ROI: peak aggregate PSS | 3.706 GB | **1.688 GB** | 54% less |
| Dexelas ROI: peak aggregate USS | 3.131 GB | **1.385 GB** | 56% less |

GB denotes decimal gigabytes. The memory measurements cover experiment setup
and fitting, sampling the parent and descendants every 0.1 seconds. Timings
cover `fit_grains()`, including lazy image loading, worker startup and IPC;
package imports and result-pickle writing are outside the timer. Normal-fit
and full-spot-validation timings are separate runs.

These are individual runs on a shared development machine, not statistical
performance guarantees. Both implementations use Python 3.12.7, NumPy 2.3.5,
SciPy 1.17.1, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`. Master was
built separately from its own native sources. The first implementation was
preserved as a separate source tree with its own copy of the native extension.
The [results file](fit-grains-performance-results.json) records input/source
hashes, raw timing and memory measurements, and the earlier investigation.

## Implementation

- Fuse sparse pixel gathering, connected-component labeling, and peak
  statistics in a private C++ extraction path. Bound each batch by its sample
  count, releasing temporary pixel buffers before returning statistics.
  Component ordering, peak ties, intensity dtypes, and centroid arithmetic
  follow the original implementation. Exact integer moments are used only
  within a conservative bound; other data retains NumPy contraction order.
- Build compact CSR indices directly from canonical COO frames. Avoid
  temporary int64 flat indices and differences. Unsorted/duplicate input
  retains the SciPy conversion fallback; readonly input buffers are supported.
- Borrow frame buffers and ROI windows instead of constructing full sparse
  ROI copies. Honor materialized/replaced images and use the existing path for
  unsupported image processing. Borrowed views live outside serialized image
  state and are refreshed for each fit or public spot-pull snapshot.
- Store internal spot records as compact field arrays. Cull and fit these
  arrays directly, materializing the public list representation only when
  requested. Smaller worker chunks reduce retained IPC data.
- Prepare shared immutable frame views and small compilation caches before
  forking. Freeze the inherited Python heap during pool creation to reduce
  copy-on-write churn; restore the parent's prior GC state, including on
  errors and cancellation. Serial fitting releases temporary parameters too.
- Generate angular grids natively. For nearest-neighbor sampling on
  undistorted panels, simplify the projection algebra and reuse grid
  trigonometry. Samples near pixel boundaries trigger the original projection
  using a conservative rounding envelope. Continuous coordinates, distorted
  panels, bilinear interpolation, full spot output, and unusual pixel pitches
  retain their appropriate original calculations. Measured XY uses the
  original scalar projection arithmetic inside a batched native call.
- Cache fixed detector transforms, undistorted measurements, and B-matrix/HKL
  products within each least-squares fit. Replace tiled omega selection and
  sorting with broadcasting and an equivalent two-way comparison, retaining
  first-branch ties and NaN ordering.
- Retain the first implementation's paired CSR sampling, moment-only
  centroids, cached patch geometry, optional omission of unused vertex/area
  output, bounded projection batches, and NPZ index-publication race fix.

The optimization method, finite-difference steps, reflection selection,
thresholds, tolerances, refinement passes, and configured worker count remain
the same. The native extension must be rebuilt after these source changes.

## Numerical acceptance

All **1,690 Inconel fits and their returned spot data match master exactly**.
The normal-fit result also matches master. Comparison uses no numerical
tolerance: it checks grain parameters, completeness, chi-square, signal flags,
reflection ordering, peak/HKL IDs, HKLs, integrated/max intensities,
predicted/measured angles and positions, array shapes/dtypes, and matching
NaNs. Five written spot tables are also **byte-for-byte identical** to master.

All nine example runs match master for both fits and returned spots:

| Dataset | Runs | Coverage |
| --- | ---: | --- |
| NIST ruby, single GE | 6 | GE distortion and all six image series |
| Multi-ruby, composite Dexelas | 1 | Eight detector panels and three grains |
| Multi-ruby, ROI Dexelas | 1 | Shared whole-panel images with eight ROI views |
| DAC alpha-zirconium | 1 | Discontinuous omega wedges |

The scientific and integration regression suite passes **465 tests**. This
includes existing fit-grains example integration and find-orientations
numerical tests, sparse sampling, native statistics, boundary projections,
ROI ownership/invalidation, dtype/threshold edge cases, GC cleanup, and the
NPZ publication race.

Three Textual dashboard UI tests time out during asyncio shutdown; the
scientific suite excludes those tests. The same three-test selection also
hangs against the preserved first implementation. Dashboard source files are
unchanged. This remains a test-suite limitation, rather than a passing check.

## Reproduce

Use independently built source trees for master and the candidate. Do not
share the candidate's compiled transforms with master.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

env/bin/python scripts/benchmark_fit_grains.py run \
  --source /tmp/hexrd-fit-grains-master \
  --config "$PWD/inconel/fitting_config.yml" \
  --grains "$PWD/inconel/inconel_test/grains.out" \
  --workers 10 --output /tmp/fit-master

env/bin/python scripts/benchmark_fit_grains.py run \
  --source "$PWD" \
  --config "$PWD/inconel/fitting_config.yml" \
  --grains "$PWD/inconel/inconel_test/grains.out" \
  --workers 10 --output /tmp/fit-branch

env/bin/python scripts/benchmark_fit_grains.py compare \
  /tmp/fit-master.pkl /tmp/fit-branch.pkl

env/bin/python scripts/benchmark_fit_grains.py run \
  --source "$PWD" \
  --config "$PWD/inconel/fitting_config.yml" \
  --grains "$PWD/inconel/inconel_test/grains.out" \
  --workers 10 --memory --no-return-spots --output /tmp/fit-production

env/bin/python scripts/benchmark_fit_grains.py examples \
  --reference-source /tmp/hexrd-fit-grains-master \
  --source "$PWD" --examples "$PWD/examples" \
  --output-dir /tmp/fit-examples
```

The runner writes isolated analysis paths, resolved configurations, metadata
with input hashes, and fit/spot results. Default runs retain spot data for
validation. `--repeat` also checks exact equality between repeats;
`--profile --workers 1` produces cProfile files; `--write-spots` exercises
text output. Pickles are trusted local benchmark artifacts. Raw artifacts are
in `/tmp/hexrd-fit-grains-stage2`, with master and first-stage validation
artifacts in `/tmp/hexrd-fit-grains-bench`.

```bash
HEXRD_EXAMPLE_REPO_PATH="$PWD/examples" env/bin/python -m pytest \
  tests/test_npz.py tests/test_experiment.py tests/test_fit-grains.py \
  tests/test_benchmark_fit_grains.py tests/test_fit_grains_buffers.py \
  tests/core/imageseries tests/core/instrument \
  tests/core/fitting/test_fitting_grains.py tests/test_find_orientations.py \
  -k 'not dashboard_retains_completion_and_failure and not dashboard_navigation_and_cancellation' -q
```

## Follow-up: dashboard and centroid research (October 28)

The [follow-up report](fit-grains-dashboard-and-centroids.md) documents the shared
Textual fit-grains dashboard, fresh exact-master comparisons, and centroid
experiments. The previously reported UI shutdown limitation is resolved; those
tests now pass. The earlier measurements above remain historical results.

## Further acceleration (October 28)

The [latest acceleration report](fit-grains-acceleration.md) measures the new
implementation against the frozen working tree from the start of that follow-up:
5.66× with normal spot output and 5.14× without, using three-run medians.
It records fresh exact-master checks, output-file comparisons, and 537 passing
tests. Earlier measurements in this document remain historical results.
