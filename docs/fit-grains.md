# Fit-grains

Run `hexrd fit-grains configuration.yml`. An interactive terminal displays the
shared Textual workflow dashboard; `--no-tui` uses plain logs. The dashboard
shows stages, memory, grain progress, frame/spot work and activity age. Cancel
run, `q` and Ctrl+C cancel the workflow; after completion, `q` closes the UI.

## Configuration

Keep the experiment's existing threshold, tolerances, refit and material
settings. These determine the scientific result. The following options control
execution and image access:

```yaml
fit_grains:
  measurement_mode: staged
  image_cache:
    mode: lazy
    max_size_mb: 512
    read_workers: 4
    reuse_pixels: true
```

`measurement_mode` defaults to `grain`, which processes grains independently
with the configured multiprocessing setting. `staged` advances grains by
fitting pass and shares source-frame reads across a memory-bounded group.

`image_cache.mode` accepts `auto` (default), `lazy` and `eager`. Lazy readers
open files on demand, share an aggregate cache across detector/ROI views in
each worker, and release handles on cancellation or completion. Frame-cache
NPZ/FCH5, HDF5, TIFF and processed/shared detector views retain their existing
image-series semantics. Unsupported chunk layouts use HDF5 or the original
frame/CSR path.

`max_size_mb` is a workspace/cache allowance in MiB, not a hard limit on total
process RAM. The grain workflow divides its allowance between workers; staged
execution includes planning and live pixel support in its group budget. Large
frames can require temporary storage beyond the cache. The benchmark's separate
address-space guard enforces a hard allocation ceiling. An automatic allowance
is selected when `max_size_mb` is omitted. Zero disables the grain image cache;
staged execution keeps a minimum bounded workspace of 16 MiB.

`read_workers` accepts 1–8 and defaults to 4. A single native decoder pool is
shared across panels, pixel reads, and bounded row-index read-ahead. Each active
worker owns bounded input and decode buffers. No Python executor or image
scratch files are created.

On Linux, `HEXRD_FIT_GRAINS_DIRECT_IO=1` uses aligned direct FCH5 reads to avoid
filling the OS page cache with a large source file. It does not change the pixel
values. The benchmark exposes this as `--direct-io` with lazy image access.

## Exact measurements

Staged execution segments the original grain-dependent angular windows. It
retains pixel support until each window is complete and preserves threshold,
omega order, 26-neighbor connectivity, component selection, centroid arithmetic
and optimizer behavior. Detector-space centroids alone are not equivalent to
these measurements; the unused standalone centroid experiments have been removed.

The planner captures one immutable set of source coordinates per window, with
its frame IDs, shape and statistics destinations. The native extension measures
those windows directly; there are no separate batch and patch models.
Compatible NumPy buffers are borrowed.
Fallbacks remain for supported formats, noncanonical sparse data, numerical
ranges and user-provided geometry that require the reference implementation.
Use the Python code and native extension from the same branch; earlier native
I/O APIs are not supported.

Exact source pixels can be reused across fitting passes. A byte-bounded cache
keys entries by source frame and integer detector coordinates. Under pressure,
C-Blosc compresses entries losslessly in RAM; each compressed entry decodes to
at most 1 MiB. Unsupported codecs retain raw entries within the same allowance.
All pixel-cache requests use the native query path. Unsupported requests bypass
the cache and use the regular reader. HDF5 fallback sampling uses one chunk
accessor and releases its borrowed chunk on completion or cancellation.

Integer/half-integer floating counts use native integer moments only when
range checks prove the required sums and moments exactly representable. Other
intensities retain the original NumPy/BLAS reductions. Floating-point arithmetic
is not relaxed.

Row indexing checks cancellation between chunks; native waits check every
25 ms. Pending work is cancelled and
active jobs drain before buffers and file descriptors are released. A running
OS read must return before its job can finish cancelling. The solver checks
cancellation at objective evaluations. UI counters are presentation-only.

## Build

The C++ extensions use `-O3` and release assertions. To enable the optional GCC
LTO profile, rebuild while HEXRD is stopped:

```bash
BOOST_INCLUDE_DIR=/path/to/boost/include HEXRD_ENABLE_LTO=1 \
  python setup.py build_ext --inplace --force --parallel 1
```

Boost.Unordered is optional and header-only; builds without its flat-map header
use the standard-library query path. The include hint is unnecessary when the
header is on the compiler's normal search path. The build does not download
Boost. LTO uses one worker and retains IEEE arithmetic and portable CPU targeting.
Changing build flags requires `--force` or a clean build.

## Validation

Use the benchmark tool for guarded runs and exact comparisons:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_fit_grains.py run \
  --source /path/to/checkout --config /path/to/config.yml \
  --grains /path/to/grains.out --measurement-mode staged \
  --workers 1 --image-mode lazy --image-cache-mb 512 \
  --memory-limit-mb 4096 --direct-io --limit 4 --write-spots \
  --output /tmp/fit-current

python scripts/benchmark_fit_grains.py compare \
  /tmp/fit-reference.pkl /tmp/fit-current.pkl
```

The hard address-space guard requires one fitting process. Run large datasets
sequentially and keep BLAS single-threaded. Compare all example studies in both
measurement modes with master, as well as the Sven fits, extracted spots and
spot files. Guarded master validation uses bounded pixel input when eager
loading would exceed local RAM.

The last performance measurements, before simplification, showed 1.11×, 1.22×
and 1.48× gains for 4, 13 and 52 Sven grains against `e968097b`; measured peak RSS
stayed below 1.09 GiB. These are not speed comparisons with master. Full
920-grain throughput and remote HDD scaling remain unmeasured.

[Validation evidence](fit-grains-validation.json) records reference commits,
configuration fingerprints, guarded timings and the cleanup checks. Earlier
validation records remain in git history at `203808c8`; experiment reports and
raw measurements are also available at `42f7623e`.
