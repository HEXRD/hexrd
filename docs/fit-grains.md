# Fit-grains

Run `hexrd fit-grains configuration.yml`. An interactive terminal displays the
shared Textual workflow dashboard; `--no-tui` uses plain logs. The dashboard
shows stages, memory, grain progress, frame/spot work and activity age. Cancel
run, `q` and Ctrl+C cancel the workflow; after completion, `q` closes the UI.
Both commands share console logging, analysis-log ownership and optional
profiling. Worker-pool ownership is separate from fit setup; serial, staged
and worker results share completion-event formatting. These internal helpers
do not change command options or the dashboard presentation.

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
For large FCH5 scans with at least 512 MiB available per fitting worker, grain
mode now shares frame reads across all patch batches within each grain's pass.
It uses the same staged coordinator, without changing the grain fit order,
command options or dashboard. Smaller caches retain the existing extraction path.

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
worker owns reusable input and decode buffers. Both measurement modes enable
the native decoder, including after worker serialization or a process change.
The reader count also respects the CPU share available to each fitting process.
Adjacent chunks are read together in physical order, retaining enough batches
for the available readers. No Python executor or image scratch files are created.

On Linux, `HEXRD_FIT_GRAINS_DIRECT_IO=1` uses aligned direct FCH5 reads to avoid
filling the OS page cache with a large source file. It does not change the pixel
values. The benchmark exposes this as `--direct-io` with lazy image access.

## Exact measurements

Staged execution segments the original grain-dependent angular windows. It
retains pixel support until each window is complete and preserves threshold,
omega order, 26-neighbor connectivity, component selection, centroid arithmetic
and optimizer behavior. Detector-space centroids alone are not equivalent to
these measurements; the unused standalone centroid experiments have been removed.

Both measurement modes use one coordinator and one window model, containing
source coordinates, frame IDs, shape and statistics destinations. Reads are
split into at most 65,536 pixels. The coordinator releases completed windows
and calls the unchanged native statistics kernel.
Repeated polar subpixels share one source-pixel value within each omega window.
A compact inverse map reconstructs the original grid before segmentation,
preserving connectivity, pixel bits and accumulation order. Workspace accounting
includes that reconstruction and the smaller live buffers.
Fallbacks remain for supported formats, noncanonical sparse data, numerical
ranges and user-provided geometry that require the reference implementation.
Use the Python code and native extension from the same branch; earlier native
I/O APIs are not supported.

Exact source pixels can be reused across fitting passes. A byte-bounded cache
keys entries by source frame and integer detector coordinates. Entries are
immutable byte blocks, optionally compressed losslessly with C-Blosc. Resizing
the cache evicts entries without migrating their representation. Each compressed
array decodes to at most 1 MiB; one decoded key/value pair is memoized. Without
the native codec, entries use raw byte blocks within the same allowance.
Grain mode retains raw pixels only until that grain finishes, avoiding compression
and growth with the number of grains. Its 512 MiB workspace comes from the existing
per-worker allowance: decoder/index storage and retained pixels each receive at
most 128 MiB, leaving room for planned geometry and live windows. Oversized plans
use the existing bounded extraction fallback. `reuse_pixels: false` disables
retained source pixels while preserving shared frame reads.
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

The additional per-grain batching change is compared with `0596c332`, the branch
revision before this change (not master):

| Sven sample | Fitting workers | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| First 4 grains | 1 | 36.8 s | 16.4 s | 2.24× |
| First 13 grains | 1 | 117.8 s | 47.5 s | 2.48× |
| 32 distinct, evenly sampled grains | 2 | 139.7 s | 67.3 s | 2.08× |

The serial timings are medians of two warm runs; the two-worker comparison uses
one matched run per revision. Each worker received 512 MiB of image workspace.
Direct FCH5 reads bypassed the filesystem cache, runs were sequential, and BLAS
used one thread. Single-process runs had a 4 GiB address-space limit; the
two-worker run used 1.5 GiB per process plus a 3 GiB process-tree RSS watchdog.
Observed peak RSS was 0.76 GiB for the new serial runs and 1.57 GiB for the new
two-worker run. A separate 2,000-grain orchestration regression checks that
measurements remain exact and retained pixels are cleared after each grain;
it is not a throughput measurement on 2,000 real grains.

The eager Inconel 200-grain check took 6.46 s before and 6.51 s after. The unchanged
staged Sven 52-grain path took 47.36 s before and 47.23 s after. This change adds
50 net runtime lines, reuses the existing native kernels and coordinator, and
requires no native-extension rebuild. The measurements do not establish remote
HDD throughput or full 920/2,000-grain default-mode throughput.
All benchmark fits and returned spot data were bitwise identical, as were the
written grain arrays; all 301 spot files and written `grains.out` files matched
byte for byte. All 18 example runs matched master bitwise, and the guarded
four-grain Sven case matched the master scientific reference. The 442 focused
tests passed. During real grain-mode FCH5 reads, TUI cancellation cleaned up in
0.081 s after the button request and 0.247 s after Ctrl+C.

Earlier guarded comparisons against the simplified branch at `205d8483` used
one fitting process, a 512 MiB cache/workspace and a 4 GiB allocation ceiling:

| Mode | Grains | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Grain (default) | 4 | 112.8 s | 39.8 s | 2.83× |
| Grain (default) | 13 | 277.5 s | 113.8 s | 2.44× |
| Staged | 52 | 64.2 s | 44.6 s | 1.44× |
| Staged (synthetic stress test) | 2,000 | 2,600.8 s | 990.6 s | 2.63× |

The four-grain figures are medians of two runs; the larger cases use one matched
run per revision. For the 2,000-estimate stress test, all 920 local Sven estimates
were followed by 1,080 perturbed copies (seed 20261029; standard deviations
0.0003 for orientation components and 0.005 for translations). Both runs used
the same image data and scientific settings. All fitted values were bitwise
identical, all 2,000 spot files were byte-identical, and the written grain
results matched. Peak RSS was 1.40 GiB for that run. The first pass
used 12 groups instead of 28, though full groups left almost no pixel-reuse
cache within the workspace.

A separate default-mode test of 32 distinct, evenly sampled Sven estimates with
two fitting workers took 348.0 s before and 131.7 s after (2.64×); fitted results
and spot files matched exactly. Its process-tree peak RSS was 1.53 GiB after
and 2.03 GiB before. Each process had a 1.5 GiB address-space limit and an
additional 3 GiB process-tree RSS watchdog. These tests do not establish the
speed of default mode at 2,000 grains, 2,000 independent real grains, or 10–16
workers on the remote HDD. Full 920-real-grain throughput remains unmeasured.
Master is the numerical reference for the example comparisons.

[Validation evidence](fit-grains-validation.json) records reference commits,
configuration fingerprints, guarded timings and the cleanup checks. Earlier
validation records remain in git history at `205d8483`, `42687db7` and `c6faf32d`;
experiment reports and raw measurements are also available at `42f7623e`.
