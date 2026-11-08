# Fit-grains

Run `hexrd fit-grains configuration.yml`. An interactive terminal displays the
shared Textual workflow dashboard; `--no-tui` uses plain logs. The dashboard
shows stages, memory, grain progress, frame/spot work and activity age. Cancel
run, `q` and Ctrl+C cancel the workflow; after completion, `q` closes the UI.
Both commands share console logging, analysis-log ownership and optional
profiling. Worker-pool ownership is separate from fit setup; serial, staged
and worker results share completion-event formatting. These internal helpers
do not change command options or the dashboard presentation.

During startup, the existing stage-details panel reports window planning and
image-chunk indexing. During extraction it shows frame counts across workers'
current batches, even while the completed-grain count is zero. These counters
reset as batches advance; they are not a percentage of the entire fit. Shared
scans belong to a group rather than its last planned grain. The layout and
controls are unchanged; no separate Fit Activity panel is added.

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

`measurement_mode` defaults to `grain`, which fits grains independently
with the configured multiprocessing setting. `staged` advances grains by
fitting pass and shares source-frame reads across a memory-bounded group.
For large FCH5 scans with at least 512 MiB available per fitting worker, grain
mode now shares frame reads across up to 128 grains per worker task, using the
same staged coordinator. Parallel tasks have similar grain counts to avoid a
short final wave leaving most workers idle. Each grain keeps its own tolerance
passes, fitted parameters, refit and output; results retain input order. Command
options and dashboard layout are unchanged. Smaller caches and single-grain
tasks retain the existing extraction path.

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
Within a compressed chunk, the native sampler now skips codec blocks that
contain no requested pixels. Complete queried rows are still checked for sparse
coordinate ordering, and fully requested chunks use the existing full decoder.
This reduces decompression work; compressed chunks are still read in full.

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
Single-grain tasks retain raw pixels until that grain finishes. Shared batches
use the staged lossless cache and clear it after the batch. Their 512 MiB workspace
comes from the existing per-worker allowance; planning, decoder/index storage,
retained pixels and live windows share the same accounting. Groups that exceed
the workspace are split, and oversized individual plans use the existing bounded
extraction fallback. `reuse_pixels: false` disables retained source pixels while
preserving shared frame reads.
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

Native extensions belong to the source checkout that built them. Switching Git
branches leaves ignored `.so` files in place; rebuild after switching branches.
Fit-grains checks its required native kernels before accessing images and gives
the rebuild command when an incompatible extension is installed. This prevents
an orientation-only build from silently using slow fitting fallbacks.

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

### First-run requirement and native compatibility (current)

Production fitting uses the original images. The separate threshold-cache
preparation experiment was withdrawn: its 648.52-second preprocessing cost made
the first 2,103-grain run 9.9% slower overall. Its prepared-input timing is not
an end-to-end production speedup; see the [archived experiment](fit-grains-input-cache-experiment.md).
The preparation command and its tests have been removed.

A locally installed extension was identified as an orientation-only build,
missing native spot statistics, sparse views and pixel queries. A clean guarded
rebuild reproduced the validated fitting binary byte-for-byte. Startup now
rejects that mismatch before image access. Patch geometry also avoids copying
the detector's full mask into a configuration it does not use; panel clipping
continues to consult the original mask.

The CLI now computes the detector-wide maximum angle only when applying new
exclusions. With `reset_exclusions: false`, that expensive result was discarded.
Saved exclusions remain intact. On Sven, the fresh-process CLI setup path
(configuration/material/estimate loading, stopped at fit entry) fell from a
median **2.72 to 1.26 seconds**; peak RSS fell from **1.82 to 0.61 GiB**.
This setup check used a four-grain estimate and the full Sven detector.
Both measurements use the same five selected HKL families and exclude package
imports and image analysis. Two runs per revision were measured in ABBA order.

The full 2,103-grain original-image fit took **749.85 seconds**, versus
**763.98 seconds** for the previous implementation, with **2.29 GiB** peak
process-tree RSS. There is no preprocessing cost. The 13- and 160-grain pairs
are within setup/I/O variation and do not demonstrate an overall speedup.
These measurements support removing the known first-run penalties; they do not
establish universally faster wall-clock times.

All 460 focused tests and all 18 master example comparisons pass. The Sven
13-, 160- and 2,103-grain fits and written outputs match exactly; returned spots
were also checked for 13 grains. A four-grain original-input CLI run matches
the full reference subset exactly, with two workers, a 1.5 GiB allocation cap
per process and 1.54 GiB peak tree RSS including the external sampler.
Detailed scope and timings are in the [validation record](fit-grains-validation.json).

The local `sven_dataset/fitting_original.yml` uses original images, two workers
and a 1 GiB aggregate cache. The tested Linux launch keeps allocations bounded
in the parent and both workers, and bypasses the large input's filesystem cache:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HEXRD_FIT_GRAINS_DIRECT_IO=1 python -c \
  'import resource; resource.setrlimit(resource.RLIMIT_AS, (1536*1024**2,)*2); from hexrd.hedm.cli.main import main; main()' \
  fit-grains --no-tui sven_dataset/fitting_original.yml
```

### Selective decompression and startup progress (previous)

Compared with `8c9eda8d`, native FCH5 sampling skips unused interior codec
blocks as well as unused ends of chunks. A small bit mask tracks requested
source bytes; pixels, sparse-coordinate validation, segmentation and fitting
arithmetic are unchanged. This requires rebuilding the native extension, with
no new dependency or floating-point compiler relaxation.

| Sven grains | Fitting workers | Before | After | Speedup | After peak tree RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 1 | 26.38 s | 23.77 s | 1.11× | 0.80 GiB |
| 160 | 2 | 73.55 s | 74.12 s | 0.99× | 2.25 GiB |

The 13-grain figures are medians of two runs per revision. The 160-grain case
uses one run per revision with real, evenly sampled estimates. The larger case
is effectively unchanged; this work does **not** achieve another 2× speedup.
Both comparisons write spot files and use direct reads, one BLAS thread, and
512 MiB workspace per worker. Serial runs have a 4 GiB address-space limit;
two-worker runs have 1.5 GiB per process and a 3 GiB aggregate RSS watchdog.

Startup reporting now exposes planning, chunk indexing and current-batch frame
counts in the existing details panel. Completed-grain counts remain truthful:
a shared batch can scan many frames before its first grain finishes. This is
a visibility fix, not a claim that the first completed grain arrives sooner.
In a separate 13-grain run with telemetry enabled, planning appeared at 0.68 s,
the first sampled-frame update at 3.08 s, and the first completed grain at
23.36 s, measured from entry into `fit_grains`.
Row-sharding and larger speculative pixel-cache experiments were discarded
because their measured benefit did not justify their complexity.

All 452 focused tests pass. All 18 example runs and the bounded four-grain Sven
case match the master scientific reference bitwise. The before/after Sven fits,
all 173 written spot files, `grains.out` files and `grains.npz` arrays match
exactly. Returned spot arrays were also compared for 13 grains; the 160-grain
run retained written spots only. Grain-mode cancellation during actual FCH5
reads cleaned up within 0.26 s after the request reached the workflow. Full UI
dispatch timings, startup measurements and comparison scope are in the
[validation record](fit-grains-validation.json).

### Shared reads between grains (previous)

The shared-batch change is measured against `8b981bdf`, the previous optimization,
not master. Each worker shares exact sparse reads across a bounded batch of
independent grains. No optimizer, threshold, segmentation or floating-point
rules change.

| Sven grains | Fitting workers | Before | After | Speedup | After peak tree RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 19.1 s | 14.2 s | 1.34× | 0.76 GiB |
| 13 | 1 | 54.8 s | 24.8 s | 2.21× | 0.80 GiB |
| 32 | 2 | 68.5 s | 32.6 s | 2.10× | 1.69 GiB |
| 160 | 2 | 304.0 s | 73.5 s | 4.14× | 2.25 GiB |

The 4/13/32-grain figures are medians of two runs per revision. The 160-grain
comparison is one run per revision using evenly sampled real estimates. All
runs write spot files, use direct FCH5 reads and allocate 512 MiB of workspace
per worker. Serial benchmarks have a 4 GiB address-space limit; two-worker runs
have 1.5 GiB per process and a 3 GiB process-tree RSS watchdog. The largest
comparison increases peak RSS from 1.58 to
2.25 GiB while keeping the workspace allowance unchanged.

This adds 37 net runtime lines, reuses the existing C++ reader/statistics kernels
and lossless Blosc cache, and requires no binary rebuild or new dependency.
The sparse path already avoids full detector-frame expansion. A separate
13-grain instrumented comparison reduced frame visits from
70,769 to 10,795 and FCH5 sampling calls from
30,368 to 5,270. Only the original small angular windows
are reconstructed for unchanged segmentation and centroid arithmetic.

All 450 focused tests pass, all 18 example runs match master bitwise, and the
bounded four-grain Sven comparison matches the master scientific reference.
Benchmark fits and returned spot arrays are bitwise identical; all 461 spot
files and generated `grains.out` files match byte for byte, with identical
`grains.npz` arrays. During actual FCH5 work, grain-mode cancellation cleaned
up within 0.23 seconds after the Cancel/CTRL+C request was received. The full
TUI dispatch and cleanup timings are recorded separately in the validation JSON.

A 2,000-grain synthetic orchestration check validates repeated batches, cleanup,
result ordering and progress. Full 920/2,000-grain default-mode throughput and
remote HDD performance remain unmeasured. The eager Inconel check took
6.16 → 6.16 seconds; explicit staged Sven took
53.69 → 55.42 seconds. These paths receive no new speedup.
See [the validation record](fit-grains-validation.json) for measurements and scope.

### Shared reads within a grain (previous)

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
