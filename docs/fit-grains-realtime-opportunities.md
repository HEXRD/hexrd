# HEXRD core processing: implemented gains and next targets

The strongest opportunity is to remove redundant image representations. Existing
NumPy/SciPy operations already execute native loops; simply translating their
Python callers would retain most of the work.

## Measured opportunities

The installed C++ kernel now serves ordinary offline map building and live
incremental maps. Times start with a decoded image and exclude acquisition, decoding,
source reads, static geometry setup, scoring, and fitting.

| Core path | Existing mean | Native mean / P95 | Scope |
|---|---:|---:|---|
| Dense Inconel image → ring histograms | 50.00 ms | **4.90 / 5.48 ms** | 48 real frames; about 10× faster |
| Dense Sven image → ring histograms | 340.55 ms | **0.91 / 1.14 ms** | 24 real frames; about 374× faster |
| Complete incremental append, Inconel | — | **5.67 / 6.21 ms** | 24 real frames; masking, panel merge and storage included |
| Complete incremental append, Sven | — | **1.19 / 1.51 ms** | 24 real frames; original float64 pixels |

These are stage improvements, **not complete fit-grains speedups or proof of
100 Hz end-to-end operation**. The dense comparison describes the generic
streaming-adapter route. Existing FCH5 readers already provide sparse and
threshold-filtered input; preserve those capabilities rather than densifying
batch inputs to use the dense kernel. Both routes are selected automatically
in offline `build_eta_omega_maps()`; no streaming dependency or new CLI option
is needed. Fitting already-identified grains does not build these maps and
does not gain from this change.

Three alternating full offline Inconel map builds, including reads and per-call
geometry, improved from **14.33 s to 1.89 s median (7.58×)**. A 24-frame Sven
FCH5 prefix improved only **1.03× (1.88 s to 1.82 s)**. Every map and axis matched
bitwise. These are one-worker map-stage measurements, exclude material/config
construction and fitting, and are not cold-disk benchmarks. Peak process RSS
was 3.01 GiB under a 4 GiB allocation ceiling.

### 1. One typed native map plan for dense frames

`_AdapterImageSeries.pixel_chunks()` converts generic input frames to CSR and
then flat pixel IDs. `build_eta_omega_maps()` reconstructs a full float64 weight
image, thresholds/scatters values into it, and calls `_histogramPrefix` for each
histogram block. A stream starts with dense pixels, making that round trip
unnecessary.

The native plan caches the existing ring pixel indices/cuts and reads the original
typed array directly. It applies the same threshold, preserves 65,536-pixel
block boundaries and sorted-eta prefix-sum order, and returns all rings in one
call with the GIL released. There is no full-image CSR, pixel-ID array, float64
copy, or weight image. Sven's configuration needs 378,525 planned pixel visits
per frame, compared with 18,093,576 image pixel slots.

All 72 sampled outputs matched the **public** `build_eta_omega_maps()` rows
bitwise, including NaN coverage. Typed, strided, unaligned and readonly inputs
are covered by regression tests. The installed extension uses the branch's
optimized compilation settings without fast-math; ordered prefix additions
remain sequential.

### 2. Improve integer fast-path selection for sparse sources

`_direct_sum_safe()` bounds sums using the entire detector area. Inconel's
largest configured ring has only 335,144 pixels. A conservative uint32 bound is
therefore about 1.44×10¹⁵, below float64's exact-integer limit of 2⁵³.

With that bound, the existing `_binPixels` kernel reproduces the prefix route
exactly and avoids reconstructing a dense weight image. Each frame is binned
into a temporary histogram before the usual panel merge, preserving panel and
repeated-row rounding order. The older direct-to-map route also bounds repeated
rows and panel sums. Arbitrary floating-point sources retain ordered prefix
arithmetic.

### 3. Persistent incremental maps

`EtaOmegaMapPlan` prepares geometry before acquisition;
`EtaOmegaMapAccumulator` bins each frame once and exposes readonly acquired
prefixes. Capacity and a byte budget bound storage, optionally backed by an NPY
file. Existing filters operate on snapshot copies. The sandbox enables this with
`receive --incremental-maps`. Production CLI options and the TUI are unchanged.

The omega-axis helper also fixes a gapped scan ending exactly one rotation
after its start: that endpoint previously wrapped to the beginning. Contiguous
scan arithmetic is unchanged.

### 4. Native sparse packing and bounded frame workers

A small native SIMD packer replaces dense-image coordinate construction in
generic offline adapters, lazy fitting views and live pixel spooling. It emits
canonical CSR arrays in row order, retaining every nonzero and its dtype.
Uncommon layouts/dtypes retain SciPy's implementation. Conversion alone improved
**8.43× on Inconel (26.51 → 3.15 ms)** and **2.39× on Sven's fractional float64
frames (227.56 → 95.20 ms)**, with all CSR buffers bitwise identical.

The receiver uses reusable native decode destinations and casts/checks only
nonzero values when restoring source dtypes. Four threads perform independent
decode/map/spool jobs; at most four jobs are submitted, and the owner commits
maps and snapshots in acquisition order. Original pixels remain on disk for
exact fitting. No full images are sent between Python processes.

## Next priorities inside HEXRD

The [complete-processing strategy](fit-grains-realtime-strategy.md) separates
the remaining scientific workload from receiver throughput and describes the
persistent scheduler, exact reuse rules and acceptance tests needed for 100 Hz.
Persistent analysis and archive overlap now reduce the measured full-analysis
process wall time from 265.69 to 147.91 seconds (1.80×) with bitwise-equal outputs.
The [paired and live results](live-fit-grains.md#persistent-analysis-and-archive-overlap)
separate this gain from the earlier receiver-only improvements.

| Priority | Change | Benefit and correctness requirement |
|---|---|---|
| 4 | Representation-aware image-source interface | Preserve existing zero-copy decoded array views and already-sparse inputs. The `csrnpz` Eiger decoder currently densifies a CSR array before generic map building makes it sparse again. Canonical sparse access should bypass that round trip; preserve duplicate-pixel semantics through fallback where needed. |
| 5 | Extend frame-centric spot sampling across active grain groups | Existing staged extraction already groups frame requests and compacts repeated subpixels. Share immutable source tiles and coalesce useful same-frame requests under one byte budget. Recompute windows when fitted parameters change; cached centroids alone cannot preserve master results. Measure repeated decodes and query duplication first. |
| 6 | Profile and extend float64 sparse extraction | `_SparseFrames.extract` has an integer-only shortcut. Sven's fractional float64 data use generic gathering plus native statistics. An exact floating-point sparse kernel could remove further window inflation, but its gain and reduction-order equivalence are not measured yet. Do not cast fractional data to integers. |
| 7 | Amortize orientation discovery and fitting across snapshots | Decode/map/spool work is now bounded and parallel. Scoring and window-ready fits still need their own latency budgets; avoid rerunning a full orientation search after every frame. Preserve final complete-scan search/fitting semantics. |

`grain_objective.hpp` is already native and deliberately retains NumPy's matrix
product calls to preserve reference arithmetic. Solver/Jacobian changes are a
lower-confidence starting point under exact-output requirements. The measured
representation changes offer much larger, safer initial gains.

The complete 1,440-frame incremental-map Inconel replay preserves every source
pixel and omega value. All 1,690 final grains, 10,736,115 trials/scores, archive
arrays and 1,691 text files match the independent offline reference bitwise.
Eighteen example comparisons (nine studies, both fitting modes) and a four-grain
Sven sample match master `0a041b17`, including extracted spot data. A full
high-grain-count Sven streamed fit remains unvalidated.

**The parallel receiver now exceeds 100 Hz on local Inconel replay:** **150.33 Hz**
for complete plain-LZ4 decode/map/spool processing with concurrent prefix analysis,
up from 13.27 Hz for the serial incremental receiver. Worker service time was
17.70 ms mean / 22.24 ms P95; throughput differs from per-image latency. Every
pixel/omega edge and all 1,690 final-grain outputs again matched offline bitwise.
Eighteen master example comparisons passed again. Aggregate sampled RSS peaked
at **2.92 GiB**, with concurrent role allocation allowances totaling **7.5 GiB**.
This is not a complete fit every 6.7 ms. Production qualification still needs
long raw Eiger 16M captures, realistic sparsity and durable HDD throughput.
The complete bitshuffle-LZ4 replay also passed at **135.30 Hz**, with identical
pixels and final outputs. Its worker service was 26.55 ms mean / 29.06 ms P95
and aggregate sampled RSS peaked at 2.94 GiB. Both local runs used 3500×3500
frames and offered 150 Hz; bitshuffle-LZ4 slowed the sender through backpressure.
Concurrent work was prefix orientation analysis; full fitting lagged acquisition.

Earlier prototypes, `integrated_core` and the new parallel receiver measurements are recorded in
[the validation record](live-fit-grains-rate-validation.json). Diagnostic code
and raw traces remain in `.local-bench/live-fast/`. Numerical
experiments used allocation ceilings and single numerical-library threads.
