# Live fit-grains sandbox

This experimental sandbox replays an image series as an Eiger detector and
receives it into an asynchronous orientation-discovery/fit-grains workflow.
It leaves the production CLI and scientific algorithms unchanged.

## Current acceptance status

**The complete 10 ms live-update goal is not met.** Final master-equivalent
results may finish after the scan, as clarified by the user; every arriving
frame still needs a provisional scientific update within 10 ms.

The earlier 111.43 Hz full-Sven measurement is aggregate capture throughput,
with several frames in flight. It does not establish per-frame latency or
include live grain fitting. The earlier 24-frame capture check took
**43.34 ms mean / 45.85 ms P95 / 47.68 ms maximum**; all frames missed 10 ms.
With the native sparse decoder and asynchronous checksums integrated, the same
24-frame input offered at 100 Hz now takes **9.13 ms mean / 12.49 ms P95 /
12.72 ms maximum**, including scheduling delay, maps, support persistence,
archive validation and status publication. **Four frames still miss 10 ms.**
Two other short runs average 9.68 and 9.70 ms, with six and seven misses;
the largest observed delay is 15.68 ms. These are short probes, not a bound.
Static setup and preloading the 225 MiB detector-emulator fixture are excluded;
this uses a previously prepared raw archive, not a new full-payload disk write.
The receiver uses a 2 GiB allocation guard, with 3 GiB during static setup.
This test excludes grain discovery and fitting and is not a complete 100 Hz
scientific workflow result.

A 200-grain native diagnostic attributes 5.69 of 11.96 seconds to spot
extraction, versus 0.45 seconds to optimizer residuals. It uses already-arrived
fitting support and excludes discovery, so it is not an end-to-end benchmark.
The prototype still recomputes prefix discovery/fits and lacks a per-frame
scientific update path. Its 2.5 GiB analysis-coordinator allowance also cannot
hold Sven's configured 79.8-million-trial grid plus scientific working arrays;
the successful isolated discovery probes used larger, separate bounded roles.
These integration gaps must be resolved before claiming the target rate.

A native bit-selection kernel performs block LZ4 decoding and exact
threshold selection directly on bit planes, reconstructing retained pixel
values without inflating a full image. Across **all 3,600 Sven frames**, eight
workers take **3.62 ms mean / 4.57 ms P95 / 7.64 ms maximum**, with zero kernel
10 ms misses. All **279,046,070** retained values and coordinates match the
independently source-verified support bank bitwise; payload MD5s are also
verified outside the kernel timing. This excludes checksums, geometry,
persistence, discovery and fitting, so it does not establish the full live
10 ms deadline. That standalone experiment uses a **1 GiB** allocation guard;
the implementation now lives in HEXRD's native extension and is integrated
into the sandbox receiver. A separate full-scan correctness run verifies all
**3,600 frames / 279,046,070 retained pixels**, all sparse support buffers and
all fresh master map rows bitwise, without inflating full images. Every MD5 is
verified before finalization. This run includes disk reads and parity checks
and is not a performance benchmark. Other codecs, dtypes and unsafe sparse
sums retain the established dense path.

The receiver now retains its persistent analysis service after provisional
results finish during acquisition; a regression covers two such updates and
the final analysis. A fresh 200-grain sample from current master discovery
also matches unmodified master's fitted fields and every spot file exactly,
using bounded exact-support adapters. This does not validate fitting all
15,559 freshly discovered Sven grains.

## Sven replay with provisional grain parameters

An in-process replay read all 3,600 Eiger messages, decoded and persisted their
support, updated the maps, and tested all 79,828,799 configured trial orientations
once across acquisition. At frame 900 the experiment began tracking the first
trial in that frame's block above the configured completeness threshold. It
published 152 fits as more spot windows closed. All 12 parameters after the
last update matched an ordinary offline fit of the same trial at the first
tolerance exactly. Extending this to all grains requires clustering and
scheduling newly discovered models. The complete canonical tail also needs
separate validation.

| Measured stage | Mean time per frame |
|---|---:|
| Read emulated wire message from disk | 8.42 ms |
| Capture, map and persist support | 8.44 ms |
| Score the current block of trial orientations | 9.05 ms |
| Advance the tracked fit and publish the frame update | 0.48 ms |

The serial replay took 95.11 seconds (37.85 Hz) under a 4 GiB address-space
limit. All 3,600 updates missed the 10 ms deadline measured from the offered
arrival time. A compiler ran during part of the replay. On its own, direct
orientation scoring finished the scan in 36.00 seconds with four threads, but
67 updates still missed the deadline. The earlier method that materialized
reflection bins before scoring took 46.60 seconds and missed 3,582 deadlines.
The isolated and joined timings cover different work.

A second replay overlapped four prefetched wire messages and twelve blocks of
static geometry within a 4 GiB process limit. It took 56.43 seconds (63.79 Hz).
The same 152 fit updates were issued and the final first-pass parameters still
matched. All 3,600 deadlines were missed. The compact support arrays for every
frame and the complete map matched the independently verified reference.

Live coverage requires every provisionally discoverable grain. Rescoring the
entire grid on Sven's first two frames found 1,257,041 passing trials on the
first frame and 3,226,487 on the second. Of the 2,020,018 new crossings, just
1,018 were in the block scheduled for frame 2. The block schedule therefore
cannot supply the requested coverage. These counts precede clustering.

An experiment indexed predicted reflections by omega and retained hit and
valid-reflection counts for every trial in a shard. Updating the affected rows
on each of the 3,600 frames reproduced direct scoring at all eleven checked
prefixes for 32,768 trials. The uncompressed index extrapolates to 36.2 GiB for
the full grid from this shard. Scaling this approach within 8 GiB requires
streaming or compressing the geometry as well as tracking cluster changes.

The complete prefix searches yielded **33,701 provisional clusters on frame
1** and **8,249 on frame 2**. The largest cluster on frame 2 contained
3,156,542 accepted trials. Merging centroids by single linkage now computes
distances on demand when the condensed array would exceed 1 GiB. Averaging
large clusters also maps their quaternions to the fundamental region in chunks.
The first two cluster runs completed in 39.07 and 25.53 seconds with a 4 GiB
process limit. The selected regression suite passed 1,014 tests, the rotation
suite passed 58, and all 18 example fit comparisons matched master, including
the spot files.

A larger geometry index covered **all 79,828,799 trials on each of the first
32 frames**. Its 174,995,480 entries occupied 668 MiB before indexing by detector
cell. The full candidate masks matched direct scoring byte for byte on the
first two frames. Updating eligibility averaged 34.23 ms initially. Grouping
reflections by detector cell and updating a bit mask on threshold crossings
reduced the mean to 11.86 ms, with a 14.26 ms maximum on eight threads. This
timing begins with the acquired histogram and ends with the candidate mask.
Live clustering, spot measurement and fit publication still need their own
share of the 10 ms budget.

The branch processed the full table of 15,559 fresh initial Sven grains
with the configured three-pass fit and refit in 139.35 seconds
using six workers and the arrived support bank. This is fit-stage time. The
full independent comparison with master also checks every spot file.

## Data flow

```text
source image series → Eiger ZeroMQ PUSH → PULL receiver → sparse pixel files
                                                            ↓
                                   immutable acquired-prefix snapshots
                                                            ↓
                                     find-orientations → fit-grains
```

The receiver stores every nonzero pixel, including saturated values, without
thresholding or centroid approximation. One persistent analysis service discovers
grains and fits acquired prefixes using a bounded process pool while reception
continues. Checkpoints encountered while the service is busy are coalesced:
no work queue grows with acquisition.
Partial results are provisional; grain IDs can change between snapshots.

End-of-series requires every expected frame, in order. It cancels any stale
prefix analysis and runs discovery and fitting afresh on the complete scan.
The service retains static calibration, material and fitting processes across
that handoff; canonical initial grains and measurements are recomputed.
It never initializes from provisional fits, an existing `grains.out`, accepted
orientations, or maps from future images. Calibration, material parameters,
and acquisition planning are static inputs. No future image is available to the
analysis process. Received pixel files and published snapshots are immutable.

With `receive --incremental-maps`, static geometry is prepared before connecting
and every acquired frame also updates an unfiltered native map once. The
receiver freezes the resolved analysis configuration, so workers use the same
settings as the map plan. Prefix and final analysis reuse only the acquired map
rows from `maps.npy`; every original pixel is still retained for fitting.
Snapshots with omega gaps fall back to ordinary pixel-based map building.

`--wire-spool` is an experimental alternative to sparse NPZ capture: retain
the incoming compressed message parts in stored ZIP members and decode them
on demand. Every original pixel and header remains available. `--fit-support`
additionally builds an append-only compact index at the configured fitting
threshold. Only fit workers request that index; raw reads and lower-threshold
fits retain the full-pixel route. Individual support records have checksums.
Integer column packing and float64-to-float32 storage are used only when they
preserve values exactly; fitting arithmetic and logical dtypes stay unchanged.
This combination is still under full-pipeline validation, not a production default.

`receive --defer-checksums` opts into explicitly **checksum-pending** provisional
snapshots, status and analysis results. Two checksum workers retain at most four
payloads; a full queue applies backpressure. Every checksum must pass before an
end-of-series snapshot or final analysis is accepted. A failure marks the run
invalid and prevents later finalization, including retries. The default still
verifies checksums before image preparation; public raw-image reads also verify
them. Pending provisional outputs must never be treated as accepted results.

The sparse decoder requires little-endian `bs32-lz4` uint32 wire images, together
with `--wire-spool --fit-support`. `--decode-workers 1` gives that decoder eight
native block workers. The ordinary emulator emits plain LZ4 and exercises the
dense fallback; `publish --wire-input DIRECTORY` can replay an existing
bitshuffle wire capture. Neither selecting these flags nor receiving at
100 Hz establishes the per-frame scientific deadline.

## Run locally

Use this branch's rebuilt native extensions and install the optional packages:

```bash
env/bin/python -m pip install pyzmq lz4
```

The equivalent installation extra is `hexrd[streaming]`. From the repository
root, start the receiver in one terminal (the capture directory must be new):

```bash
env/bin/python scripts/stream_fit_grains.py receive \
  ../inconel/fitting_config.yml .local-bench/live-demo \
  --checkpoint 360 --incremental-maps --decode-workers 4
```

Then start the emulated detector in another terminal:

```bash
env/bin/python scripts/stream_fit_grains.py publish \
  ../inconel/fitting_config.yml --rate 150 --preload-mb 512
```

`--rate` sets the offered frames/second; backpressure may slow the replay. Zero
replays as fast as the receiver can accept. `--preload-mb` optionally prepares
compressed messages before transmission and rejects captures exceeding its
budget (maximum 512 MiB). Zero, the default, loads and encodes one source frame
at a time. Use `MALLOC_ARENA_MAX=2` for the recorded benchmark configuration.
Both commands accept `--endpoint` (default
`tcp://127.0.0.1:9999`) and `--timeout` (60 seconds without progress).
`--frames N` truncates replay for transport experiments; it does **not** claim
parity with the untruncated source scan.
`publish --source-scale .5` permits an exact replay of processed half-count
images through unsigned Eiger messages. This is explicit emulator metadata:
decoded counts are multiplied by one half before scientific thresholding and
calculation. Nonrepresentable, negative, nonfinite, and out-of-range source
pixels are rejected; default replay still requires exact integer counts.

Watch the receiver log and `live-demo/status.json`. Prefix directories hold
normal HEXRD output files and a `result.json` with frame/grain counts, elapsed
analysis time, completion state, and peak RSS. Each analysis has its own adjacent
log. `live-demo/final` is authoritative only after a successful final analysis.
Ctrl+C stops reception and terminates its analysis process; it also interrupts a
publisher waiting for a receiver. Interrupted/incomplete scans are not finalized.

For an independent offline comparison using the **same** YAML:

```bash
env/bin/python scripts/stream_fit_grains.py analyze \
  ../inconel/fitting_config.yml .local-bench/live-offline
env/bin/python scripts/stream_fit_grains.py compare \
  .local-bench/live-offline .local-bench/live-demo/final
```

The comparison checks bitwise equality of discovered orientations, all trial
orientations and scores, fitted parameters/completeness/chi-square, and every
grain archive array and grain/spot text file. The offline command also discovers
its own initial grains; both routes ignore configured estimates and saved maps.

## Validation

The [recorded Inconel run](live-fit-grains-validation.json) on November 17, 2026
replayed all 1,440 frames over localhost TCP. All 7,560,173 nonzero pixel values
and all omega edges survived bitwise; the sparse spool occupied 26.72 MiB.

| Check | Result |
|---|---|
| Complete-scan discovery | Same 10,736,115 trials and 1,690 grains as offline |
| Final scientific outputs | Bitwise identical arrays and all 1,691 text files |
| First provisional fit | 360 frames, 1,691 grains; ready while only 882 frames had arrived |
| Sampled combined process RSS peak | 2.01 GiB; individual final-analysis kernel high-water mark 2.07 GiB |
| Sandbox tests | 28 passed, including frame integrity, cancellation, and allocation guards |

The provisional analysis took 87.6 seconds. The final analysis took 342.1 seconds,
compared with 337.4 seconds for the independent offline reference. These are
single runs, not a speedup measurement. The final fresh pass still runs after
acquisition. This validates the current branch's offline pipeline as the
reference; it is not a new independent master benchmark.

## 100 Hz readiness

**The bounded parallel receiver exceeds 100 Hz on the local Inconel replay.**
It sustained **150.33 Hz** with decoding, native map updates, exact pixel
spooling and concurrent prefix analysis. This is frame processing throughput;
orientation discovery and grain fitting still run asynchronously and lag
acquisition. It is not a complete fit every 6.7 ms or a production Eiger 16M
hardware qualification.

Four threads each own a reusable decoded-image buffer and at most one submitted
job. The [LZ4](https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h) and
[DECTRIS](https://raw.githubusercontent.com/dectris/compression/main/src/compression.h)
native buffer APIs avoid full-frame allocation and
copying; public codec calls remain functional fallbacks when symbols are absent.
Native SIMD packing retains every nonzero in sorted CSR order. Source dtype
restoration and exact-value checks touch only nonzeros. Workers return small
per-panel histograms; the owner commits maps and publishes prefixes in acquisition
order, after pixel files are atomically installed. No full images cross processes.

The plain-LZ4 run's worker service time was **17.70 ms mean / 22.24 ms P95**;
parallel throughput differs from individual-image latency and excludes no worker
stage. The serial commit averaged 0.78 ms. Aggregate sampled RSS peaked at
**2.92 GiB**. All 1,440 frames completed in a 9.57-second first-to-last interval;
the following final analysis took 231.3 seconds. Every pixel/omega edge, all
1,690 final grains and all 1,691 text files matched offline bitwise.

The complete bitshuffle-LZ4 replay also exceeded the target: **135.30 Hz**,
with **26.55 ms mean / 29.06 ms P95** worker service and **2.94 GiB** aggregate
sampled RSS. All pixels, omega edges and final scientific outputs again matched
offline bitwise. Both runs offered 150 Hz; bitshuffle-LZ4 used backpressure
and completed its frame interval in 10.64 seconds, followed by 236.0 seconds
of final analysis. Inputs were the real 3500×3500 Inconel frames, not full Eiger
16M frames. Both tests included concurrent prefix orientation analysis; no
provisional grain fit completed during these short acquisitions.

Earlier diagnostics on the local WSL2 i7-13700HX, with single-threaded numerical
libraries and bounded memory:

| Timed work | Input | Mean time/frame | P95 time/frame | Throughput |
|---|---|---:|---:|---:|
| Actual receiver, 100 Hz offered load, concurrent snapshot analysis | All 1,440 Inconel frames | 65.9 ms service | 72.2 ms service | **15.1 Hz** |
| Previous decode, dtype restoration/validation, CSR conversion; no disk | Inconel, 3500×3500 | 62.6 ms | 69.2 ms | 15.8 Hz |
| Decoded pixels → CSR → eta/omega map; no disk or wire decode | Inconel | 50.8 ms | 53.3 ms | 19.7 Hz |
| Decoded pixels → CSR → eta/omega map; no disk or wire decode | Sven, 4362×4148, original float64 | 356.4 ms | 447.3 ms | 2.8 Hz |
| Prototype decode → CSR, eight processes, native uint32; no maps/fits/disk | Inconel | 73.0 ms latency | 82.4 ms latency | 106.1 Hz |

These are different stage boundaries, so their rates cannot be substituted for
complete workflow throughput. Parallel throughput is distinct from individual
image latency. The dense-to-sparse costs describe the streaming frontend;
existing sparse frame-cache fitting avoids that conversion. The prototype pool
is a diagnostic experiment, not the receiver's
implementation. Its native uint32 route skips the emulator's signed-dtype
restoration; all pixel values were checked. The current restoration route reached
66.8 Hz with eight processes. A Python thread pool reached 51.2 Hz at eight
threads and 50.0 Hz at sixteen. Sixteen full processes were not attempted under
the laptop's 8 GiB constraint.

Inconel memory/pool experiments cycled 48 real frames over 384 tasks. Map timings
used 48 Inconel and 24 evenly spaced Sven frames. The separate Sven CSR probe
included 29 frames and the five densest frames. Input loading and one-off
detector geometry setup were outside per-frame timings. Sven's fractional
corrected intensities were preserved; they were not quantized into detector
counts. These results do not establish raw Eiger 16M detector-rate throughput.

The dominant costs are full-image work. Inconel's serial decode, dtype handling,
and CSR conversion averaged 13.9, 19.3, and 29.3 ms respectively. Sven's combined
decoded-image/map probe spent 254.2 ms in CSR conversion, 20.6 ms packing pixel
coordinates, and 81.6 ms in map accumulation.

The [100 Hz test record](live-fit-grains-rate-validation.json) prepared all
compressed messages in RAM before transmission (298 MiB, bounded to 512 MiB).
No source image reads or encoding limited the timed stream. The receiver took
95.3 seconds between first and last frame completions. The sender enqueued
frames at 15.6 Hz and then the receiver drained queued messages; receiver
completion rate is the relevant sustained rate. Backpressure preserved frames
in this emulator and does not demonstrate lossless physical-detector capture.
NPZ compression/buffered writes averaged 2.45 ms per frame; this is not a durable
HDD throughput measurement. Dropping that stage alone cannot reach 10 ms/frame.

**Results confirmed after the timing run:** every received Inconel pixel and
omega value was bitwise preserved. All 1,690 final grains, 10,736,115 trial
orientations and their scores, grain archive arrays, and 1,691 grain/spot text
files matched the independent offline reference bitwise. The sampled process
tree RSS peak was 2.11 GiB, with live role allowances totaling 7 GiB. The final
analysis took another 312.7 seconds after acquisition; that work is outside
the frame-ingest rate. No full streamed Sven fit or new master comparison was
performed for this measurement; its timing probes retained original float64
pixel values. Production fitting code was unchanged.

The bounded decoder/map/spool pool and incremental maps now implement the main
representation improvements. Production work still needs longer raw Eiger 16M
captures, measured durable HDD throughput, and separate discovery/fit latency
budgets. Replay uses lossless backpressure; that does not establish how a physical
detector behaves when a consumer falls behind.

A subsequent [HEXRD core investigation](fit-grains-realtime-opportunities.md)
prototyped direct typed mapping below the 10 ms/frame budget on both datasets,
with bitwise map parity. It also identified a faster exact integer sparse route.
These kernels are now integrated in the ordinary offline map builder and the
opt-in incremental receiver. Their timings exclude decoding and fitting.

### Integrated native maps and offline benefit

The installed native kernel maps decoded Inconel frames in **4.90 ms mean /
5.48 ms P95**, and Sven frames in **0.91 / 1.14 ms** (48 and 24 real frames).
The complete public incremental append, including masking, panel merge and
storage, measured **5.67 / 6.21 ms** for Inconel and **1.19 / 1.51 ms** for Sven.
Source reads, wire decoding, static setup and snapshot creation are excluded.
Sven retains its original fractional float64 values.

The ordinary offline `build_eta_omega_maps()` automatically selects the same
dense native kernel or exact integer sparse dispatch; no streaming package or
new command option is required. Three alternating before/after runs included
source reads and per-call geometry setup, with one worker:

| Offline map build | Previous median | Current median | Speedup |
|---|---:|---:|---:|
| Inconel, all 1,440 frames | 14.33 s | 1.89 s | **7.58×** |
| Sven, first 24 frames from original FCH5 | 1.88 s | 1.82 s | **1.03×** |

All map values and axes matched bitwise in all six runs per dataset. These
measurements exclude material/config construction, orientation scoring and
grain fitting; they are not cold-disk benchmarks. Sven's already-sparse route
shows little benefit when reads and setup dominate. Fitting already-known
grains does not build maps and receives no speedup from this work. The offline
benchmark process peaked at **3.01 GiB RSS** under a 4 GiB allocation ceiling.

Before the bounded worker pool, the complete incremental receiver sustained
**13.27 Hz**, below the earlier 15.10 Hz receiver without map updates. The new
150.33 Hz result supersedes that bottleneck. Incremental maps remain opt-in.
The earlier full replay also produced bitwise-equal
1,690-grain outputs, every source pixel and omega edge, and all 1,691 text files.
Aggregate sampled RSS peaked at **2.82 GiB**. Its final analysis took 280.0 s;
that single observation is not a controlled total-workflow speed comparison.

Eighteen example comparisons (nine studies, both fitting modes) and a four-grain
Sven sample also matched master `0a041b17` exactly, including spot data and
example spot files. The Sven master comparison used a bounded original-pixel
reader because eager full-scan loading exceeds the laptop budget. The core and
sandbox checks at that stage passed **439 tests**; two large reference-map checks also passed
in the preceding integration validation.
A full high-grain-count Sven streamed fit remains unvalidated.

For integrations, construct `EtaOmegaMapPlan(experiment, plane_data)` before
capture, then `EtaOmegaMapAccumulator(plan, capacity, max_size_mb=...,
filename=...)`. Call `append({panel_name: decoded_frame}, omega_edges_degrees)`
with every planned panel for each ascending interval. `snapshot()` returns an
unfiltered readonly acquired prefix, which can be supplied as `maps=` to
`find_orientations()` on an experiment describing that same prefix. Keep geometry
and crystallography fixed; the map storage budget excludes geometry and images.
`append_histograms()` accepts outputs of that same plan for ordered commits after
parallel calculation; it applies the usual panel merge and axis validation.

The same native CSR packer is used by generic offline image adapters and lazy
fitting views. On 48 Inconel and 24 Sven frames, conversion alone improved from
**26.51 to 3.15 ms (8.43×)** and **227.56 to 95.20 ms (2.39×)** respectively.
All CSR buffers matched SciPy bitwise, including Sven's fractional float64 data.
Unsupported layouts/dtypes retain SciPy semantics. Existing sparse FCH5 input
continues to avoid dense conversion; these are not total fit-grains speedups.

## Persistent analysis and archive overlap

Priorities 1 and 2 of the [complete-processing strategy](fit-grains-realtime-strategy.md)
are implemented. `receive` owns one analysis service, which keeps calibration,
crystallography and fitting processes alive across immutable snapshot requests.
Final requests cancel stale work cooperatively and drain it before starting a
new generation. Cancellation of the run terminates the service's process group.

Small orientation/grain files are published first, preserving the exact text
round trip used to initialize the reference fit. One archive thread compresses
the trial archive while fitting proceeds. Writer failures prevent completion;
archives are atomically installed and cancellation is checked between chunks.
Normal offline orientation export shares the optimized, atomic archive writer.

A sequential paired run on the same 1,440-frame, 1,690-grain Inconel input compared
frozen pre-change Python with the new implementation and the same native binary:

| Measurement | Previous prototype | Priorities 1 + 2 |
|---|---:|---:|
| Analysis process wall time, including startup | 265.69 s | **147.91 s (1.80×)** |
| Analysis function elapsed | 263.39 s | **145.26 s (1.81×)** |
| Discovery | 32.44 s | 18.09 s |
| Fitting | 214.98 s | 124.94 s |
| Archive wait after fitting | — | Under 0.1 ms |

These are single paired observations, not repeated medians or a master speed
comparison. Every final array and all 1,691 grain/spot text files match both the
frozen prototype and the independent offline reference bitwise. The inferred
pre-change export/setup interval was 13.64 seconds; it is not a compression-only
profile or an isolated measurement of archive overlap's speedup.

A fresh live replay held **149.44 Hz**, reused the same service and fitting
processes across the provisional/final handoff, and completed final analysis in
129.64 seconds. Sampled aggregate RSS peaked at **3.69 GiB**. Its outputs also
matched offline bitwise. The different timing from the standalone paired run
does not isolate the benefit of warm workers. Scientific processing still lags
100 Hz; this change narrows that gap rather than completing the realtime goal.

`--analysis-workers` accepts 1 or 2 (default 2); `--discovery-workers` accepts
1–8 (default 4). Metadata for large FCH5 inputs selects one larger fitting
worker so the existing shared-window extraction path can retain a 512 MiB
workspace and four native readers. Explicit staged mode and single-grain jobs
also use one larger worker. Other inputs use 128 MiB caches per worker.
The larger worker honors `fit_grains.image_cache.read_workers` (1–8, default 4);
smaller workers retain one reader within their allocation allowance.
This avoids converting a memory limit into repeated single-grain FCH5 scans.
The four-grain Sven sample retains exact master fits and spot files; its final
grouped fitting stage took 13.24 seconds, 16.61 seconds including pool startup.
All **18 example comparisons** (nine studies, both measurement modes) also match
master through the new persistent pool. The expanded regression suite passes
**498 tests**, including archive overlap/failure, cancellation, process-group
cleanup and 2,000-grain scheduler grouping. The 2,000-grain check exercises
scheduling; the full scientific replay contains 1,690 grains.

A follow-up Sven check on 128 real grains compared four and eight readers with
the same original pixels, 512 MiB workspace and 3 GiB hard process limit. Fitting,
setup and spot output took **93.39 versus 83.02 seconds (1.12×)** in one sequential
pair, with about **0.99 GiB** peak RSS in both. All fit arrays, returned spot data
and 128 spot files matched exactly. Grain results and text also match the first
128 entries of the earlier complete 2,103-grain reference. A 13-grain run through
the persistent pool additionally verified the configured eight-reader path
under its smaller 1,792 MiB worker limit.

This is optional concurrency tuning, not a full Sven speedup or 100 Hz result.
It does not change the default reader count or the live allocation budget.
Removing pixel-cache compression provided no meaningful gain in repeated
13-grain checks, so compression remains in place. The next substantial work is
sharing frame/window work across active grains and avoiding repeated discovery;
the current staged implementation still visits every requested frame in each
tolerance pass.

## Exact sparse extraction and geometry snapshots

A further core change avoids two forms of redundant work. Fitting snapshots
read detector geometry directly instead of serializing detector configurations
and copying their pixel masks. Native sparse extraction now handles guarded
integer and half-integer floating-point counts and applies panel masks before
labelling, avoiding dense polar-window allocation where possible. Other
intensities retain the existing native labeller and NumPy moment/reduction
arithmetic. Solver settings, component selection and scientific tolerances are
unchanged. The changes remove **37 runtime lines: 20 Python and 17 C++**.

One fresh sequential pair used all **2,103 Sven estimates**, the same existing
threshold-specific support cache and one fitting process on each side:

| Measurement | Frozen previous core | Current core |
|---|---:|---:|
| Fitting call, including image loading and spot output | 209.91 s | **149.80 s (1.40×)** |
| Peak process RSS | 3.84 GiB | 3.84 GiB |
| Hard process address-space limit | 6 GiB | 6 GiB |

This diagnostic isolates repeated fitting work after support preparation. It
**does not establish a full raw-input Sven speedup or 100 Hz**: preparing that
experimental cache previously took 648.52 seconds, and orientation discovery is
also outside this comparison. The original 42 GB input remains untouched; this
cache is not enabled automatically. These are single paired timings against
the previous core, not master performance measurements.

All 2,103 fitted grain records, spot files, `grains.out` and `grains.npz` arrays
match exactly, including comparison with the earlier complete original-input
reference. A fresh four-grain run on original Sven pixels matches independent
master fits and returned spot data. All 18 example comparisons against master
also pass, as do **762 regression tests**. Two large reference-map tests remain
covered by the earlier separately guarded run. A dedicated 1 GiB guard verifies
that a large, mostly empty half-count window stays sparse rather than creating
a 1.3–2.6 GB temporary pixel array.

The unchanged 512 MiB workspace also completed 128 original-input grains under
the live worker's 1,792 MiB process ceiling, with identical fits and spots. A
768 MiB workspace experiment hit that allocation guard and was rejected; it
did not change the production memory allowance. Smaller geometry-index types
showed no meaningful gain or reduction in pixel rereads and were also left out.

Two experiments were not retained. Fitting threads stopped helping beyond four
on the 128-grain diagnostic; eight and sixteen threads were slower. Replacing
the general native labeller with SciPy also slowed sparse floating-point cases,
so the native implementation remains. Repeated reference assertions in its
tests were consolidated while retaining and expanding the cases checked.

A separate 24-frame original-input prototype sampled only the 378,525 detector
pixels needed by the ring histograms. Its histograms match full-frame reads
bitwise, but it achieved only 9.38 Hz including source-index setup; typical
frame sampling still took roughly 25–30 ms. Frame-level parallelism and shared
index construction need testing before this could replace the offline reader.
Its geometry setup and validation reads are excluded, so this is not an
end-to-end timing. Artifacts and rejected experiments are recorded under
`.local-bench/sven-frame-work` and in the
[validation record](live-fit-grains-rate-validation.json).

## Single-pass sparse preparation

Dense-to-CSR conversion now writes directly into typed output buffers and
shrinks them before returning. Under allocation pressure it first counts the
nonzero pixels and retries with exact-size destinations. This removes vector
growth and the final copies; the allocation safeguard adds 16 runtime lines
versus the previous packer. Pixel selection, dtype, ordering and exceptional values
retain SciPy's exact representation; offline and live callers share this kernel.

A fresh pair sampled 24 real frames from each dataset:

| Conversion only | Previous kernel | Current kernel | Speedup |
|---|---:|---:|---:|
| Sven, 4362×4148 float64 | 82.10 ms/frame | **37.18 ms/frame** | **2.21×** |
| Inconel, 3500×3500 int32 | 4.07 ms/frame | **3.15 ms/frame** | **1.29×** |

Every CSR buffer matches SciPy and the previous kernel byte-for-byte. These
serial probes have a 3 GiB allocation cap. An isolated threaded probe combining
conversion and exact ring histograms reached 85.03 Hz with four threads,
108.49 Hz with six and 115.63 Hz with eight. It reuses four immutable real Sven
images, before adding the allocation safeguard; source reads, wire decoding,
persistence, discovery and fitting are excluded. Validation runs outside compute timing. Its separate 4 GiB allocation
cap and peak RSS below 1.84 GiB do **not** establish the live service's aggregate
budget or complete scientific throughput; production worker counts and role
limits are unchanged. Regression coverage passes 765 tests, including NaN
payloads and empty outputs. All 18 freshly rerun example comparisons match
master (`0a041b17`), including spot files, in both measurement modes.

Lossless geometry compression showed no useful fitting gain. It and a doubled
pixel-query batch failed the live worker's 1,792 MiB allocation guard; both were
left out. Always-two-pass compaction and stored SIMD occupancy masks were rejected
because they slowed sparse inputs. The guarded complete Inconel replay captured 1,440 frames at 150.32 Hz and
finished final analysis in 128.13 seconds. All scientific arrays, 1,691 text
files, captured pixels and omega edges match the offline reference exactly.
Peak aggregate process RSS was 3.75 GiB. This remains below the required
complete scientific throughput. Detailed timings, exclusions and artifacts
are in the [validation record](live-fit-grains-rate-validation.json).

## Read-only shared fitting experiment

A separate diagnostic stores an existing Sven support cache in one read-only
shared-memory mapping and uses four fitting processes. All 2,103 grains fit in
**33.04 seconds**, with all grain fields and spot files identical to the earlier
canonical reference. Building the shared store adds **21.94 seconds**, for
54.98 seconds total. Its earlier 648.52-second threshold-cache preparation and
orientation discovery are excluded; this is **not complete 100 Hz processing**.

An independent run of master (`0a041b17`) now checks the full 2,103-grain set:
every fitted field and all 2,103 spot files match byte-for-byte. Master uses
unchanged extraction and optimization with the same shared canonical support;
a four-grain comparison also verifies the adapter against master reading the
original pixels. Both shared runs use four workers under the same allocation
ceiling. Master fitting takes 1,039.65 seconds, plus 22.26 seconds of store setup.
These timings exclude the earlier support preparation and orientation discovery.

The mapping occupies 3,412,071,240 bytes. Five processes, including the
coordinator, each have at most 981,885,579 additional bytes of address space.
Even if every other inherited page became private, the total allocation ceiling
is **7.75 GiB**. Preparation runs alone under a separate 6 GiB cap. The measured
runs write all spot files and return grain records; returning all spot payloads
exhausted the prototype coordinator's allocation allowance during serialization.
The experiment remains outside production: its setup overhead and input-specific
support preparation must be addressed before live adoption. Artifacts reside
in `.local-bench/shared-fit`.

The next compact-store experiment reduces shared storage to 1,898,205,184 bytes:
3,098 frames store exact float32 values, and 502 retain float64. Six workers fit
all 2,103 grains in **22.48 seconds**, plus **21.57 seconds** to build the store.
Every fitted field and spot file matches the independent master run byte-for-byte.
The aggregate allocation ceiling remains **7.75 GiB**. This prototype still uses
the previously prepared support cache; its 648.52-second preparation and discovery
are excluded. The native view supports these compact buffers; live integration
is being measured separately.

## Discovery and incoming-frame experiments

Sven's configured static grid contains 79,828,799 quaternions. It can be
prepared before collection from material/configuration alone; preparation takes
148.96 seconds and opens no images or grain estimate.

**Fresh master discovery produces 15,559 grains, not 2,103.** The older saved
Sven maps zero-fill detector gaps, whereas current master (`0a041b17`) uses NaN.
Finite intensities agree, but those gaps change scoring denominators and grain
selection. The earlier 2,103-grain fitting experiments remain valid for their
supplied initial parameters; they do not validate fresh complete-scan discovery.

The independent fresh reference builds all 3,600 maps from the original FCH5
using unmodified master. That single-worker build takes 716.00 seconds under a
4.5 GiB allocation limit. Streaming maps match its first 600 frames bitwise.

| Fresh-map discovery, eight workers | Master | Current branch |
|---|---:|---:|
| Score all 79,828,799 trials | 13.54 s | 13.93 s |
| Cluster and save score array | 211.59 s | 6.48 s |
| Combined measured stages | 225.13 s | **20.41 s (11.03×)** |

Every score and orientation matches master bitwise, and the complete initial
`grains.out` is byte-identical. Map construction, static grid preparation,
trial-archive compression, fitting and process startup are excluded. These are
single sequential observations, not repeated medians.

Within clustering, symmetry images are now computed once per candidate, and
workers fill disjoint portions of one condensed distance array. Arithmetic,
SciPy linkage, centroid averaging and ordering are preserved. All 15,582 input
centroids merge into the same 15,559 master orientations: the previous native
implementation takes 11.48 seconds, cached serial evaluation 7.04 seconds,
eight workers **2.64 seconds (4.35×)**, and sixteen workers 2.67 seconds.
Both centroid-averaging passes also group labels once with a stable sort,
instead of rescanning all candidates for each grain. Cluster members retain
the same order. Tests cover exact scalar distances across all 11 Laue groups, parallel merges,
context propagation, output-buffer validation and cancellation. This shared
code benefits offline discovery as well as the sandbox; fitting supplied grains
does not perform this stage.

A 24-frame receiver probe reaches **123.29 Hz** without the fitting index, or
**93.14 Hz** including it. It includes checksum validation, bitshuffle decoding,
exact scaled maps, lossless wire storage and ordered commit, with full
3,600-frame map capacity. Geometry setup, discovery, fitting, fixture encoding,
parity checks and durable disk sync are excluded. A longer 600-frame probe that
also copies the raw archive measures only **54.20 Hz**. Neither proves sustained
complete-science 100 Hz. Wire-spool modes remain experimental.

Releasing the dense angle cache after copying immutable histogram geometry
frees about 276 MiB. Geometry setup uses a 3 GiB allocation limit, then reception
lowers it to 2 GiB; all short-probe roles total less than 4 GiB.

For full Inconel reception, retaining raw wire messages alone regresses final
analysis to 225.00 seconds because fitting repeatedly decodes full images.
Adding the compact exact fitting index restores final analysis to **120.83
seconds**, including **105.80 seconds fitting**, while capture sustains
**135.84 Hz**. All pixels, omega edges, scientific arrays and 1,691 text files
match the independent offline reference. Peak aggregate RSS is 3.52 GiB under
a 7.875 GiB aggregate allocation bound including the test supervisor.

`publish --wire-input DIR` replays preencoded detector messages without opening
scientific inputs. `receive --raw-archive DIR` can experimentally reuse a matching
wire ZIP archive while storing only the fitting index locally. Every arrived
message is checked against that archive, snapshots expose only committed frames,
and subsequent raw reads reject changed content. This avoids a second raw-data
write but requires an archive supplied by the integration. It is opt-in; this
POC does not yet adapt Eiger FileWriter HDF5. Default reception retains its own
complete raw archive. With archive reuse, a 600-frame replay reaches **100.46 Hz**
and passes all pixel/support/map comparisons. A full 3,600-frame capture measures
**111.43 Hz** and passes the same complete pixel, fitting-support and fresh-master
map comparisons. Both exclude
discovery and fitting, so complete scientific 100 Hz remains unproven.

The latest completed regression run passes **915 tests**, and all 18 fresh example
comparisons (nine studies, both fitting modes) match master bitwise, including
spot files. Full Sven fitting on the newly
discovered 15,559 grains and sustained complete scientific 100 Hz remain
unvalidated. Artifacts are under `.local-bench/wire-spool`.

## Memory and disk

Linux allocation guards are installed before numerical imports. The publisher
has 1.5 GiB and the incremental receiver 2 GiB. Analysis reserves 2.5 GiB for
the coordinator, including its archive thread, plus 1.75 GiB total for fitting
processes: two at 896 MiB, or one at 1,792 MiB. These live roles total **7.75 GiB**,
below the laptop's 8 GiB ceiling. The receiver without incremental maps uses
1.5 GiB instead.
During static map setup, before analysis or fitting workers exist, the receiver
may use 3 GiB; it releases the dense angle cache and lowers its limit to 2 GiB
before connecting. Together with the publisher, that setup phase is bounded by
4.5 GiB. Phase allowances are never added as concurrently active roles.
Decode workers are threads inside the receiver's allowance, with at most one
in-flight frame per worker. `--decode-workers` accepts 1–8, default 4; reducing
it reduces temporary frame workspace. Kernel allocation limits remain in force.
The receiver retains a 4 GiB hard ceiling solely so its child can raise its own
soft allowance after exec. BLAS, OpenMP and Numba each use one numerical thread;
analysis parallelism uses the explicitly bounded workers. Do not run an additional
analysis alongside the live pair.

The image caches are budgeted separately as described above. ZeroMQ high-water marks are
one message per socket, and frames/messages are limited to 256 MiB. Input is
read one frame at a time; decoded full scans are never retained. Disk usage grows
with acquired sparse pixel support. There is no deletion policy or crash-resume
facility in this POC. Linux filesystem cache is reclaimable and separate from
process allocation limits.

## Eiger compatibility and limits

The wire format follows DECTRIS's documented [SIMPLON legacy Stream V1
interface](https://www.dectris.com/web/content/317616): PUSH/PULL on port 9999,
multipart `dheader-1.0`, four-part `dimage-1.0` plus optional image appendix,
and `dseries_end-1.0`. The replay uses raw-block LZ4, uint32 counts, and image
appendices containing exact omega edges and original source dtype. Real streams
can instead supply `omega_start`/`omega_increment` in the detector header. The
receiver also decodes uncompressed and bitshuffle-LZ4 unsigned images using the
existing DECTRIS decoder. It validates MD5 when present.

Use one PULL consumer per detector stream. Additional consumers divide messages
between themselves rather than receiving copies; integrate behind the existing
receiver if the private system already owns the detector connection.

The optional appendix is application metadata; a private integration must wire
in synchronized motor angles if header increments do not describe actual motion.
The supplied HEXRD calibration supplies detector masks and geometry. Flatfield
and pixel-mask header blobs are not substituted for that calibration. The
emulator provides streaming data, not the detector's HTTP control API.

This POC supports one physical detector, ascending omega, a single-study YAML,
and Stream V1. [Stream V2](https://github.com/dectris/documentation/tree/main/stream_v2)
uses CBOR and port 31001 and needs a separate transport adapter.

Replay refuses negative, nonfinite or out-of-range pixels. Explicit
`publish --source-scale .5` losslessly represents processed half counts as
unsigned wire counts and restores them before scientific computation. Other
fractional values are rejected; values are never rounded or clipped. No dark, median, or other preprocessing is invented: parity requires
the same input pixels and calibration as the offline reference. Full-scan-derived
preprocessing of raw detector counts would need a final pass over retained data.

This establishes concurrent acquisition and scientific parity, not a guarantee
of detector-rate analysis. Prefix discovery and fitting are recomputed and can
lag acquisition; map rebuilding is avoided with the opt-in incremental route.
Next targets are bounded decode/map/spool workers, reuse of measured reflection
support, and asynchronous scoring and fitting, followed by the same full final pass.
