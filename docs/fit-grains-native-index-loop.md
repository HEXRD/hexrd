# Less Python dispatch during staged fit-grains

This change builds on `51b6a370`. It removes repeated coordinate preparation
and per-chunk Python dispatch from the existing staged measurement pipeline.

The subsequent [native measurement pipeline](fit-grains-native-pipeline.md)
moves the window loop and pixel queries into C++ and removes a large-frame
performance cliff with bounded query tiles.

Each planned patch now clips its coordinates and applies the detector ROI once,
then reuses compact, immutable source coordinates across all its omega frames.
The original window signature remains available for exact replay validation.
Preparation is idempotent when a memory-limited group defers a plan to the next
group, including ROIs whose source coordinates require a wider integer dtype.

FCH5 row-index workers now perform a frame's positional reads, existing Blosc
decoding and existing SIMD index construction inside one C++ call. The loop
releases the GIL, so workers no longer reacquire it between every compressed
chunk. It uses the existing decoder buffers, compact indexes and bounded queue;
it allocates no whole-frame coordinate or intensity array. Older native
extensions retain the per-chunk Python fallback.

An atomic cancellation token is checked between chunk reads and before the
native call returns. Queue shutdown sets this token before cancelling and
joining jobs, and joins jobs before their file descriptors can close. Each read
retains the existing one-MiB decoded-chunk bound. Cancellation still depends on
the current operating-system read and codec call returning.

No peak thresholds, angular windows, connected-component operations, centroid
arithmetic, tolerance passes or optimizer work change. No image scratch files
or persistent image cache are introduced.

## Measurements

Three alternating before/after pairs for each Sven subset, including all three
tolerance passes, refitting, returned spot records and written spot files:

| Grains | Before median (`51b6a370`) | After median | Speedup |
| --- | ---: | ---: | ---: |
| 4 | 19.65 s | 17.53 s | 1.12× |
| 13 | 36.21 s | 33.81 s | 1.07× |

Peak fitting RSS stays below 1.1 GiB. These are additional gains over the
preceding branch. The larger workload spends less of its time on row-index
dispatch, so this change has a smaller effect there. Compressed-data reads and
decoder coordination remain substantial costs. Extra cache sorting was also
inspected, but its measured contribution was small and it was left unchanged.

An initial 13-grain baseline took 52.18 seconds. Repeated measurements with
both native extensions rebuilt gave the more conservative results above;
the initial outlier is not used in the reported speedup.

## Validation

The focused regression suite passes 338 tests under a 1.5 GiB allocation ceiling.
It includes batched native reads across partial chunks, invalid row ordering,
unsigned overflow, empty frames, cancellation, output and input bounds, and
coordinate preparation after a rejected group. Existing format, memory,
centroid, fit and dashboard checks remain in the selection.

All nine example studies are compared with unmodified master in both staged
and grain modes, without a numerical tolerance. The Sven four-grain fits and
returned spot records are compared with the master scientific algorithms using
bounded pixel input; eager master loading would exceed local RAM. Written spot
files are compared byte for byte. The 13-grain runs are compared with the
preceding branch commit, including all written spot files.

Both the Textual cancel button and Ctrl+C are exercised during real Sven FCH5
reads. Request-to-cleanup latency and simulated UI input/dispatch latency are
recorded separately: 4.6 ms and 3.17 s for the button, and 2.1 ms and 0.17 s for
Ctrl+C, respectively. Cleanup measurements start when the application receives
the cancellation request, rather than when simulated input begins.

## Reproduction and scope

Rebuild the native transforms extension. These optimizations apply to
`fit_grains.measurement_mode: staged`; compatible FCH5 inputs use the native
index loop, and other formats also benefit from preparing coordinates once.
The existing [staged configuration and memory guidance](fit-grains-pixel-reuse.md#configuration-and-memory)
still applies.

Timing comparisons use the preceding branch commit, rather than unmodified
master. Both native extensions are built with the repository's normal optimization
flags. Runs use one fitting process, four bounded decoder workers,
single-threaded BLAS, a 512 MiB staged workspace, direct FCH5 reads and a
4 GiB hard address-space ceiling. No two large benchmarks run concurrently.
Full 920-grain throughput and remote HDD scaling remain unmeasured.

Raw timings, memory measurements, source hashes and exact-comparison results
are recorded in [the results JSON](fit-grains-native-index-loop-results.json).
