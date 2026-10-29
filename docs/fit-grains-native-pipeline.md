# Native staged measurement pipeline

The staged fit-grains path now executes its window measurement loop, sparse
frame sampling and source-pixel cache queries in C++. Python retains workflow
configuration, Textual, cache admission and the existing optimizer interface.
This is a measured extension of `7c090fa6`, preserving scientific settings and
the original peak-statistics and fitting arithmetic.

## Native execution

- The window coordinator assembles frame requests, scatters pixels into their
  original omega slots, tracks window completion and calls the existing native
  peak-statistics kernel directly. NumPy's native reductions and BLAS dispatch
  remain unchanged, preserving the reference rounding order.
- A frame reader plans required coordinate and intensity chunks, validates
  canonical sparse rows, decodes in parallel and gathers exact source pixels
  inside one native call. There are no Python futures or callbacks per decoded
  chunk. Only the coordinator obtains HDF5 metadata; workers use positional
  reads and the existing Blosc context API.
- Native cache queries pack integer coordinates, deduplicate requests, find
  cached pixels and restore the original request order. Keys remain integers
  above the precision limit of floating point, and values are copied by bytes.
  The existing byte-bounded Python catalogue retains ownership and admission.

Compatible NumPy buffers are borrowed across the boundary. Compact window
coordinates are read directly, avoiding an int64 conversion for every patch.
No new numerical, image-processing or codec dependency is introduced.

The previous Python paths remain available for older extensions, unsupported
formats/dtypes/layouts and requests exceeding native scratch capacity.
Noncanonical COO data still uses the original CSR
conversion and coalescing order.

Large frames are processed in tiles of at most 65,536 requested samples. A
window can span tiles and omega frames; statistics are computed only after
every original pixel slot has been filled. Source frames that need full-frame
fallback are borrowed across all tiles, including frames larger than the cache.

This removes an important performance cliff in the previous code: a frame
request exceeding 65,536 samples bypassed partial pixel sampling and decoded
the entire source frame. The unmodified starting branch's 52-grain run exceeded
a six-minute process timeout, even though the smaller subsets completed quickly.

## Measurements

Three alternating before/after pairs for each smaller Sven subset, with all
three tolerance passes, refitting, returned spots and spot-file output:

| Grains | Starting branch median | Native median | Speedup |
| --- | ---: | ---: | ---: |
| 4 | 17.43 s | 14.32 s | 1.22× |
| 13 | 32.71 s | 27.84 s | 1.17× |

The unmodified starting branch's **52-grain process did not finish within
360 seconds**. The native version completed in **96.12 seconds for the whole
process**, including **93.29 seconds in fit-grains**. This establishes a
**greater than 3.7× whole-process speedup** for that single larger case; the
aborted baseline has no completed fit timing.

For an independent 52-grain output reference, the starting branch was also
run with only its internal query cutoff raised from 65,536 to 1,048,576 samples,
under the same hard memory ceiling. Its scientific settings, Python measurement
code and original kernels were retained. That reference completed fitting in
91.42 seconds, and its fits, extracted spots and all **52 spot files** match
the native implementation exactly. It is a correctness reference, rather than
the unmodified performance baseline.

The cutoff-raised Python and native large-case times are similar: most of the
large-case gain comes from avoiding full-frame decoding. The native pipeline
also removes Python dispatch and keeps individual requests at 65,536 samples.
Measured native fitting RSS stays **below 1.12 GiB** across the three workload sizes.

## Memory and cancellation

All large Sven runs use one fitting process, single-threaded BLAS, direct FCH5
reads and a **4 GiB hard address-space ceiling**, with no image scratch files.
Runs are sequential. The preparation workspace is 512 MiB.

Native decoder buffers and cache-query temporaries are charged to the existing
cache reservation. A frame call reserves at most 16 MiB for its private query
and coordinate chunks; intensity chunks are consumed in batches of at most the
configured worker count. There is no allocation of whole decoded image stacks.

A shared limiter bounds simultaneous positional reads and codec operations
across native frame jobs and row-index lookahead. Four configured readers do
not become eight simultaneous decoders when both queues have work. The C++
pool is persistent and is closed before image file descriptors are released.

Native waits check cancellation every 25 ms. Cancellation and decoder errors
cancel pending jobs and drain running jobs while the GIL is released before
buffers or file descriptors can be freed. Each active read retains the one-MiB
decoded-chunk bound. An operating-system read must return before its job can
finish cancelling. The Textual cancel button and Ctrl+C are tested during real
Sven reads.

## Validation and configuration

The focused suite passes **377 tests** under a 1.5 GiB allocation ceiling. It
checks exact pixel bits across ten native intensity dtypes,
NaNs/infinities/signed zero, duplicate and reordered requests, large integer
keys and file positions, partial codec blocks, malformed sparse data, native
fallback preflight, corrupt chunks and cancellation/reuse. Existing memory,
geometry, centroid, fit and dashboard checks remain in the selection.
Tiling checks include individual windows larger than a tile, multiple windows
sharing a frame, original omega order, and borrowing a fallback frame larger
than the cache exactly once.

All nine example studies are compared with unmodified master in both staged
and grain modes, without numerical tolerance. Repeated four-grain Sven fits
and returned spots are compared with the master scientific algorithms using
bounded pixel input; eager master loading exceeds local RAM. Spot files are
compared byte for byte. Larger Sven comparisons use the starting branch.

Rebuild the transforms extension for this checkout. Merge these settings into
the existing fit-grains block, retaining its scientific parameters:

```yaml
fit_grains:
  measurement_mode: staged
  image_cache:
    mode: lazy
    max_size_mb: 512
    read_workers: 4
    reuse_pixels: true
```

The cache setting is a workspace allowance, rather than a total-process memory
limit. The benchmark command separately enforces the hard address-space limit.
Full 920-grain throughput and remote HDD scaling remain unmeasured.

Raw timings, memory measurements, source hashes, reference modifications and
exact-comparison results are recorded in
[the results JSON](fit-grains-native-pipeline-results.json).
