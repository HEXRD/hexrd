# Reusing exact pixels and feeding the first fitting pass

This builds on `f33fd58f`, the previous bounded parallel decoder. It retains
master's grain-dependent windows, resampling order, pixel intensities, peak
selection and optimizer. It does not replace those windows with global peak
centroids.

## What was still expensive

The previous implementation decoded complete storage chunks when only some of
their internal compression blocks were needed. A 250-frame Sven sample requested
844 column blocks and 1,360 intensity blocks, while selecting 472 column chunks
and 831 intensity chunks, respectively. Each of these chunks contains four
Blosc blocks. Most decoded values were never used.

It also read the same source pixels again during narrower tolerance passes,
discarded useful boundary chunks between frames, and queued row-coordinate
chunks individually before constructing their compact row indexes in the
coordinator. The first tolerance pass became the dominant remaining cost.

## Changes

1. **Decode the requested support.** The existing Blosc slice API decompresses
   only the blocks covering requested coordinate spans and intensity positions.
   A broad request uses the full decoder. Partially initialized workspaces stay
   private to their query and never enter the cache of complete chunks.
2. **Retain exact source pixels.** A bounded catalogue maps detector reader,
   source frame and integer row/column coordinates to their original sampled
   values. A changed window reuses matching coordinates and reads every missing
   pixel from the source. It retains zeros, NaNs, infinities and original dtypes.
   Duplicate sampling coordinates are read once and expanded back into the
   original window order before peak statistics are calculated.
3. **Visit frames in stored order.** Frames follow their offsets in the FCH5
   coordinate table. Decoder chunks can survive adjacent frames under the same
   LRU byte bound. Live window memory is estimated in that actual visit order,
   including wrapped omega windows, rather than assuming omega order.
4. **Reduce Python/native transitions.** One C++ operation performs a positional
   read, validates its buffer and invokes the existing Blosc codec while the GIL
   is released. It retains the codec library and input/output buffers throughout
   the call. It does not access HDF5 or the TUI from a worker.
5. **Read ahead compact row indexes.** A bounded queue constructs indexes for a
   few source frames, batching their row-coordinate chunks inside each job.
   Workers reuse their decoder buffers and return compact indexes rather than
   whole decoded row arrays. This reduces queueing and coordination overhead.

The decoder uses the public [C-Blosc API](https://github.com/Blosc/c-blosc/blob/main/blosc/blosc.h),
including its validated buffer and slice operations. The native binding releases
the GIL explicitly; see the [pybind11 threading guidance](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil).
Centroid arithmetic and fitting calculations are unchanged.

## Configuration and memory

These improvements apply to compatible large FCH5 inputs in the existing staged
mode. Merge these fields into the existing configuration, keeping its scientific
settings:

```yaml
fit_grains:
  measurement_mode: staged
  image_cache:
    mode: lazy
    max_size_mb: 512
    read_workers: 4
    reuse_pixels: true
```

`reuse_pixels` defaults to true in staged mode. Set it to false to disable the
source-pixel catalogue. Other image formats retain their existing readers.
Build the native transforms extension for this checkout: an older extension
uses the compatible Python decoder fallback and cannot provide frame-index
read-ahead.

The preparation workspace remains capped at 512 MiB. Its source-chunk cache is
at most 128 MiB, including decoder reservations and compact indexes. The pixel
catalogue gets only the remaining allowance after planned geometry and peak
live window support are accounted for, with additional headroom. Its limit
shrinks when a group needs more workspace. A scan larger than the catalogue
retains a useful subset of frames rather than continually admitting new frames
and evicting the entire previous pass. Any miss follows the original reader.

No image staging files or disk scratch cache are created. Catalogue data and
pending index work are cleared on completion, cancellation and failure.

These workspace limits are not total-process or OS RAM limits. Every local Sven
experiment uses one process with a **4 GiB hard address-space ceiling** and direct
FCH5 reads to bypass filesystem caching of the large input. BLAS uses one thread.
This preserves headroom below the user's 8 GiB ceiling. See the
[memory guide](fit-grains-memory.md) for the distinction between cache allowances
and a process limit.

## Validation and timings

Final alternating timing pairs and exact comparisons are recorded in
[the results JSON](fit-grains-pixel-reuse-results.json). The timing baseline is
`f33fd58f`, rather than unmodified master. Each run includes the first four Sven
grains, all three tolerance passes, refitting, returned spot data and written
spot files. There is no pre-populated pixel catalogue between runs.

Three alternating pairs without profiling gave:

| Measurement | Before (`f33fd58f`) | After |
| --- | ---: | ---: |
| Four-grain runs | 48.33, 45.12, 45.01 s | 21.24, 21.54, 21.25 s |
| Median | 45.12 s | 21.25 s |
| Peak process RSS | 0.81 GiB | 0.88 GiB |

This is **2.12× faster** than the preceding commit. A separate 13-grain pair
improved from **71.93 to 36.14 seconds (1.99×)**, with peak RSS increasing from
0.81 to **1.06 GiB**, still below the 4 GiB allocation ceiling.

All three four-grain repetitions reproduce the master-reference fits and spot
records exactly, including byte-identical spot files. The 13-grain result and
all 13 spot files match the starting branch exactly. All nine examples match
unmodified master in both `staged` and `grain` modes, for 18 fit/spot comparisons.
The focused regression suite passes **298 tests** under a 1.5 GiB allocation
ceiling.

Both Textual controls also pass during real Sven reads. Cleanup took 5 ms after
the button cancellation event and 7 ms after Ctrl+C. Headless simulated input
and dispatch took 3.19 s and 0.19 s respectively and are recorded separately;
these are not end-to-end hardware input latency measurements. The cancellation
runs used the same 4 GiB ceiling and peaked at 1.04 GiB RSS.

Exploratory single runs, used to choose the implementation, took:

| Change accumulated | Four-grain time |
| --- | ---: |
| Selected compression blocks | 41.81 s |
| Exact source-pixel reuse | 31.88 s |
| Stored frame order and deduplicated source queries | 29.65 s |
| Native positional read and codec call | 27.49 s |
| Batched row-index read-ahead | 21.41 s |

In the last exploratory run, the measurement groups took 18.32, 1.59 and 0.66
seconds. The first pass retained about 63 MiB of source pixels. Approximately
94% of unique second-pass source requests and 98% of third-pass requests reused
those exact values.

The master reference for Sven uses its original scientific algorithms and
separately compiled transforms, with bounded pixel input because eager loading
exceeds local RAM. Example references use unmodified master. Scientific
comparisons use no numerical tolerance; written Sven spot files are compared
byte for byte.

Index and decoder queues are bounded and check cancellation while submitting
and waiting. Index workers check a cancellation event between chunk reads.
Cleanup cancels queued work and joins active reads before closing their file
descriptors. Both Textual controls are tested during real Sven reads, with
request-to-cleanup and simulated UI dispatch measured separately in the results.

The full 920-grain scan and remote HDD scaling remain unmeasured. A larger scan
can have a lower catalogue hit rate; the memory bound and source-read fallback
remain in effect.
