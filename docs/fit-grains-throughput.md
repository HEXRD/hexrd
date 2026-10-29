# Feeding the fit-grains compute pipeline

This change targets the staged measurement workflow on Sven's FCH5 input. It
keeps the original grain-dependent windows, pixels, connected-component
selection, centroids and optimizer. It changes how those inputs reach the
compute kernels.

## Bottlenecks found

A four-grain profile of commit `dec45c99`, with all three tolerance passes,
refitting and spot-file output, took 112.26 seconds. HDF5 reads and decoding
accounted for 81.10 seconds. Peak statistics accounted for only 0.88 seconds.
The staged coordinator had no parallel decode queue. Adding more grain-fitting
processes would duplicate image work and memory without addressing that feeder.

Ordinary h5py calls share an interpreter-wide HDF5 lock, including calls on
separate files. Consequently, putting the same dataset reads in Python threads
would serialize them. See the [h5py threading documentation](https://docs.h5py.org/en/stable/threads.html).

An initial direct-chunk prototype exposed another bottleneck: asking HDF5 for
each chunk's disk address repeatedly left the decoder threads waiting. It was
slower than the original and was stopped. The retained implementation traverses
the chunk index once and stores the addresses in compact arrays. It scans
metadata, rather than image pixels, to build this directory.

## Changes

* A small, shared thread pool reads independent compressed chunks with positional
  I/O and calls the existing native Blosc decoder. HDF5 metadata stays in the
  coordinator. The rolling queue is bounded by the effective worker count;
  requests and results are consumed in their original order. One codec thread
  per job avoids nested decoder pools.
* The implementation uses C-Blosc's validated, reentrant context API, provided
  by the installed `hdf5plugin` library. It adds no codec dependency or custom
  decompressor. See the [public C-Blosc API](https://github.com/Blosc/c-blosc/blob/main/blosc/blosc.h).
* A C++ batch operation validates sparse column coordinates once per requested
  row and locates requested pixels with standard-library binary searches. It
  releases the Python GIL while searching. Noncanonical coordinates retain the
  existing SciPy fallback, including duplicate coalescing.
* Result assembly uses the geometry captured during planning. It no longer
  predicts and builds the same spot windows a second time. Invalid peak IDs
  retain their original ordering across panels.
* Direct reads reuse one aligned input buffer per decoder thread. The reader
  reads the compressed extent, rather than the full capacity of that buffer.
* Benchmark JSON now records parent process CPU time and effective CPU cores
  (CPU seconds divided by wall seconds). These include this process's threads,
  but exclude subprocess CPU time.

The fast chunk path handles immutable, local, numeric FCH5 datasets with one
Blosc filter, one coordinate/value column per chunk and decoded chunks no larger
than 1 MiB. Other filters, drivers, layouts, large chunk directories and small
cache allowances keep the existing HDF5 reader. Other supported image formats
retain their bounded readers and staged source sharing.

## Configuration and bounds

Merge these fields into the existing configuration without changing scientific
settings:

```yaml
fit_grains:
  measurement_mode: staged
  image_cache:
    mode: lazy
    max_size_mb: 512
    read_workers: 4
```

`read_workers` accepts integers from 1 to 8. It is a maximum for this staged
FCH5 decoder, shared across detector files. The effective count can decrease
with a small cache or fewer available CPUs. Setting it to 1 disables the
parallel chunk path. The default grain-by-grain mode does not create decoder
thread pools. More workers are not automatically faster on an HDD.

The staged workspace remains capped at 512 MiB and reserves at most 128 MiB
inside it for current image chunks and indexes. Decoder inputs, outputs, scratch
allowance and the chunk directory have a reservation inside that same cache;
they are not an extra per-file allowance. The directory is capped at 8 MiB
across files. A native pixel lookup can temporarily pin at most 16 MiB of
coordinate chunks, deducted from the cache while those arrays are live.

These workspace allowances are not total-process limits. All local Sven tests
use a **4 GiB hard address-space ceiling**, one process, single-threaded BLAS and
Linux direct image reads. Direct reads bypass filesystem caching of the large
FCH5 payload. No image scratch files are created. The test procedure stays below
the user's 8 GiB ceiling; it does not claim that the ordinary CLI's cache option
is an aggregate OS RAM limit. See [the memory guide](fit-grains-memory.md).

Cancellation checks cover chunk-directory construction, queue submission,
waiting for reads, window sampling and optimizer work. Cancelling discards
pending work and joins only the small bounded set of active reads before
closing their file descriptors. Decoder threads do not call HDF5 or the TUI.

## Validation and measurements

Two alternating pairs of runs without profiling produced these results:

| Measurement | Before (`dec45c99`) | After |
| --- | ---: | ---: |
| Four-grain times | 97.02, 97.86 s | 45.72, 45.80 s |
| Median | 97.44 s | 45.76 s |
| Peak process RSS | 0.74 GiB | 0.78 GiB |
| Mean effective CPU cores | 0.78 | 1.54 |

This is **2.13× faster** than the branch at the start of this change. Peak RSS
increases by about 42 MiB, while staying below 0.8 GiB. CPU utilization roughly
doubles; the sparse per-frame pipeline and disk reads still leave some decoder
capacity idle. This is not a claim of 5× speedup or complete CPU saturation.

Both repetitions reproduce all four master-reference fits and returned spot
records exactly, with no numerical tolerance. Each of their four written spot
files is byte-identical to its master reference. The focused regression suite
passes 252 tests under a 1.5 GiB allocation ceiling. All nine examples (six ruby
studies, Dexelas, Dexelas ROI and zirconium) match unmodified master in both
`staged` and `grain` modes, for 18 exact fit/spot comparisons.

Both Textual cancellation controls were exercised during real Sven chunk reads.
Cleanup completed 4 ms after the button cancellation event and 2 ms after the
Ctrl+C event. Headless input simulation and dispatch are recorded separately:
3.19 s for the simulated click and 0.17 s for the simulated keypress. These are
not end-to-end hardware input latency measurements. The cancellation checks
stayed under the 4 GiB ceiling, with peak process RSS of 1.00 GiB.

Final timings, exact comparisons, regression checks and cancellation results
are recorded in [the machine-readable results](fit-grains-throughput-results.json).
The baseline is the start of this change on `fit-grains-progression`, commit
`dec45c99`, rather than master. All timing runs use four Sven grains and the
same scientific configuration, complete fits and written spot files. They do
not time the entire 920-grain scan or establish scaling on the remote HDD.

Master remains the numerical acceptance reference. The Sven reference uses
master's original scientific algorithms and its separately compiled transforms
with bounded pixel input, since its eager image loading exceeds local memory.
The example references use unmodified master. See the results for exact
acceptance checks against these references.
