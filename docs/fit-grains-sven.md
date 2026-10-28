# Sven FCH5: targeted reads and guarded validation

The local `sven_dataset` contains a 44.3 GB FCH5 file, 3,600 frames of
4,362 × 4,148 pixels, and 920 initial grains. Its stored sparse pixels expand to
approximately 322 GiB. A small spot window previously required decompressing
an entire sparse frame, repeatedly, across tolerance passes and grains.

## Measured scope

These measurements cover **grain 0**, all three configured tolerance passes,
and refitting, with one worker and a 512 MiB retained-image allowance. Every run
had a 4 GiB Linux address-space ceiling and single-threaded BLAS. Thresholds,
tolerances, material settings and fitting algorithms were unchanged.

| Reader | Complete fit time | Peak process RSS |
| --- | ---: | ---: |
| Previous branch whole-frame reader | 864.49 s | 1.64 GiB |
| Initial chunk reader, buffered I/O | 38.01 s | 1.16 GiB |
| Final reader, direct I/O, also writing spot files | 49.49 s | 1.16 GiB |

The buffered comparison measured **22.7×** improvement. The final guarded run
was **17.5×** faster than the previous reader despite bypassing filesystem cache
and additionally writing its spot file. The latter runs have different I/O
policies and output settings; this is not a controlled estimate of the direct
I/O option's overhead. These are individual, profiled runs, not repeated medians
or a measurement of the entire 920-grain job. They do not establish server HDD
performance. The performance baseline is the previous **branch reader**, not an
unmodified master run.

An independently captured first extraction batch sampled 531,776 pixels. Every
value matched the whole-frame path exactly. It took 2.99 s with chunk reads and
32.92 s with whole-frame reads; the repeated chunk batch took 0.60 s.

## Changes

The FCH5 path constructs a compact row index from aligned coordinate chunks,
then fetches only column and intensity chunks needed for the requested pixels.
It preserves sample order, dtype and CSR zero-addition semantics. Unsorted rows,
noncanonical columns and duplicates retain the existing full-frame conversion.
Small scans keep their existing full-frame cache path.

All retained chunks and row indexes share the existing worker byte allowance.
Indexes have a protected quota inside that allowance so intensity-cache churn
cannot repeatedly evict tiny indexes. Requests follow physical file offsets,
which can differ from omega order after parallel FCH5 writing, to reduce HDD
seeks. Sample scatter and centroid arithmetic retain their original order.
Cancellation is checked between frame groups, coordinate chunks and sampled rows.

The optimization changes pixel access. It does not substitute a global centroid
catalogue, change spot boundaries or alter connected components and fitting.

## Memory precautions

`image_cache.max_size_mb: 8192` bounds retained image buffers, not total RAM.
The benchmarks used a separate hard 4 GiB `RLIMIT_AS` ceiling, one process, and
512 MiB of image cache. Allocation-limit tests verify that excess allocations
are rejected. The benchmark rejects multiple workers with this guard because
per-process limits cannot enforce an aggregate multiprocessing ceiling.

Buffered reads also caused Linux's filesystem page cache to grow despite low
process RSS. Clean cached pages for this dataset were released, and subsequent
Sven validation used `O_DIRECT` through HDF5's file-object driver. Filesystem cache
remained roughly flat during the final run instead of growing with the scan.
Direct reads fail if unsupported; they do not silently fall back to buffered I/O.
No intermediate image files or scratch cache were created. The input YAML and
scientific settings were preserved; benchmarks used a separate normalized copy.

See [the memory guide](fit-grains-memory.md) for the guarded benchmark command.
These guards are benchmark options. The ordinary CLI's image-cache setting is
still **not** a hard total-memory limit. A verified system cgroup limit is needed
to enforce a ceiling including every worker and all filesystem cache. This host
requires administrator authorization for that system control; a user systemd
unit advertised a limit without joining a constrained memory cgroup, so it was
not trusted for testing.

## Numerical checks

The complete tested Sven grain's fits and all returned spot records match both
the previous whole-frame reader and master's original extraction and optimization
code **exactly**, including dtype, ordering and NaNs. The master reference is
`ca51e7d5d278bafb6a007dd284d9867714b19465`, with separately compiled native
transforms. Master received a bounded pixel-only input adapter, since its eager
reader would require hundreds of GiB. That reference is a numerical check, not
an unmodified-master workflow timing.

All nine rotation-series examples match unmodified master exactly:
six ruby studies, Dexelas, Dexelas ROI and zirconium. Their fits and returned
spot records are compared without a numerical tolerance. Regression tests cover
chunk boundaries, physical frame ordering, empty and missing pixels, ROI/frame
mapping, duplicates and fallback, signed zero, NaNs, float/integer data,
coordinates at file offsets above 2³¹, allocation ceilings, direct-read alignment
and EOF, cache accounting/pickling, and cancellation. The focused regression
suite passed **189 tests**, under a 1.5 GiB per-process allocation ceiling.

The full 920-grain Sven job has **not** been validated against master. Scaling
and full-dataset numerical acceptance remain separate checks. Measurements and
source hashes are in [the results file](fit-grains-sven-results.json).
