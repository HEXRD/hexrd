# Bounded image memory for fit-grains

Fit-grains now chooses between eager loading for small scans and lazy,
byte-bounded frame reads for large scans. It does not write a staging cache or
use scratch disk. Existing NPZ, FCH5, HDF5 and TIFF inputs can be used directly.

For the 256 GiB server with ten workers, a reasonable starting configuration is:

```yaml
multiprocessing: 10
fit_grains:
  # Keep the existing threshold, tolerance, refit, etc. here.
  image_cache:
    mode: lazy
    max_size_mb: 65536
```

`max_size_mb` is MiB **in total across the fitting workers**, including every
panel and ROI. With ten workers, this example allows 6.4 GiB of retained frames
per worker. It is not 64 GiB per worker or per panel. The configured value is
clamped to half the currently available system memory. A zero allowance reads
frames without retaining them between calls.

The default mode is `auto`. Its total allowance is a quarter of available RAM,
capped at 64 GiB. It selects eager loading only when twice the conservative
estimate of decoded image storage fits that allowance. Metadata and ZIP member
sizes supply the estimate without decoding the scan. Processing that changes
shape or dtype samples one frame. `mode: eager` explicitly restores the previous
loading policy. Both the selected policy and the actual allowance appear in the
Textual dashboard's details/log and in plain batch logs. Worker counts are reduced
if the estimate of single-frame decoder workspace leaves insufficient headroom.

## Why this changes the memory behavior

Previously, reading one frame could decompress the entire frame cache. Fit-grains
explicitly triggered this for every detector before starting fork workers. ROI
and native extraction views could retain complete scans, and fallback adapters
could construct a second sparse representation of the same pixels. Non-fork
workers could each load a separate scan.

The lazy path passes file descriptors and metadata to spawned workers. Each
worker opens its own readers on demand and uses one byte LRU across its panels.
NPZ and FCH5 readers decode individual frames without entering the adapters'
whole-scan loaders. Other file formats use the registered image-series reader.
There is no conversion to a disk cache. File-backed workers use spawn to avoid
inheriting open HDF5 handles. Applications calling the Python API must use the
usual `if __name__ == '__main__':` guard when starting multiprocessing; the HEXRD
CLI already does this.

Extraction groups requests by source frame and reads them in increasing frame
order within each patch batch, copying only requested pixel values into the
bounded patch workspace. A native view borrows one frame at a time. Thus an omega
window spanning hundreds of frames cannot defeat eviction by retaining a native
view of all those frames. ROI coordinates refer to the shared source, avoiding
whole-scan ROI copies. Built-in dark, crop, flip and other processing, and frame
subsets, are applied with the existing image-processing operations. Native spot
statistics, fitting objectives, thresholds, tolerances and solvers are unchanged.

The budget covers retained image buffers, including underlying allocations and
entry overhead. It is **not a hard limit on process RSS**: Python/native libraries,
metadata, calibration arrays, decoder/chunk workspace, one active source frame,
patch buffers and fitting results need additional memory. HDF5 readers use a
1 MiB chunk-cache allowance per open file, and each worker retains at most eight
open readers. Already materialized, caller-owned arrays remain resident; their
contents (including caller edits) are preserved. On fork platforms these arrays
can remain shared; parallel resident-array inputs on spawn-only platforms require
using one worker or reopening file-backed inputs. Unsupported custom processing
is rejected instead of silently changing its behavior.

The normal CLI does not collect every grain's extracted spot data in RAM.
Python callers using `return_pull_spots_data=True` still explicitly request that
potentially large result collection, outside the image allowance.

## HDD tradeoff

A byte limit prevents dataset-sized RAM growth, but repeated decoding and HDD
seeks can cost time. A larger allowance reduces rereads. Frame ordering improves
locality within a batch; independent workers can still issue overlapping/random
reads. This change does not claim a speedup on the remote dataset. Retaining the
eager path for small scans avoids imposing lazy I/O on existing small workloads.

For a reproducible run, the benchmark accepts:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_fit_grains.py run \
  --source "$PWD" --config /path/to/config.yml --grains /path/to/grains.out \
  --workers 10 --image-mode lazy --image-cache-mb 65536 \
  --write-spots --no-return-spots --memory --output /path/to/small-output-prefix
```

This writes only the requested grain/spot outputs and small benchmark artifacts,
not an image staging cache. The memory report includes aggregate parent/worker
PSS and USS, as summed RSS double-counts shared pages.

## Validation

The numerical reference is the separately built master tree at
`ca51e7d5d278bafb6a007dd284d9867714b19465`, with the same Python, NumPy and SciPy
versions. Forced-lazy tests compare all 1,690 Inconel grains with ten workers and
all nine rotation-series examples (six ruby studies, Dexelas, Dexelas ROI and
zirconium). Fits and all returned spot records must match exactly, including
dtypes, ordering and NaNs; five written Inconel spot files must be byte-identical.
The examples use a deliberately small 2 MiB allowance to exercise eviction.

A separate memory probe reads 1,024 frames from a **logical** 48 GiB HDF5 scan
with 8 MiB frames and a 32 MiB allowance. Retained image storage peaks at
25,168,896 bytes (three frames plus overhead). Peak private memory is
179,920,896 bytes in both the first and second 512-frame intervals: memory
plateaus instead of growing with the scan. This uses an HDF5 fill-value dataset
whose physical file is about 100 KiB; it tests allocation behavior, not HDD
throughput, compressed-data decoding cost, or full grain fitting on 48 GiB.
The inaccessible remote 42 GB dataset has not been benchmarked here.

See [the machine-readable results](fit-grains-memory-results.json) for the
validation results and memory measurements. Regression tests also cover format
readers, processing and subsets, owner-aware eviction, native extraction
agreement, descriptor pickling without payloads, and cleanup on fitting errors.

The regression suite passed 560 tests. After the final resident-array spawn guard,
51 targeted checks passed, including all 25 memory tests. The forced-lazy CLI
also completed through the real Textual TUI in a PTY with three grains.
