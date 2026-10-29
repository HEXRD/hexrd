# Segment and measure peaks before fitting grains

Add this to the existing `fit_grains` configuration block, preserving its
threshold, tolerances, material, estimate and refit settings:

```yaml
fit_grains:
  measurement_mode: staged
```

This opt-in mode prepares peak measurements for a complete tolerance pass before
optimizing that pass's grain parameters. The existing Textual dashboard shows
planning, frame sampling and peak segmentation activity, with prepared-window,
frame-visit and completed-group diagnostics. The default `grain` mode remains
available and unchanged in its numerical behavior.

## Exact measurements rather than a fixed global centroid catalogue

Master's measurements depend on each grain's predicted angular window and
resampling grid. A standalone detector-space centroid loses that information.
The staged mode therefore retains the exact pixel support needed by each
original window:

1. Predict all current grain windows, compressing their integer coordinates
   losslessly and recording their source frames, shapes and thresholds.
2. Group requests within a workspace allowance, then visit shared source frames
   in order. Sparse FCH5 requests read the needed chunks; denser groups load one
   source frame. Repeated windows and detector ROIs share those source reads.
3. Retain sampled pixels only while an omega window is incomplete. Once complete,
   run the same connected-component selection, intensity measurements and moment
   calculation as the existing native extractor, using the original array order.
4. Discard the window pixels. The grain optimizer consumes the prepared centroid
   and reflection records, with no image reads during that fitting phase.
5. Regenerate windows after the grain parameters change, and repeat for the next
   configured tolerance pass. Refitting keeps its existing behavior.

The existing `pull_spots` entry point performs geometry planning and conversion
of completed statistics into reflection records. Native panel results retain
their planned geometry and statistics, so result assembly needs no second
prediction or image extraction. Custom extractors retain the checked replay
fallback. This is a first-run shared-read
pipeline, not a cache populated by running the full fit beforehand. It does not
add a different peak model or reuse stale measurements across changed windows.

## Memory and cancellation

The staged implementation uses one coordinator/fitting process, including when
`multiprocessing` requests more workers. The dashboard reports the effective
worker count. It forces lazy image access. Large compatible FCH5 inputs also use
a bounded native decoder thread pool; see the [throughput guide](fit-grains-throughput.md)
for the `image_cache.read_workers` setting and updated measurements.

The preparation workspace derives from the image-cache allowance and is clamped
to 16–512 MiB. It accounts for compressed geometry, event tables, compact statistics
and the estimated peak live pixel support. Up to one quarter, capped at 128 MiB,
is reserved for source chunks and compact row indexes. Decoder chunks can
survive neighboring source frames under their LRU byte bound. Small indexes survive across passes, and the remaining workspace can
retain exact source pixels for later windows. See the
[pixel-reuse guide](fit-grains-pixel-reuse.md) for the current bounds and timings.
A pending geometry plan, source-frame decoder workspace, fitting objects and returned results need additional RAM, so this allowance is
not a total process limit.

Requests are split into smaller groups when their live windows would exceed the
allowance. A single oversized geometry plan or unsupported native extraction
uses the existing bounded measurement path. That preserves the result while
reducing the opportunity to share reads. No intermediate images or disk staging
cache are written. As with ordinary fitting, inputs must remain unchanged during
a run.

Cancellation is checked during planning, source reads, window sampling, and
optimizer evaluations. Window buffers are released on cancellation and errors,
and temporary callback/context state is restored. Cancel run and Ctrl+C use the
existing Textual controls. The Linux direct FCH5 reader also checks cancellation
between HDF5 read callbacks. Cancellation skips flushing queued partial spot
files, preserving files already written for completed grains.

For a memory-constrained local benchmark, use a single worker, the allocation
guard, and direct FCH5 reads:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_fit_grains.py run \
  --source "$PWD" --config /path/to/config.yml --grains /path/to/grains.out \
  --measurement-mode staged --workers 1 \
  --image-mode lazy --image-cache-mb 512 \
  --memory-limit-mb 4096 --direct-io --limit 4 \
  --memory --write-spots --output /path/to/output-prefix
```

The 4 GiB limit is a per-process allocation ceiling, and direct FCH5 reads bypass
filesystem caching of the large image input. These benchmark guards do not turn
the ordinary CLI's image-cache setting into an aggregate OS memory limit. See the
[memory guide](fit-grains-memory.md) for the distinction.

## Measured result and acceptance scope

On four Sven grains, with all three tolerance passes, refitting and spot-file
output, using one worker, single-threaded BLAS and direct FCH5 reads:

| Workflow | Time | Peak process RSS |
| --- | ---: | ---: |
| Previous branch, grain-by-grain extraction | 118.19 s | 1.20 GiB |
| Staged segmentation and measurement | 97.67 s | 0.73 GiB |

This is **1.21× faster** with approximately **39% less peak process RAM** in one
matched pair of runs, not a 5× result or repeated-run median. An initial staged
prototype was slower on a single grain; compact-index retention was subsequently
added. The mode remains opt-in until larger-scale timing establishes when its
sharing outweighs planning and coordination overhead.

The four Sven fits, all returned spot records and all four written spot files
match both the previous branch and master's original algorithms exactly. The
master reference uses separately compiled native transforms and bounded pixel
input, because its eager reader cannot fit this image volume in local RAM. This
is a numerical reference, not an unmodified-master workflow timing.

All nine examples match unmodified master for fits and returned spot records:
six ruby studies, Dexelas, Dexelas ROI and zirconium. The default workflow was
also checked against master on ruby and Dexelas ROI after introducing the
shared optimizer state machine. Unit tests cover shared frame reads, original
pixel/centroid order, scalar dtypes, repeated and wrapped frames, ROI mapping and
signed zero, dense windows, geometry mismatch refusal, workspace fallback,
interleaved grain state, spot writing, and cancellation cleanup.
The focused regression suite passed **242 tests** under a 1.5 GiB allocation
ceiling; the latest staged-specific selection passed **29 tests**.

Both Textual cancellation controls were exercised during real Sven FCH5 reads.
Workflow cleanup took 18 ms after the button's cancellation event and 1 ms after
the Ctrl+C event was registered. These measure request-to-cleanup latency;
headless input simulation and rendering took longer (3.30 s and 0.31 s total,
respectively), and are recorded separately in the results.

The full 920-grain Sven run and remote HDD scaling remain unmeasured. No threshold,
window, solver tolerance or numerical acceptance tolerance was relaxed.
Machine-readable evidence is in
[fit-grains-staged-results.json](fit-grains-staged-results.json).
