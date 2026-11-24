# Strategy for complete scientific processing at 100 Hz

The target is a provisional scientific update within 10 ms of every image's
arrival, with a final master-equivalent processing tail allowed. Every stage
counts toward the live deadline. Receiver throughput alone does not meet it:
short capture probes still miss deadlines and the prototype has no per-frame
scientific fitting path. Work must carry useful state through acquisition,
rather than rerun successive offline jobs.

## Measured gap

The latest plain-LZ4 Inconel replay processed 1,440 frames at 150.33 Hz, then
spent 231.32 seconds on final analysis of 1,690 grains. Its final-analysis log
records:

| Stage | One-worker elapsed time | Interpretation |
|---|---:|---|
| Map handling/export | 2.20 s | Supplied incremental maps still go through archive publication |
| Seed peaks and fibers | 7.60 s | Recomputed for each independent snapshot |
| Orientation scoring | 17.28 s | 10,736,115 trials in this scan |
| Clustering | 3.24 s | Establishes the canonical initial grain parameters |
| Export/setup interval before image setup | 11.30 s | Inferred from timestamps; archive compression is on this path, but this is not a compression-only profile |
| Fit pipeline | 187.37 s | Includes planning, extraction, culling, optimization and refit; not an optimizer-only measurement |

Other setup and output work account for the remainder. These observations are
from `.local-bench/live-fast/capture/final.log`; the stage boundaries and parity
results are in [the validation record](live-fit-grains-rate-validation.json).

At 100 Hz, a 1,440-frame scan arrives in 14.4 seconds: the current final pass
has a roughly **16× analysis-throughput gap**. Reserving four of sixteen CPU
threads for acquisition and one for coordination/export leaves eleven for
analysis. Even ideal division of the fitting stage alone gives 17.0 seconds,
before discovery or output. This is optimistic arithmetic, not measured scaling;
storage contention, GIL constraints and sequential dependencies can worsen it.

## Proposed implementation order

Priorities 1 and 2 are now implemented in the sandbox. A paired full-scan
comparison improved process wall time from 265.69 to 147.91 seconds (1.80×),
with exact outputs; the live service retained its workers and completed final
analysis in 129.64 seconds while acquisition held at 149.44 Hz. See the
[implementation and validation](live-fit-grains.md#persistent-analysis-and-archive-overlap).
The remaining work below still requires measurement; complete science has not
yet reached 100 Hz.

Further profiling removed detector-mask copies from geometry snapshots and
extended the existing sparse kernel to guarded floating-point half counts and
masks. A 2,103-grain fitting diagnostic improved from 209.91 to 149.80 seconds
with exact outputs and 37 fewer runtime lines. It uses an already-prepared
support cache; its 648.52-second preparation and orientation discovery are
excluded. It therefore does not establish complete Sven throughput. See the
[scope and validation](live-fit-grains.md#exact-sparse-extraction-and-geometry-snapshots).

A subsequent change to the shared dense-to-CSR kernel removes
vector growth and final copies. Sven conversion improved from 82.10 to
37.18 ms/frame (2.21×); Inconel improved too. Conversion plus map calculation
reached 108.49 Hz at six threads in the isolated in-memory probe before the
allocation safeguard. The final safeguard passes the complete Inconel live
replay with exact outputs. These kernel timings exclude persistence and the
scientific tail; the 100 Hz
complete-processing goal remains unmet. See the
[allocation experiment](live-fit-grains.md#single-pass-sparse-preparation).

A read-only shared-store diagnostic now fits all 2,103 Sven grains with four
processes in 33.04 seconds, with identical grain fields and spot files. Store
setup adds 21.94 seconds. Its 7.75 GiB aggregate allocation ceiling counts the
shared mapping once and bounds every process's remaining address space.
The earlier support-cache preparation (648.52 seconds) and discovery are still
excluded, so this remains a prototype rather than a full-throughput result.
Building that support during acquisition is the next measured opportunity.

| Priority | Strategy | Expected benefit and exactness condition |
|---|---|---|
| 1 | Persistent analysis session and measured CPU allocation | Implemented: static geometry/material and bounded fitting workers survive prefix/final transitions. The coordinator can use four discovery threads and two fitting processes within the local memory budget. Staged and large FCH5 work retains its grouped single-worker path. |
| 2 | Remove archive publication from the fitting dependency path | Implemented for trial archives: write the canonical initial-grain table first, then overlap compression with fitting. Every output is retained and writer success is required before final completion. Map archive publication remains part of discovery. |
| 3 | Extend the staged coordinator into a frame-driven scheduler | Reuse `_fit_grain_steps`, `MeasurementPlan`, and the native statistics kernels. Index each arriving frame once; route its pixels to all ready grain windows. Release a measurement only when its complete omega support is present. Trigger the next fit/pass when dependencies are satisfied, rather than waiting for every grain in a pass. |
| 4 | Incremental discovery and scoring with conservative invalidation | Retain candidate generations, reflection-bin predictions and hit/miss counts; update affected angular neighborhoods instead of rescoring all candidates for every snapshot. Backfill older observations for newly discovered candidates. Preserve canonical candidate ordering and clustering semantics. Some changes require broad invalidation; do not assume all old candidates are stable. |
| 5 | Native shared window work across active grains | Coalesce exact source-pixel queries for all active windows on a frame, including late-discovered grains replaying retained pixels. Share immutable sparse tiles under one byte budget. Keep full pixel support: global detector centroids cannot replace the grain-dependent polar windows, connectivity and component selection used by master. |
| 6 | Profile and accelerate the remaining fitting hot path | Separate planning, sampling, segmentation, residual evaluation, callback overhead and output costs before choosing a kernel. Target duplicated queries and Python dispatch first; use native batched work where it preserves arithmetic. The objective is already native, and the optimizer already uses MINPACK. A different solver, analytic Jacobian or changed reduction order cannot be presumed bitwise equivalent. |

Beyond the measured first two priorities, speedups remain unmeasured. The next
step is the persistent frame scheduler, followed by the largest remaining
native kernels. Offline
execution should feed completed frames into the same coordinator.

## Frame-driven measurement implementation

`FrameMeasurements.append(frame, sampler)` now accepts one arrived frame at a
time through the same bounded tiled measurement code used by ordinary staged
fitting. Each angular window retains its original omega slots, pixel mask and
inverse sampling map; segmentation waits until the full window is present.
Feeding frames in a different order, repeated omega references and sparse
half counts preserves the existing native statistics exactly. Failed sampling
or cancellation prevents further use of that partial accumulator.

This is a scientific scheduling primitive, not a complete live fit. Discovery
still needs to supply provisional models from acquired data, and the receiver
must route their windows into this accumulator and publish fitted updates
under the shared 10 ms deadline. Those steps remain unimplemented. Final
discovery and fitting must still rerun canonically and pass complete-scan
parity checks.


## Correctness constraints that determine the architecture

Exact full-scan results depend on future observations. This implementation's
`_clean_map()` uses a global median; peak components and centers can change as
the scan grows. Some peak methods also rescale by global intensity extrema.
Clustering can change grain membership, ordering and starting parameters.
Consequently, a peak leaving a rolling window does not by itself certify that
its derived orientation or fit is final.

Incremental discovery can accelerate provisional results. Reuse in authoritative
outputs needs stronger evidence: identical canonical seeds and settings,
identical ordered measurements, and the same tolerance passes, culling and
optimizer behavior. Cache original pixels freely; reuse measured windows only
when their complete inputs match. Reuse an optimized grain only when its whole
reference computation is unchanged. Otherwise invalidate and recompute from the
canonical initial parameters. Provisional warm starts must not silently enter
the final reference path.

An appendable immutable pixel store supplies bounded replay for late candidates
and changed windows. A pure rolling buffer that discards old pixel support
cannot guarantee complete-scan parity. New grains must remain discoverable
through the end of acquisition; no previous-scan grain list is assumed.

## Acceptance criteria

Measure complete scientific service capacity, not just completed receiver jobs:

- Maintain at least 100 Hz without an increasing *analysis* backlog. Aim for
  125 Hz capacity as an engineering headroom target, not a measured result.
- Report first provisional orientation/fit latency, age of the newest fitted
  observation, outstanding windows/grain passes, and final drain time separately.
  A finite acquisition that buffers all scientific work until its end fails this
  test even if reception exceeds 100 Hz.
- Exercise acquisition while actual grain optimization is running. The existing
  short replays overlap prefix orientation analysis, not completed provisional
  fits. Use longer scans or repeated scans with persistent analysis to measure
  steady-state capacity; do not assume a finalized result can precede its data.
- Validate full-size raw Eiger 16M inputs, realistic density, the supported wire
  codecs, and sustained storage bandwidth. The present 3500×3500 sparse Inconel
  result does not establish those limits. Preserve Sven's fractional data in
  offline/core tests rather than quantizing them into Eiger unsigned messages.
- Keep the aggregate local allocation budget below 8 GiB, including all worker
  processes, pinned buffers and writers. Keep queues byte-bounded and cancellation
  responsive as jobs await frames, execute or drain.
- Require exact final orientations, scores, grain parameters, completeness,
  chi-square and spot outputs against the offline reference and master examples.
  Include late peaks, changing global medians, component merges, omega gaps and
  roughly 2,000-grain workloads. Final drain time must include any invalidated
  work; cache reuse alone is not evidence that the finalization tail is small.

The intended contract is sustained 100 Hz with explicitly measured provisional
latency and finalization delay. Exact parameters for a complete scan cannot be
declared before the last observations on which they depend. A maximum acceptable
lag remains a separate requirement from the acquisition rate.
