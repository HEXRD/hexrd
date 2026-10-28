# Live fit-grains metrics

The Textual fit-grains dashboard now reports work *within* a grain, including
before any grain finishes. It keeps the shared find-orientations palette,
navigation, logs, RAM display, and cancellation controls.

The new **FIT ACTIVITY** panel shows:

- **Passes and solver:** completed tolerance passes, skipped passes after an
  insufficient-reflections exit, optimizer objective evaluations, refits, and
  the numbers of workers extracting or fitting. Optimizer evaluations include
  finite-difference calls; they are not iteration counts or convergence estimates.
- **Extraction:** reflections processed in workers' current passes, source-frame
  groups sampled in their current patch batches, and cumulative reflections
  tested/containing signal. Current-pass and current-batch bars reset when work
  advances. They are not percentages of the entire input file read.
- **Image reads and cache:** frame loads, recent frames/s, decoded buffer bytes
  and MiB/s, cache occupancy/allowance, hit percentage, evictions, and cumulative
  read/decode time summed across workers. Decoded bytes are **not physical disk
  bytes**, and decoded throughput is **not disk bandwidth**. Repeated reads of
  the same frame and independent workers' reads count separately. Processing a
  frame does not count its underlying source read twice.
- **Activity age:** seconds since the newest update and the longest interval
  without an update among active workers. This shows when no worker has reported
  additional work. A long interval can also mean a long native operation or
  blocked read; it is not proof of a hang or an ETA.

The grain table adds each grain's current phase, frame/reflection work counts,
and age since its last update. Completeness and chi-square still describe the
latest completed fit. The panel adapts to smaller terminals, retaining the work
bars and hiding the summed read-time line when vertical space is limited.
Small image allowances are displayed in MiB rather than rounding to zero GiB.

Lazy frame loading now becomes visible under extraction even though preparing
readers in the existing “Load images” stage takes very little time. Eager/shared
inputs are identified explicitly instead of displaying misleading lazy-read
statistics.

## Overhead and reliability

Detailed counters are enabled for dashboard runs. Plain batch progress retains
its original grain events without detailed telemetry. No scientific thresholds,
windows, solver settings, reflection selection, or returned API result formats
change.

Workers normally publish cumulative snapshots about every 0.2 seconds, plus
phase transitions. Progress publication never waits for queue space. A later
snapshot recovers counters from skipped updates. Final snapshots also travel on
the required fit-result channel, so queue saturation cannot leave the completed
pass counter short. The parent retains one snapshot per worker and details for
at most 32 recent grains; image arrays never enter the telemetry channel.

## Validation

The numerical reference is the separately built master tree at
`ca51e7d5d278bafb6a007dd284d9867714b19465`. The starting feature branch is
`51e1c853` (`fit-grains-progression`).

With telemetry enabled, all 1,690 Inconel fits and returned spot data match
master exactly. All nine rotation-series examples also match: six ruby studies,
Dexelas, Dexelas ROI, and zirconium. Comparison checks dtypes, ordering and NaNs
without a numerical tolerance. Five written Inconel spot files remain
byte-identical. Final parallel counters reconcile all **5,070 tolerance passes**.

A matched three-run benchmark with ten workers, eager images and no returned
spot collection measured median fitting times of **6.947 s before** and
**6.943 s after**. Both runs included the dashboard's progress collection. This
small local sample shows no measured slowdown; it does not establish remote HDD
performance or a speedup.

The regression suite passed **571 tests**. UI tests exercise live counters while the main bar remains at zero completed
grains, queue saturation/recovery, bounded retention, source-read accounting,
small/wide layouts, and disabled batch telemetry. The real CLI was also run in a
PTY, completed with the Textual dashboard, and closed normally. A separate real
three-grain lazy run reported nine completed passes, frame reads, reflection
counts, and optimizer evaluations.

Machine-readable measurements are in
[fit-grains-live-metrics-results.json](fit-grains-live-metrics-results.json).
The remote C103 dataset is unavailable locally and has not been tested here.
