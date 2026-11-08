# Archived fit-specific sparse input experiment

**Withdrawn from production: this approach fails the first-run requirement.**
It required 648.52 seconds of preprocessing before fitting. On the validated
2,103-grain input, preparation plus fitting took **839.54 seconds**, compared
with **763.98 seconds** for the original-image workflow: a **9.9% slowdown**.
The earlier 4.00× figure described fitting after preparation and excluded this
cost. It is not a cold-run or end-to-end production speedup.

The experimental preparation command and its tests have been removed. Use the
original image data with the normal fit-grains workflow. Existing fitting,
masking, thresholds, tolerance passes, refitting and output features remain.

The experiment reduced the Sven FCH5 file from 44.30 GB to 816 MB by omitting
finite, nonnegative pixels at or below the configured threshold of 1,000.
Retained values and omega metadata were unchanged. Prepared-data fits and
written outputs matched the original-input branch run exactly for 2,103 grains;
18 example runs also matched master bitwise. These scientific checks do not
make the first-run performance tradeoff acceptable.

The [archived measurements](fit-grains-input-cache-experiment.json) retain
input hashes, timings, memory measurements and comparison scope. The withdrawn
implementation is preserved in commit `bb0b5bb9`; large research artifacts remain
local under `.local-bench/invalid-pixels/`. They are not a production dependency.

A separate installed-binary mismatch was found during regression investigation:
the loaded extension had find-orientations symbols but no native fit-grains
kernels. Fit-grains now checks compatibility before image access. Rebuild native
extensions after changing source branches; see the [build guide](fit-grains.md#build).
