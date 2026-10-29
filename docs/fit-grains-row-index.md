# SIMD construction of sparse row indexes

This is an incremental improvement over `b0d81b4d`, the pixel-reuse and batched
index reader. It changes integer index construction rather than peak or fitting
arithmetic.

The interrupted prototype had a C++ declaration error and had not produced a
speed measurement. That error is fixed. The retained implementation validates
row coordinates and fills row offsets in one SIMD scan using the existing
xsimd dependency. Most row coordinates are long constant runs, so only SIMD
packs containing a row boundary need scalar index updates. This avoids the
full-sized temporary boolean arrays and per-row binary searches in the NumPy
implementation. The scan releases the Python GIL.

The existing byte bounds, decoder queues and cancellation checks remain in
place. Unsupported native extensions retain the NumPy fallback. Disordered or
out-of-range coordinates still reject the partial-reader path and use the
existing full-frame behavior. No scientific settings or image values change.

## Measurements

Three alternating pairs, with four Sven grains, all three tolerance passes,
refitting, returned spot records and spot-file output:

| Measurement | Before (`b0d81b4d`) | After |
| --- | ---: | ---: |
| Runs | 23.75, 21.21, 21.62 s | 18.75, 18.30, 18.79 s |
| Median | 21.62 s | 18.75 s |

This is **1.15× faster**, an additional improvement over the preceding twofold
speedup. A separate 13-grain pair improved from **35.48 to 33.83 seconds (1.05×)**.
The larger case spends a smaller fraction of its time constructing row indexes.
These are comparisons with the preceding branch commit, not timings against
unmodified master.

All runs use one process, single-threaded BLAS, a 512 MiB staged workspace,
Linux direct FCH5 reads and a **4 GiB hard address-space ceiling**. No image
scratch files or pre-populated pixel catalogue are used. Full 920-grain
throughput and remote HDD scaling remain unmeasured.

A newer Blosc context was also tried on three sampled compression chunks. It
improved two samples by only 5–7% and slightly slowed the third, so that change
was not adopted. The retained code uses the existing codec and dependencies.

## Correctness

The focused regression suite passes **326 tests** under a 1.5 GiB allocation
ceiling. The row-index selection passes 38 tests, including an explicit check
for unaligned output buffers. All nine examples match master exactly in both
measurement modes. All repeated four-grain runs match the master reference,
including byte-identical spot files; the 13-grain run matches the preceding
branch, including all 13 spot files.

Peak fitting RSS stays below **1.1 GiB** in both workload sizes. Both Textual
cancellation controls pass during real reads. Request-to-cleanup latency is
5 ms for the button and 2 ms for Ctrl+C; simulated input/dispatch takes 3.13 s
and 0.21 s respectively and is recorded separately from cleanup latency.


The focused suite includes exact comparisons with NumPy searchsorted across
partial chunks, SIMD boundaries, missing rows, empty frames, the signed 16-bit
limit, invalid coordinates, output dtype/alignment requirements and index
overflow. The existing image, centroid, fitting and dashboard checks remain.

Sven fits and returned spot records are checked against the original master
scientific algorithms with bounded pixel input; eager master loading exceeds
local memory. Written four-grain spot files are compared byte for byte. The
13-grain fits and all spot files are checked against the preceding branch.
All nine example studies are checked against unmodified master in both modes.

Final regression and cancellation checks, timings and memory measurements are
recorded in [the results JSON](fit-grains-row-index-results.json). Use the
existing [staged configuration](fit-grains-pixel-reuse.md#configuration-and-memory)
and rebuild the native transforms extension to enable this kernel.
