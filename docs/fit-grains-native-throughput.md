# Native fit-grains throughput and bounded pixel reuse

This change extends the native staged pipeline at `e968097b`. The measured
comparison is with that commit on `fit-grains-progression`; master is a separate
scientific correctness reference.

## Changes

The existing Eigen geometry, native window coordinator, sparse sampler and
peak labeling remain in place. Additional work now executes natively:

- Boost.Unordered's flat map deduplicates source-pixel requests before sorting,
  so repeated polar subpixels do not all enter the sort. The inverse scatter
  retains the original sample order. Builds without Boost 1.81 or later use
  the existing standard-library sort path.
- Sorted cache entries merge in one C++ pass with byte copies. Query results
  already have unique, sorted keys, avoiding another NumPy deduplication.
- A native C-Blosc codec compresses cached coordinates and intensities in RAM
  when the raw pixel catalogue reaches its byte budget. Zstd level 1 and byte
  shuffle preserve the complete pixel representation, including floating-point
  NaN payloads and signed zero. No image scratch files are created.
- The run-based component kernel also handles nonnegative integer and
  half-integer floating-point intensities when all required sums and moments
  are exactly representable. Other values use the original NumPy/BLAS path.
  Thresholds, connectivity, component selection, bounding boxes and fitting
  arithmetic are retained.
- Intensity decoders use a continuous native queue. Only active workers allocate
  decoded chunk storage, and each worker scatters its chunk into disjoint
  output slots before taking the next job. A slow chunk cannot leave the other
  readers waiting at a batch barrier.

Python still owns configuration, Textual, cache admission and the optimizer
interface. Moving those interfaces wholesale would not eliminate the remaining
source reads and codec work. NumPy's original native reductions and optimizer
behavior are retained where changing arithmetic could change master results.

For the floating-point run kernel, doubling half-integer counts produces exact
integer masses. The guard bounds the total scaled mass by both the output sum's
mantissa and the largest coordinate's exact-moment range. Local bounding-box
moments are formed before division, retaining the reference rounding order.
Negative values, signed-zero backgrounds, nonfinite values, fractional
foreground counts and larger moments use the original implementation.

## Build

The C++ extensions use `-O3`, loop unrolling and release assertions. The optional
LTO build adds single-worker link-time optimization, hidden symbol visibility
and disabled semantic interposition. IEEE arithmetic and portable CPU targeting
are retained.

For a clean rebuild with GCC on Linux:

```bash
BOOST_INCLUDE_DIR=/path/to/boost/include HEXRD_ENABLE_LTO=1 \
  python setup.py build_ext --inplace --force --parallel 1
```

`BOOST_INCLUDE_DIR` is optional when the headers are installed on the compiler's
normal include path or under the Python environment's include directory. Boost
is header-only here; it is not downloaded by the build. Rebuild while HEXRD is
stopped. The private extension attributes `_pixel_query_boost` and `_fast_lto`
report which paths were compiled. Changing build flags requires `--force` or a
clean build, since setuptools does not track compiler flags as dependencies.

## Memory and cancellation

Each compressed frame decodes to at most 1 MiB. A query and its subsequent cache
merge share one restored entry; at most one restored entry is retained.
Encoding and merging fit within the existing catalogue scratch
reservation. Cache admission charges compressed allocation capacity, not raw
array sizes. Packing is activated only under pressure and for cache budgets of
at least 32 MiB. Unsupported codecs and larger entries retain the original path.

Native reader concurrency remains bounded by the configured worker allowance.
At most one decoded intensity chunk per active worker is allocated, independent
of queue length. Queued jobs retain their request buffers until completion;
errors and cancellation drain workers before releasing buffers or descriptors.
Catalogue migration checks cancellation between bounded entries. Native waits
continue to check every 25 ms.

Every large validation run uses one fitting process, single-threaded BLAS,
direct FCH5 reads and a hard 4 GiB address-space ceiling. Runs are sequential;
compilation and unit tests have separate 1.5 GiB ceilings. No validation run
requires the user's 8 GiB limit to be raised.

## Results

Fresh alternating runs against `e968097b` give these median fit-grains times:

| Sven grains | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 4 | 13.38 s | 12.06 s | 1.11× |
| 13 | 26.96 s | 22.15 s | 1.22× |
| 52 | 92.28 s | 62.32 s | 1.48× |

The smaller subsets have three before/after pairs each; the 52-grain subset has
two pairs. Runs include all three tolerance passes, refitting, returned spot
data and spot-file output. The staged workspace is 512 MiB with four readers.
Measured peak RSS for the optimized runs stays below 1.09 GiB. Full 920-grain
throughput and performance on the remote HDD remain unmeasured.

Raw runs, build options, source and binary hashes, exact comparisons and memory
measurements are in [the results JSON](fit-grains-native-throughput-results.json).

The retained build passes 440 focused tests under a 1.5 GiB ceiling. All nine
example studies match unmodified master in both staged and grain modes, with
no numerical tolerance. All repeated Sven subsets match the starting branch's
fits and extracted spots exactly; all spot files match byte for byte.
Four-grain Sven comparisons
also match the master scientific algorithms with bounded pixel input; eager
master loading would decode 322 GiB. Cancel and Ctrl+C are exercised during
real Sven reads. Request-to-cleanup times were 16.8 ms and 5.2 ms respectively;
simulated UI input plus dispatch took 3.15 s and 0.17 s, separately.

A diagnostic 52-grain run measures 39.70 s in native FCH5 sampling (reads,
decoding, sparse validation and gathering), 8.73 s in cache queries including
restoration, and 3.39 s in cache admission. These nested timers overlap and the
diagnostic run is excluded from the timing table. Native sampling remains the
largest measured cost.

An additional SIMD column-validation experiment produced 59.91 s and 65.41 s,
without a repeatable gain over the retained build's 62.24 s and 62.39 s. It was
discarded. The implementation and timings above refer to the retained build.

The 10× target is a target, not a measured result. Larger cumulative gains
against earlier branches must not be presented as gains against the starting
commit for this change.

Sources: [Boost.Unordered](https://www.boost.org/doc/libs/1_82_0/libs/unordered/doc/html/unordered.html),
[C-Blosc context API](https://github.com/Blosc/c-blosc/blob/main/blosc/blosc.h),
[GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc-13.3.0/gcc/Optimize-Options.html).
