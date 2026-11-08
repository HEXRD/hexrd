# Fit-specific sparse input experiment

**Result: 4.00× faster fitting on 2,103 real Sven grains, with exact outputs
and 3.96 GiB peak process-tree RSS.** This is an input-preparation experiment
using existing execution paths. The fitting engine and original dataset/YAML
remain unchanged. The experimental preparation command is now included in the
repository.

Initial measurements were collected on November 7, 2026, at `5020b857`.
On November 8, the working tree was reconciled with that implementation:
55 runtime/test/research files matched the older `fda659f6` snapshot exactly.
Independent ignore-file edits were preserved. A fresh full-size comparison used
a frozen copy of the current grain input, which had changed since November 7.
Detailed measurements and validation scope are in
[the experiment record](fit-grains-input-cache-experiment.json).

## What changed

The complete Sven scan contained no max-uint32 sentinel values. Instead,
99.03% of its stored entries were unnecessary at the existing fitting threshold
of 1,000. A separate fit-only FCH5 cache removes finite, nonnegative values at
or below that threshold, retaining negative values, signed negative zero and
nonfinite values. Source coordinates were checked for sorted, unique entries
before filtering. Retained float64 values, frame order and omega metadata are
unchanged.

This reduces the file from **44.30 GB to 816 MB** and stored entries from
28.83 billion to 279 million. The compact sparse representation fits in RAM,
allowing the existing eager reader to avoid repeated disk reads/decompression.
Frames are not expanded into a resident dense image stack.

## Measured performance

Single runs, identical scientific settings and grain inputs, with spot-file
writing enabled. Times include image loading inside `fit_grains`; experiment
construction adds approximately 0.5–0.6 seconds. Each run started in a fresh
process, but filesystem caches were not globally flushed.

| Grains | Input / access | Fitting workers | Fit time | Speedup vs original | Peak tree RSS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 160 | Original / lazy | 2 | 71.23 s | 1.00× | 2.31 GiB |
| 160 | Compact / lazy | 2 | 41.96 s | 1.70× | 1.95 GiB |
| 160 | Compact / eager | 1 | 25.80 s | 2.76× | 3.95 GiB |
| 2,103 | Original / lazy | 2 | 763.98 s | 1.00× | 2.28 GiB |
| 2,103 | Compact / eager | 1 | 191.03 s | 4.00× | 3.96 GiB |

The 160-grain measurements are from November 7; the full-size measurements are
the reconciled November 8 run. Each comparison uses identical grain-input bytes.
The earlier full-size result (717.99 → 183.90 seconds, 3.90×) is retained in the
experiment record with its separate input hash.

Preparation took **648.52 seconds**, with 270 MiB peak tree RSS. Including
preparation, the first full run takes **839.54 seconds**, versus 763.98 seconds
without preparation. Reusing the cache breaks even on the **second full run**
(the 160-grain sample needs 15 runs). The 4.00× figure excludes preparation;
it combines input reduction and a change in memory-access mode/worker count.
Preparation was timed with the local prototype; the published command generalizes
that implementation and adds an output-flush byte limit. The large cache was
reused during reconciliation.

Runs were sequential with single-threaded BLAS. Lazy benchmarks used direct
reads, a 1 GiB aggregate cache, 1.5 GiB address-space limits per process and a
3 GiB process-tree RSS watchdog. Eager benchmarks used one fitting process,
two image-loading threads and a hard **5 GiB address-space limit**. The YAML
cache allowance alone is not a hard total-memory limit, particularly in eager
mode. These figures do not establish memory safety for arbitrary worker counts
or the interactive CLI's process arrangement.

## Correctness and scope

- All 2,103 fits match the original-input branch run bitwise. All 2,103 spot
  files and `grains.out` match byte-for-byte; `grains.npz` arrays match bitwise.
  The 160-grain lazy and eager comparisons also pass these checks.
- Four Sven grains from the initial validation, including returned spot arrays, match master's
  extraction and optimizer bitwise. Master used bounded pixel input because
  its eager loading would exceed local RAM; this is not an end-to-end master
  performance comparison or a master run over all 2,103 grains.
- All 18 example runs (nine studies, both measurement modes, original example
  images) match the saved master reference bitwise. All 452 focused tests pass;
  800 additional threshold-filtering statistics cases match bitwise.
- Eight preparation checks also pass, for 460 focused tests total. The guarded
  no-TUI CLI was checked on four current grains: fitted results and written
  outputs match the fresh reference, with 4.23 GiB peak tree RSS including the
  external memory sampler. The master reference is `ca51e7d5`; key reference
  source files were verified against that commit.

The compact cache is specific to nearest-neighbor thresholded fitting at
1,000 or higher. It does not preserve background pixels for lower thresholds,
interpolation, orientation finding or raw-patch output. Its threshold metadata
is informational: existing readers do not enforce this restriction. Preserve
the original input. A saved panel mask alone is insufficient justification for
deleting arbitrary sentinel pixels from other datasets.

Local preparation, configurations and evidence are under
`.local-bench/invalid-pixels/`: `prepare.py`, `sven-fit-threshold1000.fch5`,
`fit-only-eager.yml`, `large.py`, `exact-eager.json` and
`examples-bitwise.json`. The generalized preparation command is
[`scripts/prepare_fit_grains_cache.py`](../scripts/prepare_fit_grains_cache.py);
large caches and benchmark artifacts remain local. The matching Python 3.12
native extension and CLI launcher were restored locally before benchmarking, with backups;
no speedup is attributed to comparison against a stale binary.

## Use the prepared input

On Linux, prepare a separate cache once, keeping the original file:

```bash
mkdir -p .local-bench
python scripts/prepare_fit_grains_cache.py \
  sven_dataset/data/images_median_and_25_counts_removed_eiger.fch5 \
  .local-bench/sven-fit-threshold1000.fch5 --threshold 1000
```

The command uses two workers, four queued frames, hard 1.5 GiB allocation
limits per process, direct input reads and bounded output flushing. It rejects
duplicate/unsorted coordinates, invalid frame bounds and existing output files.
Eight regression checks cover exact retained values and metadata, malformed coordinates and
threshold restrictions. A failed preparation leaves an `.incomplete` file;
completed caches are published only after all frames succeed.

For fitting, copy the experiment YAML, point its image-series file at the
compact cache and keep the scientific settings. Use `style: fch5` with
`max_workers: 2`, `multiprocessing: 1`, and `fit_grains.image_cache.mode: eager`.
Keep an explicit estimate path so fitting does not need orientation finding.
The local reconciled configuration is `sven_dataset/fitting_compact.yml`.

For a guarded command without the TUI, with the local configuration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -c \
  'import resource; resource.setrlimit(resource.RLIMIT_AS, (5*1024**3,)*2); from hexrd.hedm.cli.main import main; main()' \
  fit-grains --no-tui sven_dataset/fitting_compact.yml
```

The fitting CLI and dashboard remain the existing implementation. The new
preparation command is opt-in; threshold restrictions are not enforced when a
compact file is opened through other commands.

## Next performance targets

1. Produce a fit-specific representation during initial image reduction to
   eliminate the separate 44 GB preparation scan. A supported implementation
   needs explicit threshold/use restrictions and access to the original data.
2. Reduce eager-reader overhead. The diagnostic profile identifies repeated
   HDF5 reads and COO-to-CSR conversion; canonical sparse input could avoid
   redundant reconstruction after validating its ordering.
3. Reuse immutable detector geometry and panel masks across grains/passes.
   The profile also identifies repeated panel-buffer deep copies and spot-grid
   planning. Measure these changes at 2,000+ grains before claiming a gain.

The multithreaded diagnostic profile is evidence for investigation, not an
additive time budget or proof of another achievable 2× improvement.
