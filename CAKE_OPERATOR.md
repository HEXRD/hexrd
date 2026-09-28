# Persistent cake matrices

The first public implementation exports the agreed **provisional CSR format**.
The matrix is ordinary SciPy-compatible CSR; geometry, the uniform grid, and
compact detector-pixel mapping are saved alongside it. No output mask, duplicate
panel names/shapes, or composite-frame shape/indices are stored.

```text
example.hexrdcake
  attributes: format = "HEXRD cake matrix", version = "draft-csr-1"
  instrument/                 Embedded instrument snapshot
  grid/
    tth                       [start, step, count]
    eta                       [start, step, count]
  input/
    panel_offsets
    pixel_indices
  matrix/
    attributes: format = "csr", shape = [noutput, ninput]
    data
    indices
    indptr
```

Angles are degrees. `start` is the first bin's lower edge; output order is
`(eta, tth)` in C order. Input segments follow lexicographically sorted embedded
instrument detector names, including empty segments for unused panels. Local
pixel indices are sorted, unique, zero-based C-order positions within each panel.
`panel_offsets` has one more entry than the number of panels. Matrix columns
index this compact input vector. Empty matrix rows define missing coverage.
Grid counts are positive integers stored in the numeric triples. The effective
upper edge is `start + step * count`, which can differ slightly from a requested
upper bound because PolarView rounds the number of bins.

## Build and use

```python
from hexrd.core.cake import CakeMatrix, CakeOperatorSpec, export_cake_matrix

spec = CakeOperatorSpec((0, 16), (-180, 180), (0.04, 5))  # Coarse test grid.
export_cake_matrix('example.hexrdcake', instrument, spec)
operator = CakeMatrix.load('example.hexrdcake')
x = operator.gather_active_composite_pixels(frame)
cake = (operator.matrix @ x).reshape(operator.shape)
cake[operator.mask] = float('nan')
profile = np.nanmean(cake, axis=0)
```

Import NumPy as `np` for the profile reduction. `operator.apply(panel_images)`
and `operator.apply_composite(frame)` also return masked polar images.
`apply_active_pixels` accepts `(pixels,)` or `(pixels, frames)` and returns
`(eta, tth)` or `(eta, tth, frames)` respectively. Choose bounded batches.
`tth_centers` and `eta_centers` expose the angular axes without a PolarView.
The embedded instrument is constructed lazily through `operator.instrument`;
application reads only saved arrays and the panel layout. Composite gather
indices are cached in memory and rebuilt when the incoming frame shape changes.

The loader checks format/version, grid values, instrument dimensions, ROIs,
panel boundaries, local pixel bounds, CSR shape/pointers/indices, and finite
coefficients before applying the matrix. `verify_cake_matrix` runs these checks.
There are no mandatory cryptographic hashes in this simple format.

## Export behavior and numerical semantics

- Export writes a temporary sibling and atomically replaces the destination
  after success. `overwrite=False` is the default. Raising from the optional
  `update_progress` callback cancels export and preserves the destination.
- A contribution is omitted when any of its four source pixels is invalid in
  the static instrument panel buffer. Weights are not renormalized.
- Each output row retains original panel/contribution order; input sorting
  remaps columns without changing accumulation order. CSR duplicates and zero
  coefficients are retained. Standard sparse multiplication may still differ
  in low floating-point bits from other executors; use tolerance comparisons.
- Dynamic NaNs propagate through multiplication, including zero-weight terms.
  Masked input arrays must be explicitly filled. Caller-side `np.nanmean`
  ignores NaN results and static missing coverage after filling with NaN.
- The file does not prescribe dark correction, thresholds, omega integration,
  wedge reduction, image decoding, scan metadata, or result writing.
- Current construction projects bounded eta chunks into an intermediate
  bilinear archive, then streams its coefficients to CSR. This bounds geometry
  memory but entails an extra disk pass and temporary disk usage. Direct CSR
  streaming is a follow-up optimization.
- HDF5 numeric datasets use portable gzip-1, shuffle and Fletcher32 when large;
  consumers read their storage metadata directly. No required GPU dependencies.

`CakeOperator`, `export_cake_operator`, and their provisional version-1/2 readers
remain for the earlier bilinear prototypes and research compatibility. New GUI
exports use `CakeMatrix`/`export_cake_matrix`. The old cached `PolarView` API and
interactive GUI rendering are unchanged.

## Verification and next steps

Coarse core tests cover sorted input ordering independent of contribution
ordering, ordinary CSR multiplication, panel/composite input, padded composite
frames, batches, lazy instrument loading, invalid archives, static masks,
NaNs, and cancellation atomicity. The GUI branch exports through this API and
has a menu-to-file-to-core-loader numerical test.

Realistic CHESS example evidence lives outside this repository in the workspace's
`hexrd-wip-branches/cake-data-research/realistic-2026-09-28/` bundle. Its large
matrix is a demonstration artifact, not a dependency of the unit tests.

Alessandro owns CLI/workflow integration. Follow-up decisions are command naming,
which geometry flags conflict with a supplied matrix, center-based wedge
selection and partial edge wedges, NaN handling and full-profile normalization,
and preserving existing omega averaging/flooring behavior. The current format
remains provisional until that integration is agreed and exercised end to end.
