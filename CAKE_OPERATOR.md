# Persistent cake matrices

The version `1` format stores a standard CSR matrix with
full-image columns. Composite images use ordinary `C @ frame.ravel()` with
no pixel extraction or column remapping. Interactive PolarView is unchanged.

- example.hexrdcake
  - Attributes: format = "HEXRD cake matrix", version = "1"
  - instrument/: Embedded instrument snapshot
  - grid/
    - tth: [start, step, count]
    - eta: [start, step, count]
  - matrix/
    - Attributes: format = "csr", shape, input_layout
    - data
    - indices
    - indptr

There is no input/ group, pixel-extraction list, stored output mask, duplicate
panel names/shapes, frame shape, or mandatory hash/provenance hierarchy.

## Input and output conventions

- `input_layout = "composite"`: columns address the complete C-order detector
  image. For subpanels, its shape is the maximum ROI stop in each dimension;
  a single panel without an ROI uses its panel dimensions. Unused columns store
  no coefficients. Image dimensions and orientation must match this layout.
- `input_layout = "panels"`: columns concatenate complete panel images in
  lexicographically sorted detector-name order, each in C order. No compact
  pixel selection is involved. This supports instruments with independent
  detector images or multiple physical detector groups.
- Automatic export chooses composite for one panel or disjoint ROIs in one
  detector group; otherwise it chooses panels. Callers can specify the layout.
- Geometry uses instrument-defined panel dimensions. Other HEXRD workflows
  can extract ROIs from a larger padded raw frame; that does not establish
  its total width from the instrument. This format deliberately requires the
  defined composite shape and rejects differently shaped input, including
  equal-size arrays with a different row/column shape. Apply any required image
  orientation changes before caking. Same-shaped flips cannot be detected.
- Output is `(eta, tth)` flattened in C order. Grid angles are degrees, start
  is a lower bin edge, and upper edge is `start + step * count`. Rounding may
  make that differ slightly from the originally requested range endpoint.
- Empty CSR rows indicate missing coverage and define the output mask.

## Build and apply

```python
import numpy as np
from hexrd.core.cake import CakeMatrix, CakeOperatorSpec, export_cake_matrix

spec = CakeOperatorSpec((0, 16), (-180, 180), (0.04, 5))
export_cake_matrix('example.hexrdcake', instrument, spec)
operator = CakeMatrix.load('example.hexrdcake')
assert frame.shape == operator.composite_shape
cake = (operator.matrix @ frame.ravel()).reshape(operator.shape)
cake[operator.mask] = np.nan
profile = np.nanmean(cake, axis=0)
```

Load once for the series. `apply_composite(frame)` checks the shape and returns
masked output without assembling panels or gathering selected pixels. A
contiguous frame's ravel is a view; noncontiguous layouts and integer-to-float
promotion may copy. `apply(panel_images)` remains an adapter for panel
dictionaries; composite archives assemble the full image from those panels.
Their ROIs must refer to the same physical composite. The direct composite
API is the batch path. Separate-panel archives assemble a full panel vector.

`matrix_for_composite(frame_shape)` validates and returns the stored matrix
without copying for new files. It can also prepare a legacy compact archive
once for a raw shape. `apply_active_pixels` and gather methods remain for
legacy callers; for new files their vectors contain all input pixels.
`tth_centers`, `eta_centers`, and lazy `instrument` expose saved geometry.

## Persistence and numerical behavior

The loader validates format/version, grid, dimensions, ROIs, layout, matrix
shape, pointers, indices, and finite coefficients. New files write atomically;
`overwrite=False` is the default. Raising from the progress callback preserves
an existing destination. Construction currently uses a bounded eta-chunk
bilinear archive followed by streamed CSR conversion, so an extra temporary
disk pass remains. Numeric datasets use gzip-1, shuffle and Fletcher32.

A bilinear contribution is omitted if any of its four pixels is invalid in the
static panel buffer. Coefficients are not renormalized. Original contribution
order, duplicate entries, and zero coefficients are preserved. Dynamic NaNs
propagate, including through zero-weight terms; callers choose reduction and
missing-data policies. Matrix application does not replace threshold-channel
combination, omega integration/flooring, dark correction, or orientation changes.
Do not reorder those steps without numerical verification.

The `draft-csr-1` compact reader remains supported, as do `CakeOperator` and
its earlier version-1/2 bilinear prototypes. New GUI exports use version 1; draft-csr-2 files remain readable.

## Verification and performance research

Coarse tests cover direct multiplication, instrument-derived dimensions,
separate panels/groups, single panels, masks, NaNs, invalid archives, legacy
loading, noncontiguous frames, batches, and atomic cancellation. The GUI test
exports via the actual handler, loads the file, and compares with PolarView.
Real 32-panel coarse and CHESS-resolution files are separate research artifacts;
both full-image files match their previous validated matrix output exactly.

Research under the workspace's `cake-data-research` compares ordinary SciPy,
parallel CSR rows, threaded row chunks, and cached PolarView on matching grids.
The optimized executors remain benchmark prototypes; the public default is
ordinary SciPy multiplication. Choose either frame-level or within-frame
parallelism with a shared worker budget, then benchmark full workflow throughput.
