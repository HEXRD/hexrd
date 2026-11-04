# Cake matrix file format

`.hexrdcake` is HDF5 with root attributes `format="HEXRD cake matrix"` and
`version="hexrd-cake-matrix-1"`. The reader also accepts `1` and `draft-csr-2`.

| Object | Contents |
| --- | --- |
| `/instrument` | Instrument configuration, including geometry, groups, ROIs, and static buffers |
| `/instrument` attribute `active_beam` | Selected beam name |
| `/grid/tth`, `/grid/eta` | `[start_edge_degrees, step_degrees, count]` |
| `/matrix` attribute `format` | `csr` |
| `/matrix` attribute `input_layout` | `composite` or `panels`; see [detector inputs](../users/cake-data.md) |
| `/matrix` attribute `shape` | `(eta_count * tth_count, input_pixel_count)` |
| `/matrix/data` | Float64 interpolation weights |
| `/matrix/indices`, `/matrix/indptr` | Integer column indices and row pointers |

Grid centers are `start + step * (arange(count) + 0.5)`. Output is flattened
in C order, with two-theta varying fastest. Input shape and panel offsets are
reconstructed from the instrument.

Each valid panel contribution retains all four bilinear neighbors, including
zero weights and duplicate columns, in instrument contribution order. A buffered
neighbor excludes that panel contribution; an empty row is masked.

The reader checks grid/layout consistency, finite weights, integer indices and
complete pointers before SciPy's full CSR format check. Loading closes the file;
contiguous eta slices share weights and indices with rebased row pointers.

The writer applies gzip level 1, shuffle, and Fletcher32 to CSR and numeric
instrument arrays. Compression is optional and lossless: the Eiger example
shrinks from 26.10 to 5.64 MiB, mainly through compressing panel masks.
See [h5py filters](https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline).
