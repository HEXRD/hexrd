# Reusable cake matrices and `cake-data`

A `.hexrdcake` file maps detector pixels to two-theta and eta angles using a
reusable sparse matrix. Use it directly in Python for a 2D cake, or with
`hexrd cake-data` to turn image series and CHESS scan metadata into powder profiles.

## Try one image

The [Eiger ceria example](https://github.com/HEXRD/examples/tree/master/eiger/first_ceria)
includes an image, calibration, saved matrix, and `cake-multiply.py`.
From that directory, run `python -i cake-multiply.py`:

```python
import h5py
import numpy as np
from hexrd.powder.cake import CakeMatrix

matrix = CakeMatrix.load('cake_matrix.hexrdcake')
with h5py.File('ff_000_data_000001.h5') as stream:
    frame = stream['entry/data/data'][0]

cake = (matrix.matrix @ frame.ravel()).reshape(matrix.shape)
cake[matrix.mask] = np.nan
```

The result has eta rows and two-theta columns; NaNs mark uncovered bins.
`matrix.eta_centers` and `matrix.tth_centers` give their angles in degrees.
Alternatively, `matrix.apply(frame).filled(np.nan)` also checks the input shape.

## Create a matrix

In **HEXRDGUI**, load the calibration and images, select **Polar** view, set the
angular ranges and pixel sizes, then choose **File → Export → Cake Matrix**.
This requires a GUI build containing that action.

From **Python**, pass a calibrated `HEDMInstrument` to `export_cake_matrix`:

```python
from hexrd.powder.cake import export_cake_matrix

export_cake_matrix(
    'cake_matrix.hexrdcake', instrument,
    tth_range=(0, 13.9), eta_range=(-180, 180), pixel_size=(0.025, 1),
)
```

Angles are in degrees; `pixel_size` is `(two_theta_step, eta_step)`.
Use `overwrite=True` to replace a file. The example's
[create-cake-matrix.py](https://github.com/HEXRD/examples/blob/master/eiger/first_ceria/create-cake-matrix.py)
includes loading its calibration.

Rebuild after changing geometry, active beam, static panel buffers, or polar grid.
The matrix excludes dark subtraction, flips, display masks, intensity corrections,
SNIP, and polar distortion. Apply any needed image preprocessing yourself.

## Detector inputs

The default `input_layout='auto'` chooses:

| Layout | Input |
| --- | --- |
| `composite` | One image, flattened in C order. Used for a single panel or nonoverlapping ROIs within one group. |
| `panels` | Panel images flattened and concatenated in sorted panel-name order. Used for other instruments. |

Composite images must exactly match `matrix.input_shape`, including ROI offsets.
For multiple detectors, `apply()` and `assemble()` accept images keyed by **all
panel names** or **all group names**. Group inputs require a group and ROI on
every panel, with parent images large enough to cover those ROIs.

For **dual Dexelas** with eight subpanels in groups `ff1` and `ff2`:

```python
cake = matrix.apply({'ff1': image_ff1, 'ff2': image_ff2})
```

Orient the two parent images before passing them in; HEXRD crops the subpanels.
For direct multiplication, use `matrix.assemble(images).ravel()` as the input.

## Process a scan

The example's `cake-data.yaml` uses demonstration metadata for two frames:

```yaml
data_dir: .
output_dir: cake-data-output
sample_name: ceria
matrix: cake_matrix.hexrdcake
par_file: cake-data-input.par
image_pattern: ff_000_data_000001.h5
image_format: hdf5
image_options:
  path: entry/data
  dataname: data
ome_start: 0
ome_end: 2
dome: 1
cake_width: 5
```

```bash
hexrd cake-data --config cake-data.yaml
hexrd cake-data --help
hexrd cake-data --generate-default-config > cake-data-template.yaml
```

Paths resolve from the **working directory**. CLI options override YAML.
Each image glob must match exactly one file per layer. `data_dir` supports
`{sample_name}` and `{image_number:04d}` placeholders, or `%s` and `%04d`.
Use `instrument` (YAML or HDF5) instead of `matrix` to build a temporary matrix;
a saved matrix cannot be combined with instrument or geometry overrides.

For dual Dexelas, replace `image_pattern` with a mapping:

```yaml
image_pattern:
  ff1: "*ff1*.h5"
  ff2: "*ff2*.h5"
```

Use reader options matching your files. Both streams must share the format,
options, frame count, and synchronized frame indices. Optional
`image_flips: {ff1: h, ff2: v}` reverses rows in `ff1` and columns in `ff2`
**before cropping**; choose flips matching your calibration.

### Metadata and averaging

Input is a version-5 CHESS par file with at least 25 columns:

| Zero-based columns | Meaning |
| --- | --- |
| 3, 6 | Scan number, image number |
| 7, 8, 9 | Sample position |
| 24 | Frame count after `skip_frames` |

An optional same-stem JSON sidecar maps column numbers to names.
`fly_axis0_rising > fly_axis0_falling` reverses omega-bin order.

- `dome`: average complete omega intervals in float64, then floor before caking.
  `max_input_frames` limits processing to initial complete intervals.
- `cake_width`: average valid eta samples per wedge, then weight valid wedges
  equally. It must be at least one eta row; the final wedge may be partial.
- `start_layer`: start a new output at this layer; resuming is not supported.

### Outputs

The command replaces `<sample_name>.h5` and `<sample_name>.par` after processing.
The par output uses the legacy 12-column CakeData format. HDF5 stores `XYZ_key`
at the root and these datasets under `Cake/<layer>/`:

| Dataset | Contents |
| --- | --- |
| `xyz`, `tth` | Sample position; two-theta centers in degrees |
| `Ome/<bin>/IvTTH`, `Ome/<bin>/eta_key` | Wedge profiles; wedge indices and eta centers |
| `IvTTHvW`, `ome_key` | Combined profiles per omega interval; indices and omega centers |
| `IvTTH_Full`, `nFrames` | Mean profile over omega intervals; number of output intervals |

## Numerical behavior

The matrix uses bilinear interpolation and adds overlapping panel contributions.
Static buffers exclude affected panel contributions; runtime NaNs propagate
through the sparse product. `cake-data` converts the uint32 Eiger invalid-pixel
sentinel to NaN and ignores NaNs when averaging eta samples. Its final omega
mean propagates NaNs. Direct Python callers must handle invalid input pixels.

Files use lossless HDF5 compression. See the [format reference](../dev/cake-matrix.md)
for storage details.
