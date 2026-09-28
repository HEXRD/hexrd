# Cake detector data

The `cake-data` command creates azimuthally segmented powder profiles from a
composite detector image series. It writes `<sample-name>.h5`, using the
historical CakeData HDF5 layout, and a `<sample-name>.par` sidecar.

Run `hexrd cake-data --help` for all options. Every processing parameter has a
short and long option. Parameters may instead be placed in one JSON object:

```console
hexrd cake-data --config cake-data.json
```

Command-line values override values in the JSON file. Generate a template with:

```console
hexrd cake-data --generate-default-config > cake-data.json
```

The required JSON keys are `data_dir`, `output_dir`, `sample_name`,
`instrument`, and `par_file`. `data_dir` may be a literal directory or a
template containing `{sample_name}` and `{image_number}`. The command builds a
single sparse bilinear interpolation matrix for the requested full eta range,
then uses matrix multiplication for every omega-integrated frame. Eta profiles
are sliced from that result at `cake_width` intervals.
