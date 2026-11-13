# Per-frame beam energy

Use per-frame beam energies to correct spot prediction and grain fitting when
the beam energy changes during a scan. Keep the instrument's nominal energy as
the starting value.

Add these settings to your fit-grains YAML:

```yaml
fit_grains:
  beam_energy_file: energies.txt
  beam_energy_iterations: 1
```

`energies.txt` contains one absolute energy in **keV** per frame:

```text
# Energy in keV
41.5
42.0
42.5
```

Use one column of finite, positive values in image-frame order. The number of
values must match every detector's loaded frames, after any frame selection.
Blank lines and `#` comments are ignored. The same file applies to all detectors.
Relative paths use `working_dir`. For the CLI, this defaults to the directory
where you run the command.

`beam_energy_iterations` controls prediction correction passes. The default is
one; larger values repeat the correction. Zero disables prediction correction,
but predictions remain limited to the energy trace's omega coverage. Fitting
still uses the supplied energies at each spot's measured omega.
To run without energy correction, omit the file and any `beam_energy` metadata.

Instead of a text file, you can supply a one-dimensional `beam_energy` array in
image-series metadata, with one energy in keV per frame alongside the usual
`omega` metadata. A configured text file takes precedence over this metadata.

Keep these limits in mind:

- Orientation indexing and instrument calibration still use nominal energy.
  Use constant-energy scans for instrument calibration.
- Omega ranges must increase and must not overlap modulo 360 degrees; trim
  overscan frames. Spot extraction assumes synchronized, uniform omega steps.
  Material reflection selection must cover the energy range.
- Abrupt energy jumps can prevent prediction convergence. Fitting uses one
  energy per spot centroid, so large changes across a rocking curve need care.
