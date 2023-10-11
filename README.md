[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8033939.svg)](https://doi.org/10.5281/zenodo.8033939) ![conda-package](https://github.com/HEXRD/hexrd/workflows/conda-package/badge.svg) ![test](https://github.com/HEXRD/hexrd/workflows/test/badge.svg) ![latest version](https://anaconda.org/hexrd/hexrd/badges/version.svg) ![last updated](https://anaconda.org/hexrd/hexrd/badges/latest_release_relative_date.svg) ![downloads](https://anaconda.org/hexrd/hexrd/badges/downloads.svg)
# HEXRD
The HEXRD project is developing a cross-platform, open-source library for the general analysis of X-ray diffraction data.  This includes powder diffraction, Laue diffraction, and High Energy Diffraction Microscopy (_a.k.a._ 3DXRD, multi-grain rotation method) modalities.  At its core, HEXRD provides an abstraction of a generic diffraction instrument with support for multiple detectors.  This includes optimized transforms from the direct and reciprocal crystal lattices to the local detector coordinates, harnesses for interpolating image data into scattering angle coordinates, and sophisticated calibration routines.

# Installing

Requires Python 3.8+ (Python 3.9 recommended).  It is generally recommended for interactive use to install `hexrd` into a fresh environment with Spyder and Jupyter as well.  The GUI is in a separate package [here](https://github.com/hexrd/hexrdgui).

## conda (main releases)

To install the latest stable release, do:

```bash
conda install -c hexrd -c conda-forge hexrd
```

## conda (prerelease)
To install the latest changes on master, do the following (Note that this release may be unstable!):

```bash
conda install -c hexrd/label/hexrd-prerelease -c hexrd -c conda-forge hexrd
```

# Run

There is currently a CLI for far-field HEDM analysis (a.k.a. 3DXRD).  Other CLI's for high-level functions are under development.

```bash
> hexrd --help

usage: hexrd [-h] [--debug] [--inst-profile INST_PROFILE] [--version] command ...

High energy diffraction data analysis

positional arguments:
  command
    help                Displays a list of available conda commands and their help strings.
    test                runs the hexrd test suite
    documentation       Launches the hexrd documentation (work in progress) in a web browser
    find-orientations   Process rotation image series to find grain orientations
    fit-grains          Extracts G vectors, grain position and strain
    pickle23            modify old material files (pickles) to be compatible with hexrd3; it makes a backup and overwrites the
                        original file

optional arguments:
  -h, --help            show this help message and exit
  --debug               verbose reporting
  --inst-profile INST_PROFILE
                        use the following files as source for functions to instrument
  --version             show program's version number and exit

```

# Citing

We are very glad you found our software helpful for your research! In order for us to keep track of the impact our software is having, can you please cite us in your papers?

See [Citing HEXRD](https://hexrdgui.readthedocs.io/en/latest/citing/#hexrd) for more information.

# Development

Requires Python 3.8+ and a C compiler (_e.g._, `gcc` or VisualStudio).  First clone the Git repository:

```bash
git clone https://github.com/hexrd/hexrd.git
```

## pip

```bash
# For now we need to explicitly install hexrd, until we push it to PyPI
pip install -e hexrd
```

## conda
It is highly recommended to install hexrd in its own virtual env

```bash
conda create --name hexrd-dev python=3.9 hexrd
conda activate hexrd-dev
```

### Linux and Mac OS
```bash
# First, make sure python3.8+ is installed in your target env.
# If it is not, run the following command:
conda install -c conda-forge python=3.9

# Install deps using conda package
conda install -c hexrd/label/hexrd-prerelease -c hexrd -c conda-forge hexrd

# Now using pip to link repo's into environment for development
CONDA_BUILD=1 pip install --no-build-isolation --no-deps -U -e hexrd
```

### Windows
```bash
# First, make sure python3.8+ is installed in your target env.
# If it is not, run the following command:
conda install -c conda-forge python=3.9

# Install deps using conda package
conda install -c hexrd/label/hexrd-prerelease -c hexrd -c conda-forge hexrd

# Now using pip to link repo's into environment for development
set CONDA_BUILD=1
pip install --no-build-isolation --no-deps -U -e hexrd
```

Have fun!

# Authors

Many thanks go to Hexrd's [contributors](https://github.com/HEXRD/hexrd/graphs/contributors).

* Joel Bernier
* Donald Boyce
* Saransh Singh
* Darren Pagan
* Kelly Nygren
* Rachel Lim
* Patrick Avery
* Chris Harris
* Nathan Barton
* and many more...

# Contributing

Please submit any bugfixes or feature improvements as [pull requests](https://help.github.com/articles/using-pull-requests/).

# License

Hexrd is distributed under the terms of the BSD 3-Clause license. All new contributions must be made under this license.

See [LICENSE](https://github.com/hexrd/hexrd/blob/master/LICENSE) and [NOTICE](https://github.com/hexrd/hexrd/blob/master/NOTICE) for details.

`SPDX-License-Identifier: BSD 3-Clause`

``LLNL-CODE-819716``
