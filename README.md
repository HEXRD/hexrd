![conda-package](https://github.com/HEXRD/hexrd/workflows/conda-package/badge.svg)  ![test](https://github.com/HEXRD/hexrd/workflows/test/badge.svg) ![latest version](https://anaconda.org/hexrd/hexrd/badges/version.svg) ![last updated](https://anaconda.org/hexrd/hexrd/badges/latest_release_relative_date.svg) ![downloads](https://anaconda.org/hexrd/hexrd/badges/downloads.svg)
# HEXRD
The HEXRD project is developing a cross-platform, open-source library for the general analysis of X-ray diffraction data.  This includes powder diffraction, Laue diffraction, and High Energy Diffraction Microscopy (_a.k.a._ 3DXRD, multi-grain rotation method) modalities.  At its core, HEXRD provides an abstraction of a generic diffraction instrument with support for multiple detectors.  This includes optimized transforms from the direct and reciprocal crystal lattices to the local detector coordinates, harnesses for interpolating image data into scattering angle coordinates, and sophisticated calibration routines.

# Installing

Requires Python 3.8+

### OSX

On OSX `hexrd` requires Python from conda-forge, to ensure it is built with the
latest SDK. See the following issue for more details: https://github.com/HEXRD/hexrdgui/issues/505.
This can be installed using the following command:

```bash
conda install -c conda-forge python=3.8
```

#### Big Sur (OS X 11)

OS X 11 does not work with the Python from conda-forge. Please install a the version
from the HEXRD channel

```bash
conda install -c hexrd python=3.8.4
```

## conda (release)

To install the latest stable release

```bash
conda install -c hexrd -c conda-forge hexrd
```

## conda (prerelease)
To install the latest changes on master, do the following.  Note that this release may be unstable.

```bash
conda install -c hexrd/label/hexrd-prerelease -c hexrd -c conda-forge hexrd
```

# Run

There is currently a CLI for far-field HEDM analysis (a.k.a. 3DXRD)

```bash
hexrd
```

# Development

Requires Python 3.8+ and a C compiler (_e.g._, `gcc`).  First clone the Git repository

```bash
git clone https://github.com/HEXRD/hexrd.git
```

## pip

```bash
# For now we need to explicitly install hexrd, until we push it to PyPI
pip install -e hexrd
```

## conda
It is highly recommended to install hexrd in its own virtual env

```bash
conda create --name hexrd
conda activate hexrd
```

### Linux
```bash
# First, make sure python3.8+ is installed in your target env.
# If it is not, run the following command:
conda install -c conda-forge python=3.8
# Install deps using conda package
conda install -c hexrd -c conda-forge hexrd
# Now using pip to link repo's into environment for development
CONDA_BUILD=1 pip install --no-build-isolation --no-deps -U -e hexrd
```

### Mac OS
```bash
# First, make sure python3.8+ is installed in your target env.
# On OSX you will need to use the Python package from conda-forge
# See the following issue for more details: https://github.com/HEXRD/hexrdgui/issues/505
conda install -c conda-forge python=3.8
# Install deps using conda package
conda install -c hexrd -c conda-forge hexrd
# Now using pip to link repo's into environment for development
CONDA_BUILD=1 pip install --no-build-isolation --no-deps -U -e hexrd
```

### Windows
```bash
# First, make sure python3.8+ is installed in your target env.
# If it is not, run the following command:
conda install -c conda-forge python=3.8
# Install deps using conda package
conda install -c hexrd -c conda-forge hexrd
# Now using pip to link repo's into environment for development
set CONDA_BUILD=1
pip install --no-build-isolation --no-deps -U -e hexrd
```
