#!/usr/bin/env bash
cd /github/workspace/

# Install dependencies
# Use gcc-14 for manylinux_2_28
yum install -y wget git gcc-toolset-14

# Enable this toolset
source /opt/rh/gcc-toolset-14/enable

# Download and install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b

# Activate conda
source $HOME/miniconda3/bin/activate

# Set up the hexrd channel
conda create --override-channels -c conda-forge -y -n hexrd python=3.11
conda activate hexrd

conda config --set conda_build.pkg_format 1 # force tar.bz2 files


# Remove anaconda telemetry (it is causing errors for us)
conda remove -n base conda-anaconda-telemetry

# Install conda build and create output directory
conda install --override-channels -c conda-forge conda-build -y

# conda-forge builds conda in an over-long placeholder prefix, so its `conda`
# entry point ships with a "#!/usr/bin/env python" shebang instead of an
# absolute one. During conda-build the host env (which has no conda) is first
# on PATH, so `conda` runs under that python and dies with "No module named
# 'conda'". Pin the shebang to this env's interpreter.
sed -i "1s|^#!/usr/bin/env python.*|#!${CONDA_PREFIX}/bin/python|" "${CONDA_PREFIX}/bin/conda"

mkdir output

# Build the package
conda-build --override-channels -c conda-forge --output-folder output/ conda.recipe/
