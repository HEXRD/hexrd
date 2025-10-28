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

# The base needs to have the same python version (3.11, right now)
conda install --override-channels -c conda-forge python=3.11 -y

# Set up the hexrd channel
conda create --override-channels -c conda-forge -y -n hexrd python=3.11
conda activate hexrd

# Install the libmamba solver (it is much faster)
conda install -n base -c conda-forge conda-libmamba-solver
conda config --set solver libmamba
conda config --set conda_build.pkg_format 1 # force tar.bz2 files


# Remove anaconda telemetry (it is causing errors for us)
conda remove -n base conda-anaconda-telemetry

# Install conda build and create output directory
conda install --override-channels -c conda-forge conda-build -y
mkdir output

# Build the package
conda build --override-channels -c conda-forge --output-folder output/ conda.recipe/
