#!/usr/bin/env bash
cd /github/workspace/

# Install dependencies
yum install -y wget git centos-release-scl eigen3-devel

# Install xsimd headers
git clone https://github.com/xtensor-stack/xsimd.git
mkdir -p /usr/include/xsimd
cp -r xsimd/include/xsimd/* /usr/include/xsimd/

# Need to install packages that depend on centos-release-scl on a different line.
# This will use gcc==10, which is the same as what manylinux2014 uses.
yum install -y devtoolset-10

# Enable this toolset
source /opt/rh/devtoolset-10/enable

# Download and install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b

# Set up the hexrd channel
${HOME}/miniconda3/bin/conda create --override-channels -c conda-forge -y -n hexrd python=3.9
${HOME}/miniconda3/bin/activate hexrd
${HOME}/miniconda3/bin/conda activate hexrd

# Install conda build and create output directory
${HOME}/miniconda3/bin/conda install --override-channels -c conda-forge conda-build -y
${HOME}/miniconda3/bin/conda install -c conda-forge pybind11 -y
mkdir output

# Build the package
${HOME}/miniconda3/bin/conda build --override-channels -c conda-forge --output-folder output/ conda.recipe/
