#!/usr/bin/env bash
cd /github/workspace/

# Use archive mirror for CentOS 7 until we are ready to migrate to CentOS 8
sed -i -e 's/mirrorlist/#mirrorlist/g' \
	-e 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' \
	/etc/yum.repos.d/CentOS-*

# Install dependencies
yum install -y wget git centos-release-scl ca-certificates

# Need to install packages that depend on centos-release-scl on a different line.
# This will use gcc==10, which is the same as what manylinux2014 uses.
yum install -y devtoolset-10

# Enable this toolset
source /opt/rh/devtoolset-10/enable

# Download and install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b

# The base needs to have the same python version (3.11, right now)
${HOME}/miniconda3/bin/conda install python=3.11 -y

# Set up the hexrd channel
${HOME}/miniconda3/bin/conda create --override-channels -c conda-forge -y -n hexrd python=3.11
${HOME}/miniconda3/bin/activate hexrd
${HOME}/miniconda3/bin/conda activate hexrd

# Install the libmamba solver (it is much faster)
${HOME}/miniconda3/bin/conda install -n base -c conda-forge conda-libmamba-solver
${HOME}/miniconda3/bin/conda config --set solver libmamba
${HOME}/miniconda3/bin/conda config --set conda_build.pkg_format 1 # force tar.bz2 files


# Remove anaconda telemetry (it is causing errors for us)
${HOME}/miniconda3/bin/conda remove -n base conda-anaconda-telemetry

# Install conda build and create output directory
${HOME}/miniconda3/bin/conda install --override-channels -c conda-forge conda-build -y
mkdir output

# Build the package
${HOME}/miniconda3/bin/conda build --override-channels -c conda-forge --output-folder output/ conda.recipe/
