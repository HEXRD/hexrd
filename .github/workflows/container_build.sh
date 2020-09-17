#!/usr/bin/env bash
cd /github/workspace/
yum install -y wget git gcc
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
${HOME}/miniconda3/bin/conda create -y -n hexrd
${HOME}/miniconda3/bin/activate hexrd
${HOME}/miniconda3/bin/conda activate hexrd
${HOME}/miniconda3/bin/conda install conda-build
mkdir output
${HOME}/miniconda3/bin/conda build -c anaconda -c conda-forge --output-folder output/ conda.recipe/
