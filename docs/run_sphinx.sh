#!/usr/bin/env bash

SCRIPT_DIR=`dirname "$0"`
pushd .
cd $SCRIPT_DIR

# Clean up the current documentation
make clean html

# Run the apidoc command
./run_apidoc.sh

# Build the html files
make html

popd
