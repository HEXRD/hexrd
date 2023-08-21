#!/usr/bin/env bash

SCRIPT_DIR=`dirname "$0"`
pushd .
cd $SCRIPT_DIR

# sphinx-apidoc argument explanation:
# -d 1        - only use a depth of one for table of contents (TOC)
# -T          - don't generate the modules.rst TOC file
# -e          - put documentation for each module on its own page
# -f          - overwrite previously existing files
# -o source/  - place the output files into the source directory
# ../hexrd    - the path to the root source directory
# Extra arguments at the end are exclude patterns
sphinx-apidoc -d 1 -T -e -f -o source/ ../hexrd

popd
