rm -rf build

# Use the xsimd and eigen headers from the conda host environment rather than
# downloading them at build time (see get_include_path() in setup.py).
export XSIMD_INCLUDE_DIR="$PREFIX/include"
export EIGEN3_INCLUDE_DIR="$PREFIX/include/eigen3"

$PYTHON setup.py install --single-version-externally-managed --record=record.txt
