rmdir build /s /q

REM Use the xsimd and eigen headers from the conda host environment rather than
REM downloading them at build time (see get_include_path() in setup.py).
REM On Windows, conda places headers under %LIBRARY_PREFIX%\include.
set "XSIMD_INCLUDE_DIR=%LIBRARY_PREFIX%\include"
set "EIGEN3_INCLUDE_DIR=%LIBRARY_PREFIX%\include\eigen3"

"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
