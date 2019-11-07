Extensions source code
======================

This directory contains the source code for the different C extensions
used by hexrd.

The resulting code will appear as a python module under
'hexrd.extensions', with a name

Currently, the modules are:

```_transforms_CAPI```: The original transforms CAPI, that will appear
    as ```hexrd.extensions.transforms_CAPI``` in Python.

```_transforms_CAPI_new```: Code reorganized in multiple files. Easier
    to modify on a per-function basis. Will appear as
    ```hexrd.extensions.transforms_CAPI_new``` in Python



