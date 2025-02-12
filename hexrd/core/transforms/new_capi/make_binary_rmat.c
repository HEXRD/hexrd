
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION void
make_binary_rmat(const double *aPtr, double * restrict rPtr)
{
    int i, j;

    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            rPtr[3*i+j] = 2.0*aPtr[i]*aPtr[j];
        }
        rPtr[3*i+i] -= 1.0;
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#    include "ndargs_helper.h"
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_makeBinaryRotMat =
    "c module implementation of make_binary_rmat.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_makeBinaryRotMat(PyObject * self, PyObject * args)
{
    nah_array axis = { NULL, "axis", NAH_TYPE_DP_FP, { 3 }};
    PyArrayObject *result = NULL;
    npy_intp dims[2] = { 3, 3 };

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "O&",
                          nah_array_converter, &axis))
        return NULL;

    /* Allocate the result matrix with appropriate dimensions and type */
    result = (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (NULL == result)
        goto fail_alloc;
    
    /* Call the actual function */
    make_binary_rmat((double *)PyArray_DATA(axis.pyarray),
                     (double *)PyArray_DATA(result));

    return (PyObject*)result;

 fail_alloc:
    Py_XDECREF(result);

    return PyErr_NoMemory();
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
