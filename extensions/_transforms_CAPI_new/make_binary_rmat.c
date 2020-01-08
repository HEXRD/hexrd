
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION void
make_binary_rmat(double *aPtr, double *rPtr)
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
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_makeBinaryRotMat =
    "c module implementation of make_binary_rmat.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_makeBinaryRotMat(PyObject * self, PyObject * args)
{
    PyArrayObject *axis, *rMat;
    int da;
    npy_intp na, dims[2];
    double *aPtr, *rPtr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"O", &axis)) return(NULL);
    if ( axis  == NULL ) return(NULL);

    /* Verify shape of input arrays */
    da = PyArray_NDIM(axis);
    assert( da == 1 );

    /* Verify dimensions of input arrays */
    na = PyArray_DIMS(axis)[0];
    assert( na == 3 );

    /* Allocate the result matrix with appropriate dimensions and type */
    dims[0] = 3; dims[1] = 3;
    rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

    /* Grab pointers to the various data arrays */
    aPtr = (double*)PyArray_DATA(axis);
    rPtr = (double*)PyArray_DATA(rMat);

    /* Call the actual function */
    make_binary_rmat(aPtr,rPtr);

    return((PyObject*)rMat);
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
