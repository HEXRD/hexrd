
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION void
make_rmat_of_expmap(double *ePtr, double *rPtr)
{
    double e0 = ePtr[0];
    double e1 = ePtr[1];
    double e2 = ePtr[2];
    double sqr_phi = e0*e0 + e1*e1 + e2*e2;

    if ( sqr_phi > epsf ) {
        double phi = sqrt(sqr_phi);
        double s = sin(phi)/phi;
        double c = (1.0-cos(phi))/sqr_phi;

        rPtr[0] =  1.0  - c*(e1*e1 + e2*e2);
        rPtr[1] = -s*e2 + c*e0*e1;
        rPtr[2] =  s*e1 + c*e0*e2;
        rPtr[3] =  s*e2 + c*e1*e0;
        rPtr[4] =  1.0  - c*(e2*e2 + e0*e0);
        rPtr[5] = -s*e0 + c*e1*e2;
        rPtr[6] = -s*e1 + c*e2*e0;
        rPtr[7] =  s*e0 + c*e2*e1;
        rPtr[8] =  1.0  - c*(e0*e0 + e1*e1);
    } else {
        matrix33_set_identity((matrix33*)rPtr);
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_makeRotMatOfExpMap =
    "c module implementation of make_rmat_of_expmap.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_makeRotMatOfExpMap(PyObject * self, PyObject * args)
{
    PyArrayObject *expMap, *rMat;
    int de;
    npy_intp ne, dims[2];
    double *ePtr, *rPtr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"O", &expMap)) return(NULL);
    if ( expMap == NULL ) return(NULL);

    /* Verify shape of input arrays */
    de = PyArray_NDIM(expMap);
    assert( de == 1 );

    /* Verify dimensions of input arrays */
    ne = PyArray_DIMS(expMap)[0];
    assert( ne == 3 );

    /* Allocate the result matrix with appropriate dimensions and type */
    dims[0] = 3; dims[1] = 3;
    rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

    /* Grab pointers to the various data arrays */
    ePtr = (double*)PyArray_DATA(expMap);
    rPtr = (double*)PyArray_DATA(rMat);

    /* Call the actual function */
    make_rmat_of_expmap(ePtr,rPtr);

    return((PyObject*)rMat);
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
