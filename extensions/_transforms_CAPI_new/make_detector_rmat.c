
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION void
make_detector_rmat(double * tPtr, double * rPtr)
{
    int i;
    double c[3],s[3];

    for (i=0; i<3; i++) {
        c[i] = cos(tPtr[i]);
        s[i] = sin(tPtr[i]);
    }

    rPtr[0] =  c[1]*c[2];
    rPtr[1] =  s[0]*s[1]*c[2]-c[0]*s[2];
    rPtr[2] =  c[0]*s[1]*c[2]+s[0]*s[2];
    rPtr[3] =  c[1]*s[2];
    rPtr[4] =  s[0]*s[1]*s[2]+c[0]*c[2];
    rPtr[5] =  c[0]*s[1]*s[2]-s[0]*c[2];
    rPtr[6] = -s[1];
    rPtr[7] =  s[0]*c[1];
    rPtr[8] =  c[0]*c[1];
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_makeDetectorRotMat =
    "c module implementation of makeDetectorRotMat.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_makeDetectorRotMat(PyObject * self, PyObject * args)
{
    PyArrayObject *tiltAngles, *rMat;
    int dt;
    npy_intp nt, dims[2];
    double *tPtr, *rPtr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"O", &tiltAngles)) return(NULL);
    if ( tiltAngles == NULL ) return(NULL);

    /* Verify shape of input arrays */
    dt = PyArray_NDIM(tiltAngles);
    assert( dt == 1 );

    /* Verify dimensions of input arrays */
    nt = PyArray_DIMS(tiltAngles)[0];
    assert( nt == 3 );

    /* Allocate the result matrix with appropriate dimensions and type */
    dims[0] = 3; dims[1] = 3;
    rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

    /* Grab pointers to the various data arrays */
    tPtr = (double*)PyArray_DATA(tiltAngles);
    rPtr = (double*)PyArray_DATA(rMat);

    /* Call the actual function */
    make_detector_rmat(tPtr,rPtr);

    return((PyObject*)rMat);
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
