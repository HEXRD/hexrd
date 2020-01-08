
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
/* TODO: No printf should go here. Also, it should be possible to avoid malloc
 */
#  include <stdio.h>
#  include <sdlib.h>

#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION double
quat_distance(size_t nsym, double * q1, double * q2, double * qsym)
{
    size_t i;
    double q0, q0_max = 0.0, dist = 0.0;
    double *q2s;

    if ( NULL == (q2s = (double *)malloc(4*nsym*sizeof(double))) ) {
        printf("malloc failed\n");
        return(-1);
    }

    /* For each symmetry in qsym compute its inner product with q2 */
    for (i=0; i<nsym; i++) {
        q2s[4*i+0] = q2[0]*qsym[4*i+0] - q2[1]*qsym[4*i+1] - q2[2]*qsym[4*i+2] - q2[3]*qsym[4*i+3];
        q2s[4*i+1] = q2[1]*qsym[4*i+0] + q2[0]*qsym[4*i+1] - q2[3]*qsym[4*i+2] + q2[2]*qsym[4*i+3];
        q2s[4*i+2] = q2[2]*qsym[4*i+0] + q2[3]*qsym[4*i+1] + q2[0]*qsym[4*i+2] - q2[1]*qsym[4*i+3];
        q2s[4*i+3] = q2[3]*qsym[4*i+0] - q2[2]*qsym[4*i+1] + q2[1]*qsym[4*i+2] + q2[0]*qsym[4*i+3];
    }

    /* For each symmetric equivalent q2 compute its inner product with inv(q1) */
    for (i=0; i<nsym; i++) {
        q0 = q1[0]*q2s[4*i+0] + q1[1]*q2s[4*i+1] + q1[2]*q2s[4*i+2] + q1[3]*q2s[4*i+3];
        if ( fabs(q0) > q0_max ) {
            q0_max = fabs(q0);
        }
    }

    if ( q0_max <= 1.0 )
        dist = 2.0*acos(q0_max);
    else if ( q0_max - 1. < 1e-12 )
        /* in case of quats loaded from single precision file */
        dist = 0.;
    else
        dist = NAN;

    free(q2s);

    return(dist);
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_quat_distance =
    "c module implementation of quat_distance.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_quat_distance(PyObject * self, PyObject * args)
{
    PyArrayObject *q1, *q2, *qsym;
    double *q1Ptr, *q2Ptr, *qsymPtr;
    int dq1, dq2, dqsym;
    int nq1, nq2, nqsym, nsym;
    double dist = 0.0;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"OOO", &q1,&q2,&qsym)) return(NULL);
    if ( q1 == NULL || q2 == NULL || qsym == NULL ) return(NULL);

    /* Verify shape of input arrays */
    dq1   = PyArray_NDIM(q1);
    dq2   = PyArray_NDIM(q2);
    dqsym = PyArray_NDIM(qsym);
    assert( dq1 == 1 && dq2 == 1 && dqsym == 2 );

    /* Verify dimensions of input arrays */
    nq1   = PyArray_DIMS(q1)[0];
    nq2   = PyArray_DIMS(q2)[0];
    nqsym = PyArray_DIMS(qsym)[0];
    nsym  = PyArray_DIMS(qsym)[1];
    assert( nq1 == 4 && nq2 == 4 && nqsym == 4 );

    /* Grab pointers to the various data arrays */
    q1Ptr   = (double*)PyArray_DATA(q1);
    q2Ptr   = (double*)PyArray_DATA(q2);
    qsymPtr = (double*)PyArray_DATA(qsym);

    /* Call the actual function */
    dist = quat_distance(nsym,q1Ptr,q2Ptr,qsymPtr);
    if (dist < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory");
        return NULL;
    }
    
    return(PyFloat_FromDouble(dist));
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
