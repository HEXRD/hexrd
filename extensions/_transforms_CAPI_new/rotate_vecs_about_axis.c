
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION void
rotate_vecs_about_axis(size_t na, double *angles,
                       size_t nax, double *axes,
                       size_t nv, double *vecs,
                       double *rVecs)
{
    size_t i, j, sa, sax;
    double c, s, nrm, proj, aCrossV[3];

    if ( na == 1 ) sa = 0;
    else sa = 1;
    if ( nax == 1 ) sax = 0;
    else sax = 3;

    for (i=0; i<nv; i++) {

        /* Rotate using the Rodrigues' Rotation Formula */
        c = cos(angles[sa*i]);
        s = sin(angles[sa*i]);

        /* Compute projection of vec along axis */
        proj = 0.0;
        for (j=0; j<3; j++)
            proj += axes[sax*i+j]*vecs[3*i+j];

        /* Compute norm of axis */
        if ( nax > 1 || i == 0 ) {
            nrm = 0.0;
            for (j=0; j<3; j++)
                nrm += axes[sax*i+j]*axes[sax*i+j];
            nrm = sqrt(nrm);
        }

        /* Compute projection of vec along axis */
        proj = 0.0;
        for (j=0; j<3; j++)
            proj += axes[sax*i+j]*vecs[3*i+j];

        /* Compute the cross product of the axis with vec */
        for (j=0; j<3; j++)
            aCrossV[j] = axes[sax*i+(j+1)%3]*vecs[3*i+(j+2)%3]-axes[sax*i+(j+2)%3]*vecs[3*i+(j+1)%3];

        /* Combine the three terms to compute the rotated vector */
        for (j=0; j<3; j++) {
            rVecs[3*i+j] = c*vecs[3*i+j]+(s/nrm)*aCrossV[j]+(1.0-c)*proj*axes[sax*i+j]/(nrm*nrm);
        }
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_rotate_vecs_about_axis =
    "c module implementation of rotate_vecs_about_axis.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_rotate_vecs_about_axis(PyObject * self, PyObject * args)
{
    PyArrayObject *angles, *axes, *vecs;
    PyArrayObject *rVecs;
    int da, dax, dv;
    npy_intp na, nax0, nax1, nv0, nv1;
    double *aPtr, *axesPtr, *vecsPtr;
    double *rPtr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"OOO", &angles,&axes,&vecs)) return(NULL);
    if ( angles == NULL || axes == NULL || vecs == NULL ) return(NULL);

    /* Verify shape of input arrays */
    da  = PyArray_NDIM(angles);
    dax = PyArray_NDIM(axes);
    dv  = PyArray_NDIM(vecs);
    assert( da == 1 && dax == 2 && dv == 2 );

    /* Verify dimensions of input arrays */
    na   = PyArray_DIMS(angles)[0];
    nax0 = PyArray_DIMS(axes)[0];
    nax1 = PyArray_DIMS(axes)[1];
    nv0  = PyArray_DIMS(vecs)[0];
    nv1  = PyArray_DIMS(vecs)[1];
    assert( na == 1   || na == nv0 );
    assert( nax0 == 1 || nax0 == nv0 );
    assert( nax1 == 3 && nv1 == 3 );

    /* Allocate the result vectors with appropriate dimensions and type */
    rVecs = (PyArrayObject*)PyArray_EMPTY(2,PyArray_DIMS(vecs),NPY_DOUBLE,0.0);
    assert( rVecs != NULL );

    /* Grab pointers to the various data arrays */
    aPtr    = (double*)PyArray_DATA(angles);
    axesPtr = (double*)PyArray_DATA(axes);
    vecsPtr = (double*)PyArray_DATA(vecs);
    rPtr    = (double*)PyArray_DATA(rVecs);

    /* Call the actual function */
    rotate_vecs_about_axis(na,aPtr,nax0,axesPtr,nv0,vecsPtr,rPtr);

    return((PyObject*)rVecs);
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
