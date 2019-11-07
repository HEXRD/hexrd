
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION int
unit_row_vector(size_t n, double * cIn, double * cOut)
{
    size_t j;
    double nrm;

    nrm = 0.0;
    for (j = 0; j < n; j++) {
        nrm += cIn[j]*cIn[j];
    }

    nrm = sqrt(nrm);
    if ( nrm > epsf ) {
        for (j=0; j<n; j++) {
            cOut[j] = cIn[j]/nrm;
        }
        return 0;
    } else {
        for (j=0; j<n; j++) {
            cOut[j] = cIn[j];
        }
        return 1;
    }
}

XRD_CFUNCTION void
unit_row_vectors(size_t m, size_t n, double *cIn, double *cOut)
{
    size_t i, j;
    double nrm;

    for (i=0; i<m; i++) {
        nrm = 0.0;
        for (j=0; j<n; j++) {
            nrm += cIn[n*i+j]*cIn[n*i+j];
        }
        nrm = sqrt(nrm);
        if ( nrm > epsf ) {
            for (j=0; j<n; j++) {
                cOut[n*i+j] = cIn[n*i+j]/nrm;
            }
        } else {
            for (j=0; j<n; j++) {
                cOut[n*i+j] = cIn[n*i+j];
            }
        }
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_unitRowVector =
    "c module implementation of unit_row_vector (one row).\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER const char *docstring_unitRowVectors =
    "c module implementation of unit_row_vector (multiple rows).\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_unitRowVector(PyObject * self, PyObject * args)
{
    PyArrayObject *aop_out;
    named_array_1d in = { "vecIn", NULL };
    npy_intp out_dims[1];
    
    if ( !PyArg_ParseTuple(args,"O&",
                           array_1d_converter, &in) )
        return(NULL);

    out_dims[0] = (npy_intp)in.count;
    aop_out = (PyArrayObject*)PyArray_EMPTY(1, out_dims, NPY_DOUBLE, 0);

    unit_row_vector(in.count, in.data, (double *)PyArray_DATA(aop_out));

    return (PyObject*)aop_out;
}


XRD_PYTHON_WRAPPER PyObject *
python_unitRowVectors(PyObject *self, PyObject *args)
{
    PyArrayObject *aop_out;
    named_array_2d in = { "vecIn", NULL };
    npy_intp out_dims[2];

    if ( !PyArg_ParseTuple(args, "O&",
                           array_2d_converter, &in) )
        return(NULL);

    out_dims[0] = (npy_intp)in.outer_count;
    out_dims[1] = (npy_intp)in.inner_count;
    aop_out = (PyArrayObject*)PyArray_EMPTY(2, out_dims, NPY_DOUBLE, 0);

    unit_row_vectors(in.outer_count, in.inner_count, in.data,
                     (double *)PyArray_DATA(aop_out));

    return (PyObject*)aop_out;
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
