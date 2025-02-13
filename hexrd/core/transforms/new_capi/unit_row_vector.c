
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#  include "ndargs_helper.h"
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
    PyArrayObject *aop_out = NULL;
    nah_array in = { NULL, "vecIn", NAH_TYPE_DP_FP, { NAH_DIM_ANY, NAH_DIM_OPT}}; 
    
    if ( !PyArg_ParseTuple(args,"O&",
                           nah_array_converter, &in) )
        return(NULL);

    /*
      At this point, nah_array_converter should just let pass one or two
      dimensional arrays with double dtype that are properly aligned
    */
    aop_out = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(in.pyarray),
                                            PyArray_SHAPE(in.pyarray),
                                            NPY_DOUBLE, 0);
    if (aop_out)
    {
        if (PyArray_NDIM(in.pyarray) == 1)
            unit_row_vector(PyArray_DIM(in.pyarray, 0),
                            (double*)PyArray_DATA(in.pyarray),
                            (double*)PyArray_DATA(aop_out));
        else
            unit_row_vectors(PyArray_DIM(in.pyarray, 0),
                             PyArray_DIM(in.pyarray,1),
                             (double *)PyArray_DATA(in.pyarray),
                             (double *)PyArray_DATA(aop_out));

            
                            
    }

    return (PyObject*)aop_out;
}


XRD_PYTHON_WRAPPER PyObject *
python_unitRowVectors(PyObject *self, PyObject *args)
{
    PyArrayObject *aop_out = NULL;
    nah_array in = { NULL, "vecIn", NAH_TYPE_DP_FP, { NAH_DIM_ANY, NAH_DIM_ANY }}; 
    
    if ( !PyArg_ParseTuple(args,"O&",
                           nah_array_converter, &in) )
        return(NULL);

    /*
      At this point, nah_array_converter should just let pass two
      dimensional arrays with double dtype that are properly aligned.

      Note that this functions should not be needed as this case is also
      hndled by unitRowVector right now.
    */
    aop_out = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(in.pyarray),
                                            PyArray_SHAPE(in.pyarray),
                                            NPY_DOUBLE, 0);
    if (aop_out)
    {
        unit_row_vectors(PyArray_DIM(in.pyarray, 0),
                         PyArray_DIM(in.pyarray,1),
                         (double *)PyArray_DATA(in.pyarray),
                         (double *)PyArray_DATA(aop_out));
    }

    return (PyObject*)aop_out;
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
