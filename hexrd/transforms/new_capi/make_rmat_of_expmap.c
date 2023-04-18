
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#  include "ndargs_helper.h"
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
        m33_set_identity(rPtr);
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
    nah_array exp_map = { NULL, "exp_map", NAH_TYPE_DP_FP, { 3 }};
    PyArrayObject *result = NULL;
    npy_intp dims[2] = { 3, 3 };

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "O&",
                          nah_array_converter, &exp_map))
        return NULL;

    /* Allocate the result matrix with appropriate dimensions and type */
    result = (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (NULL == result)
        goto fail_alloc;

    /* Call the actual function */
    make_rmat_of_expmap(PyArray_DATA(exp_map.pyarray),
                        PyArray_DATA(result));

    return (PyObject *)result;
                        
 fail_alloc:
    Py_XDECREF(result);

    return PyErr_NoMemory();
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
