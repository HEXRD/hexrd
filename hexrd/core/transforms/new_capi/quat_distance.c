
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
/* TODO: No printf should go here. Also, it should be possible to avoid malloc
 */
#  include <stdio.h>
#  include <sdlib.h>

#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


static double
quat_distance(size_t nsym, double * q1, double * q2, double * qsym)
{
    size_t i;
    double q0_max = 0.0, dist = 0.0;

    /* For each symmetric equivalent q2 compute its inner product with inv(q1) */
    for (i=0; i<nsym; i++) {
        double *qs = qsym + 4*i;
        double q2s[4], abs_q0;
        
        q2s[0] = q2[0]*qs[0] - q2[1]*qs[1] - q2[2]*qs[2] - q2[3]*qs[3];
        q2s[1] = q2[1]*qs[0] + q2[0]*qs[1] - q2[3]*qs[2] + q2[2]*qs[3];
        q2s[2] = q2[2]*qs[0] + q2[3]*qs[1] + q2[0]*qs[2] - q2[1]*qs[3];
        q2s[3] = q2[3]*qs[0] - q2[2]*qs[1] + q2[1]*qs[2] + q2[0]*qs[3];
        
        
        abs_q0 = fabs(q1[0]*q2s[0] + q1[1]*q2s[1] + q1[2]*q2s[2] + q1[3]*q2s[3]);
        if (abs_q0 > q0_max)
        {
            q0_max = abs_q0;
        }
    }

    if ( q0_max <= 1.0 )
        dist = 2.0*acos(q0_max);
    else if ( q0_max - 1. < 1e-12 )
        /* in case of quats loaded from single precision file */
        dist = 0.;
    else
        dist = NAN;


    return dist;
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#    include "ndargs_helper.h"
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_quat_distance =
    "c module implementation of quat_distance.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_quat_distance(PyObject * self, PyObject * args)
{
    nah_array q1 = { NULL, "q1", NAH_TYPE_DP_FP, { 4 }};
    nah_array q2 = { NULL, "q2", NAH_TYPE_DP_FP, { 4 }};
    nah_array qsym = { NULL, "qsym", NAH_TYPE_DP_FP, { 4, NAH_DIM_ANY }};
    double dist = 0.0;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"O&O&O&",
                          nah_array_converter, &q1,
                          nah_array_converter, &q2,
                          nah_array_converter, &qsym))
        return NULL;

    /* Call the actual function */
    dist = quat_distance(PyArray_DIM(qsym.pyarray, 0),
                         PyArray_DATA(q1.pyarray),
                         PyArray_DATA(q2.pyarray),
                         PyArray_DATA(qsym.pyarray));
    
    return PyFloat_FromDouble(dist);
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
