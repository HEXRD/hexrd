
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif

XRD_CFUNCTION void
validate_angle_ranges(size_t na, double *aPtr, size_t nr, double *minPtr,
                      double *maxPtr, bool *rPtr, int ccw)
{
    size_t i, j;
    double thetaMax, theta;
    double *startPtr, *stopPtr;

    if ( ccw ) {
        startPtr = minPtr;
        stopPtr  = maxPtr;
    } else {
        startPtr = maxPtr;
        stopPtr  = minPtr;
    }

    /* Each angle should only be examined once. Any more is a waste of time.  */
    for (i=0; i<na; i++) {

        /* Ensure there's no match to begin with */
        rPtr[i] = false;

        for (j=0; j<nr; j++) {

            /* Since the angle values themselves are unimportant we will
               redefine them so that the start of the range is zero.  The
               end of the range will then be between zero and two pi.  It
               will then be quite easy to determine if the angle of interest
               is in the range or not. */

            thetaMax = stopPtr[j] - startPtr[j];
            theta    = aPtr[i] - startPtr[j];

            while ( thetaMax < 0.0 )
                thetaMax += 2.0*M_PI;
            while ( thetaMax > 2.0*M_PI )
                thetaMax -= 2.0*M_PI;

            /* Check for an empty range */
            if ( fabs(thetaMax) < sqrt_epsf ) {
                rPtr[i] = true;

                /* No need to check other ranges */
                break;
            }

            /* Check for a range which spans a full circle */
            if ( fabs(thetaMax-2.0*M_PI) < sqrt_epsf ) {

                /* Double check the initial range */
                if ( (ccw && maxPtr[j] > minPtr[j]) || ((!ccw) && maxPtr[j] < minPtr[j]) ) {
                    rPtr[i] = true;

                    /* No need to check other ranges */
                    break;
                }
            }

            while ( theta < 0.0 )
                theta += 2.0*M_PI;
            while ( theta > 2.0*M_PI )
                theta -= 2.0*M_PI;

            if ( theta >= -sqrt_epsf && theta <= thetaMax+sqrt_epsf ) {
                rPtr[i] = true;

                /* No need to check other ranges */
                break;
            }
        }
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#    include "ndargs_helper.h"
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_validateAngleRanges =
    "c module implementation of validate_angle_ranges.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_validateAngleRanges(PyObject * self, PyObject * args)
{
    nah_array ang_list = { NULL, "ang_list", NAH_TYPE_DP_FP, { NAH_DIM_ANY }};
    nah_array start_ang = { NULL, "start_ang", NAH_TYPE_DP_FP, { NAH_DIM_ANY }};
    nah_array stop_ang = { NULL, "stop_ang", NAH_TYPE_DP_FP, { NAH_DIM_ANY }};
    int ccw = 1;
    PyArrayObject *result = NULL;;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"O&O&O&|p",
                          nah_array_converter, &ang_list,
                          nah_array_converter, &start_ang,
                          nah_array_converter, &stop_ang,
                          &ccw))
        return NULL;

    /* Verify that start_ang and stop ang have the same length */
    if (PyArray_DIM(start_ang.pyarray, 0) != PyArray_DIM(stop_ang.pyarray, 0))
        goto fail_start_stop_mismatch;

    /* Allocate the result matrix with appropriate dimensions and type */
    result = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(ang_list.pyarray),
                                           PyArray_SHAPE(ang_list.pyarray),
                                           NPY_BOOL, false);
    if (NULL == result)
        goto fail_alloc;


    /* Call the actual function */
    validate_angle_ranges(PyArray_DIM(ang_list.pyarray, 0),
                          (double *)PyArray_DATA(ang_list.pyarray),
                          PyArray_DIM(start_ang.pyarray,0),
                          (double *)PyArray_DATA(start_ang.pyarray),
                          (double *)PyArray_DATA(stop_ang.pyarray),
                          (bool*)PyArray_DATA(result),
                          ccw);

    return (PyObject*)result;

 fail_start_stop_mismatch:
  PyErr_Format(PyExc_RuntimeError, "'%s' and '%s' must have the same length.",
               start_ang.name, stop_ang.name);
  goto release_objs;
  
 fail_alloc:
  PyErr_NoMemory();
  goto release_objs;
  
 release_objs:
  Py_XDECREF(result);
  return NULL;
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */

