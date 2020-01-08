
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
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_validateAngleRanges =
    "c module implementation of validate_angle_ranges.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_validateAngleRanges(PyObject * self, PyObject * args)
{
  PyArrayObject *angList, *angMin, *angMax, *reflInRange;
  PyObject *ccw;
  int ccwVal = 1; /* ccwVal set to True by default */
  int da, dmin, dmax;
  npy_intp na, nmin, nmax;
  double *aPtr, *minPtr, *maxPtr;
  bool *rPtr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OOOO", &angList,&angMin,&angMax,&ccw)) return(NULL);
  if ( angList == NULL || angMin == NULL || angMax == NULL ) return(NULL);

  /* Verify shape of input arrays */
  da   = PyArray_NDIM(angList);
  dmin = PyArray_NDIM(angMin);
  dmax = PyArray_NDIM(angMax);
  assert( da == 1 && dmin == 1 && dmax ==1 );

  /* Verify dimensions of input arrays */
  na   = PyArray_DIMS(angList)[0];
  nmin = PyArray_DIMS(angMin)[0];
  nmax = PyArray_DIMS(angMax)[0];
  assert( nmin == nmax );

  /* Check the value of ccw */
  if ( ccw == Py_True )
    ccwVal = 1;
  else
    ccwVal = 0;

  /* Allocate the result matrix with appropriate dimensions and type */
  reflInRange = (PyArrayObject*)PyArray_EMPTY(1,PyArray_DIMS(angList),NPY_BOOL,false);
  assert( reflInRange != NULL );

  /* Grab pointers to the various data arrays */
  aPtr   = (double*)PyArray_DATA(angList);
  minPtr = (double*)PyArray_DATA(angMin);
  maxPtr = (double*)PyArray_DATA(angMax);
  rPtr   = (bool*)PyArray_DATA(reflInRange);

  /* Call the actual function */
  validate_angle_ranges(na,aPtr,nmin,minPtr,maxPtr,rPtr,ccwVal);

  return((PyObject*)reflInRange);
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */

