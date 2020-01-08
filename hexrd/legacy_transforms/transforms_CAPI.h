#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <arrayobject.h>

/******************************************************************************/
/** The functions declared in this header make use of Python's C API to      **/
/** parse input arguments and verify their types (and sizes when             **/
/** appropriate).  They also allocte new Python objects for use as return    **/
/** values.                                                                  **/
/**                                                                          **/
/** In short, these functions perform all of the necessary C API calls.      **/
/**                                                                          **/
/** The actual computations are handled by functions declared in the file    **/
/** "transforms_CFUNC.h".                                                    **/
/**                                                                          **/
/** This separation of Python C API calls and C implementations allows the C **/
/** functions to call eachother without the unnecessary overhead of passing  **/
/** arguments and return values as Python objects.                           **/
/**                                                                          **/
/******************************************************************************/

/******************************************************************************/
/* Funtions */

static PyObject * anglesToGVec(PyObject * self, PyObject * args);

static PyObject * anglesToDVec(PyObject * self, PyObject * args);

static PyObject * makeGVector(PyObject * self, PyObject * args);

static PyObject * gvecToDetectorXY(PyObject * self, PyObject * args);

static PyObject * gvecToDetectorXYArray(PyObject * self, PyObject * args);

static PyObject * detectorXYToGvec(PyObject * self, PyObject * args);

static PyObject * detectorXYToGvecArray(PyObject * self, PyObject * args);

static PyObject * oscillAnglesOfHKLs(PyObject * self, PyObject * args);

/******************************************************************************/
/* Utility Funtions */

static PyObject * arccosSafe(PyObject * self, PyObject * args);

static PyObject * angularDifference(PyObject * self, PyObject * args);

static PyObject * mapAngle(PyObject * self, PyObject * args);

static PyObject * columnNorm(PyObject * self, PyObject * args);

static PyObject * rowNorm(PyObject * self, PyObject * args);

static PyObject * unitRowVector(PyObject * self, PyObject * args);

static PyObject * unitRowVectors(PyObject * self, PyObject * args);

static PyObject * makeDetectorRotMat(PyObject * self, PyObject * args);

static PyObject * makeOscillRotMat(PyObject * self, PyObject * args);

static PyObject * makeOscillRotMatArray(PyObject * self, PyObject * args);

static PyObject * makeRotMatOfExpMap(PyObject * self, PyObject * args);

static PyObject * makeRotMatOfQuat(PyObject * self, PyObject * args);

static PyObject * makeBinaryRotMat(PyObject * self, PyObject * args);

static PyObject * makeEtaFrameRotMat(PyObject * self, PyObject * args);

static PyObject * validateAngleRanges(PyObject * self, PyObject * args);

static PyObject * rotate_vecs_about_axis(PyObject * self, PyObject * args);

static PyObject * quat_distance(PyObject * self, PyObject * args);

static PyObject * homochoricOfQuat(PyObject * self, PyObject * args);
