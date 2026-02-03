/*
 * gcc -c -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy transforms_CAPI.c
 *
 * gcc -bundle -flat_namespace -undefined suppress -o _transforms_CAPI.so transforms_CAPI.o
 */

#include "transforms_CAPI.h"
#include "transforms_CFUNC.h"

static PyMethodDef _transform_methods[] = {
  {"anglesToGVec",anglesToGVec,METH_VARARGS,"take angle tuples to G-vectors"},
  {"anglesToDVec",anglesToDVec,METH_VARARGS,"take angle tuples to unit diffraction vectors"},
  {"makeGVector",makeGVector,METH_VARARGS,"Make G-vectors from hkls and B-matrix"},
  {"gvecToDetectorXY",gvecToDetectorXY,METH_VARARGS,""},
  {"gvecToDetectorXYArray",gvecToDetectorXYArray,METH_VARARGS,""},
  {"detectorXYToGvec",detectorXYToGvec,METH_VARARGS,"take cartesian coordinates to G-vectors"},
  {"detectorXYToGvecArray",detectorXYToGvecArray,METH_VARARGS,"take cartesian coordinates to G-vectors"},
  {"oscillAnglesOfHKLs",oscillAnglesOfHKLs,METH_VARARGS,"solve angle specs for G-vectors"},
  {"unitRowVector",unitRowVector,METH_VARARGS,"Normalize a single row vector"},
  {"unitRowVectors",unitRowVectors,METH_VARARGS,"Normalize a collection of row vectors"},
  {"makeDetectorRotMat",makeDetectorRotMat,METH_VARARGS,""},
  {"makeOscillRotMat",makeOscillRotMat,METH_VARARGS,""},
  {"makeOscillRotMatArray",makeOscillRotMatArray,METH_VARARGS,""},
  {"makeRotMatOfExpMap",makeRotMatOfExpMap,METH_VARARGS,""},
  {"makeRotMatOfQuat",makeRotMatOfQuat,METH_VARARGS,""},
  {"makeBinaryRotMat",makeBinaryRotMat,METH_VARARGS,""},
  {"makeEtaFrameRotMat",makeEtaFrameRotMat,METH_VARARGS,"Make eta basis COB matrix"},
  {"validateAngleRanges",validateAngleRanges,METH_VARARGS,""},
  {"rotate_vecs_about_axis",rotate_vecs_about_axis,METH_VARARGS,"Rotate vectors about an axis"},
  {"quat_distance",quat_distance,METH_VARARGS,"Compute distance between two unit quaternions"},
  {"homochoricOfQuat",homochoricOfQuat,METH_VARARGS,"Compute homochoric parameterization of list of unit quaternions"},
  {NULL,NULL}
};

static struct PyModuleDef transforms_capi_pymodule =
{
    PyModuleDef_HEAD_INIT,
    "_transforms_CAPI", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    _transform_methods
};

PyMODINIT_FUNC PyInit__transforms_CAPI(void)
{
  import_array();
  return PyModule_Create(&transforms_capi_pymodule);
}

/******************************************************************************/
/* Funtions */

static PyObject * anglesToGVec(PyObject * self, PyObject * args)
{
  PyArrayObject *angs, *bHat_l, *eHat_l, *rMat_c;
  PyArrayObject *gVec_c;
  double chi;
  npy_intp nvecs, rdims[2];

  int nangs, nbhat, nehat, nrmat;
  int da1, db1, de1, dr1, dr2;

  double *angs_ptr, *bHat_l_ptr, *eHat_l_ptr, *rMat_c_ptr;
  double *gVec_c_ptr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OOOdO",
			 &angs,
			 &bHat_l, &eHat_l,
			 &chi, &rMat_c)) return(NULL);
  if ( angs == NULL ) return(NULL);

  /* Verify shape of input arrays */
  nangs = PyArray_NDIM(angs);
  nbhat = PyArray_NDIM(bHat_l);
  nehat = PyArray_NDIM(eHat_l);
  nrmat = PyArray_NDIM(rMat_c);

  assert( nangs==2 && nbhat==1 && nehat==1 && nrmat==2 );

  /* Verify dimensions of input arrays */
  nvecs = PyArray_DIMS(angs)[0]; //rows
  da1   = PyArray_DIMS(angs)[1]; //cols

  db1   = PyArray_DIMS(bHat_l)[0];
  de1   = PyArray_DIMS(eHat_l)[0];
  dr1   = PyArray_DIMS(rMat_c)[0];
  dr2   = PyArray_DIMS(rMat_c)[1];

  assert( da1 == 3 );
  assert( db1 == 3 && de1 == 3);
  assert( dr1 == 3 && dr2 == 3);

  /* Allocate C-style array for return data */
  rdims[0] = nvecs; rdims[1] = 3;
  gVec_c = (PyArrayObject*)PyArray_EMPTY(2,rdims,NPY_DOUBLE,0);

  /* Grab pointers to the various data arrays */
  angs_ptr   = (double*)PyArray_DATA(angs);
  bHat_l_ptr = (double*)PyArray_DATA(bHat_l);
  eHat_l_ptr = (double*)PyArray_DATA(eHat_l);
  rMat_c_ptr = (double*)PyArray_DATA(rMat_c);
  gVec_c_ptr = (double*)PyArray_DATA(gVec_c);

  /* Call the actual function */
  anglesToGvec_cfunc(nvecs, angs_ptr,
		     bHat_l_ptr, eHat_l_ptr,
		     chi, rMat_c_ptr,
		     gVec_c_ptr);

  /* Build and return the nested data structure */
  return((PyObject*)gVec_c);
}

static PyObject * anglesToDVec(PyObject * self, PyObject * args)
{
  PyArrayObject *angs, *bHat_l, *eHat_l, *rMat_c;
  PyArrayObject *dVec_c;
  double chi;
  npy_intp nvecs, rdims[2];

  int nangs, nbhat, nehat, nrmat;
  int da1, db1, de1, dr1, dr2;

  double *angs_ptr, *bHat_l_ptr, *eHat_l_ptr, *rMat_c_ptr;
  double *dVec_c_ptr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OOOdO",
			 &angs,
			 &bHat_l, &eHat_l,
			 &chi, &rMat_c)) return(NULL);
  if ( angs == NULL ) return(NULL);

  /* Verify shape of input arrays */
  nangs = PyArray_NDIM(angs);
  nbhat = PyArray_NDIM(bHat_l);
  nehat = PyArray_NDIM(eHat_l);
  nrmat = PyArray_NDIM(rMat_c);

  assert( nangs==2 && nbhat==1 && nehat==1 && nrmat==2 );

  /* Verify dimensions of input arrays */
  nvecs = PyArray_DIMS(angs)[0]; //rows
  da1   = PyArray_DIMS(angs)[1]; //cols

  db1   = PyArray_DIMS(bHat_l)[0];
  de1   = PyArray_DIMS(eHat_l)[0];
  dr1   = PyArray_DIMS(rMat_c)[0];
  dr2   = PyArray_DIMS(rMat_c)[1];

  assert( da1 == 3 );
  assert( db1 == 3 && de1 == 3);
  assert( dr1 == 3 && dr2 == 3);

  /* Allocate C-style array for return data */
  rdims[0] = nvecs; rdims[1] = 3;
  dVec_c = (PyArrayObject*)PyArray_EMPTY(2,rdims,NPY_DOUBLE,0);

  /* Grab pointers to the various data arrays */
  angs_ptr   = (double*)PyArray_DATA(angs);
  bHat_l_ptr = (double*)PyArray_DATA(bHat_l);
  eHat_l_ptr = (double*)PyArray_DATA(eHat_l);
  rMat_c_ptr = (double*)PyArray_DATA(rMat_c);
  dVec_c_ptr = (double*)PyArray_DATA(dVec_c);

  /* Call the actual function */
  anglesToDvec_cfunc(nvecs, angs_ptr,
		     bHat_l_ptr, eHat_l_ptr,
		     chi, rMat_c_ptr,
		     dVec_c_ptr);

  /* Build and return the nested data structure */
  return((PyObject*)dVec_c);
}

static PyObject * makeGVector(PyObject * self, PyObject * args)
{
  return(NULL);
}

/*
    Takes a list of unit reciprocal lattice vectors in crystal frame to the
    specified detector-relative frame, subject to the conditions:

    1) the reciprocal lattice vector must be able to satisfy a bragg condition
    2) the associated diffracted beam must intersect the detector plane

    Required Arguments:
    gVec_c -- (n, 3) ndarray of n reciprocal lattice vectors in the CRYSTAL FRAME
    rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components to LAB FRAME
    rMat_s -- (3, 3) ndarray, the COB taking SAMPLE FRAME components to LAB FRAME
    rMat_c -- (3, 3) ndarray, the COB taking CRYSTAL FRAME components to SAMPLE FRAME
    tVec_d -- (3, 1) ndarray, the translation vector connecting LAB to DETECTOR
    tVec_s -- (3, 1) ndarray, the translation vector connecting LAB to SAMPLE
    tVec_c -- (3, 1) ndarray, the translation vector connecting SAMPLE to CRYSTAL

    Outputs:
    (m, 2) ndarray containing the intersections of m <= n diffracted beams
    associated with gVecs
*/
static PyObject * gvecToDetectorXY(PyObject * self, PyObject * args)
{
  PyArrayObject *gVec_c,
                *rMat_d, *rMat_s, *rMat_c,
                *tVec_d, *tVec_s, *tVec_c,
                *beamVec;
  PyArrayObject *result;

  int dgc, drd, drs, drc, dtd, dts, dtc, dbv;
  npy_intp npts, dims[2];

  double *gVec_c_Ptr,
         *rMat_d_Ptr, *rMat_s_Ptr, *rMat_c_Ptr,
         *tVec_d_Ptr, *tVec_s_Ptr, *tVec_c_Ptr,
         *beamVec_Ptr;
  double *result_Ptr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OOOOOOOO",
			 &gVec_c,
			 &rMat_d, &rMat_s, &rMat_c,
			 &tVec_d, &tVec_s, &tVec_c,
			 &beamVec)) return(NULL);
  if ( gVec_c  == NULL ||
       rMat_d  == NULL || rMat_s == NULL || rMat_c == NULL ||
       tVec_d  == NULL || tVec_s == NULL || tVec_c == NULL ||
       beamVec == NULL ) return(NULL);

  /* Verify shape of input arrays */
  dgc = PyArray_NDIM(gVec_c);
  drd = PyArray_NDIM(rMat_d);
  drs = PyArray_NDIM(rMat_s);
  drc = PyArray_NDIM(rMat_c);
  dtd = PyArray_NDIM(tVec_d);
  dts = PyArray_NDIM(tVec_s);
  dtc = PyArray_NDIM(tVec_c);
  dbv = PyArray_NDIM(beamVec);
  assert( dgc == 2 );
  assert( drd == 2 && drs == 2 && drc == 2 );
  assert( dtd == 1 && dts == 1 && dtc == 1 );
  assert( dbv == 1 );

  /* Verify dimensions of input arrays */
  npts = PyArray_DIMS(gVec_c)[0];

  assert( PyArray_DIMS(gVec_c)[1]  == 3 );
  assert( PyArray_DIMS(rMat_d)[0]  == 3 && PyArray_DIMS(rMat_d)[1] == 3 );
  assert( PyArray_DIMS(rMat_s)[0]  == 3 && PyArray_DIMS(rMat_s)[1] == 3 );
  assert( PyArray_DIMS(rMat_c)[0]  == 3 && PyArray_DIMS(rMat_c)[1] == 3 );
  assert( PyArray_DIMS(tVec_d)[0]  == 3 );
  assert( PyArray_DIMS(tVec_s)[0]  == 3 );
  assert( PyArray_DIMS(tVec_c)[0]  == 3 );
  assert( PyArray_DIMS(beamVec)[0] == 3 );

  /* Allocate C-style array for return data */
  dims[0] = npts; dims[1] = 2;
  result = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  /* Grab data pointers into various arrays */
  gVec_c_Ptr  = (double*)PyArray_DATA(gVec_c);

  rMat_d_Ptr  = (double*)PyArray_DATA(rMat_d);
  rMat_s_Ptr  = (double*)PyArray_DATA(rMat_s);
  rMat_c_Ptr  = (double*)PyArray_DATA(rMat_c);

  tVec_d_Ptr  = (double*)PyArray_DATA(tVec_d);
  tVec_s_Ptr  = (double*)PyArray_DATA(tVec_s);
  tVec_c_Ptr  = (double*)PyArray_DATA(tVec_c);

  beamVec_Ptr = (double*)PyArray_DATA(beamVec);

  result_Ptr     = (double*)PyArray_DATA(result);

  /* Call the computational routine */
  gvecToDetectorXY_cfunc(npts, gVec_c_Ptr,
			 rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
			 tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
			 beamVec_Ptr,
			 result_Ptr);

  /* Build and return the nested data structure */
  return((PyObject*)result);
}

/*
    Takes a list of unit reciprocal lattice vectors in crystal frame to the
    specified detector-relative frame, subject to the conditions:

    1) the reciprocal lattice vector must be able to satisfy a bragg condition
    2) the associated diffracted beam must intersect the detector plane

    Required Arguments:
    gVec_c -- (n, 3) ndarray of n reciprocal lattice vectors in the CRYSTAL FRAME
    rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components to LAB FRAME
    rMat_s -- (n, 3, 3) ndarray, the COB taking SAMPLE FRAME components to LAB FRAME
    rMat_c -- (3, 3) ndarray, the COB taking CRYSTAL FRAME components to SAMPLE FRAME
    tVec_d -- (3, 1) ndarray, the translation vector connecting LAB to DETECTOR
    tVec_s -- (3, 1) ndarray, the translation vector connecting LAB to SAMPLE
    tVec_c -- (3, 1) ndarray, the translation vector connecting SAMPLE to CRYSTAL

    Outputs:
    (m, 2) ndarray containing the intersections of m <= n diffracted beams
    associated with gVecs
*/
static PyObject * gvecToDetectorXYArray(PyObject * self, PyObject * args)
{
  PyArrayObject *gVec_c,
                *rMat_d, *rMat_s, *rMat_c,
                *tVec_d, *tVec_s, *tVec_c,
                *beamVec;
  PyArrayObject *result;

  int dgc, drd, drs, drc, dtd, dts, dtc, dbv;
  npy_intp npts, dims[2];

  double *gVec_c_Ptr,
         *rMat_d_Ptr, *rMat_s_Ptr, *rMat_c_Ptr,
         *tVec_d_Ptr, *tVec_s_Ptr, *tVec_c_Ptr,
         *beamVec_Ptr;
  double *result_Ptr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OOOOOOOO",
			 &gVec_c,
			 &rMat_d, &rMat_s, &rMat_c,
			 &tVec_d, &tVec_s, &tVec_c,
			 &beamVec)) return(NULL);
  if ( gVec_c  == NULL ||
       rMat_d  == NULL || rMat_s == NULL || rMat_c == NULL ||
       tVec_d  == NULL || tVec_s == NULL || tVec_c == NULL ||
       beamVec == NULL ) return(NULL);

  /* Verify shape of input arrays */
  dgc = PyArray_NDIM(gVec_c);
  drd = PyArray_NDIM(rMat_d);
  drs = PyArray_NDIM(rMat_s);
  drc = PyArray_NDIM(rMat_c);
  dtd = PyArray_NDIM(tVec_d);
  dts = PyArray_NDIM(tVec_s);
  dtc = PyArray_NDIM(tVec_c);
  dbv = PyArray_NDIM(beamVec);
  assert( dgc == 2 );
  assert( drd == 2 && drs == 3 && drc == 2 );
  assert( dtd == 1 && dts == 1 && dtc == 1 );
  assert( dbv == 1 );

  /* Verify dimensions of input arrays */
  npts = PyArray_DIMS(gVec_c)[0];

  if (npts != PyArray_DIM(rMat_s, 0)) {
    PyErr_Format(PyExc_ValueError, "gVec_c and rMat_s length mismatch %d vs %d",
                 (int)PyArray_DIM(gVec_c, 0), (int)PyArray_DIM(rMat_s, 0));
    return NULL;
  }
  assert( PyArray_DIMS(gVec_c)[1]  == 3 );
  assert( PyArray_DIMS(rMat_d)[0]  == 3 && PyArray_DIMS(rMat_d)[1] == 3 );
  assert( PyArray_DIMS(rMat_s)[1]  == 3 && PyArray_DIMS(rMat_s)[2] == 3 );
  assert( PyArray_DIMS(rMat_c)[0]  == 3 && PyArray_DIMS(rMat_c)[1] == 3 );
  assert( PyArray_DIMS(tVec_d)[0]  == 3 );
  assert( PyArray_DIMS(tVec_s)[0]  == 3 );
  assert( PyArray_DIMS(tVec_c)[0]  == 3 );
  assert( PyArray_DIMS(beamVec)[0] == 3 );

  /* Allocate C-style array for return data */
  dims[0] = npts; dims[1] = 2;
  result = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  /* Grab data pointers into various arrays */
  gVec_c_Ptr  = (double*)PyArray_DATA(gVec_c);

  rMat_d_Ptr  = (double*)PyArray_DATA(rMat_d);
  rMat_s_Ptr  = (double*)PyArray_DATA(rMat_s);
  rMat_c_Ptr  = (double*)PyArray_DATA(rMat_c);

  tVec_d_Ptr  = (double*)PyArray_DATA(tVec_d);
  tVec_s_Ptr  = (double*)PyArray_DATA(tVec_s);
  tVec_c_Ptr  = (double*)PyArray_DATA(tVec_c);

  beamVec_Ptr = (double*)PyArray_DATA(beamVec);

  result_Ptr     = (double*)PyArray_DATA(result);

  /* Call the computational routine */
  gvecToDetectorXYArray_cfunc(npts, gVec_c_Ptr,
			 rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
			 tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
			 beamVec_Ptr,
			 result_Ptr);

  /* Build and return the nested data structure */
  return((PyObject*)result);
}

/*
    Takes a list cartesian (x, y) pairs in the detector coordinates and calculates
    the associated reciprocal lattice (G) vectors and (bragg angle, azimuth) pairs
    with respect to the specified beam and azimth (eta) reference directions

    Required Arguments:
    xy_det -- (n, 2) ndarray or list-like input of n detector (x, y) points
    rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components to LAB FRAME
    rMat_s -- (3, 3) ndarray, the COB taking SAMPLE FRAME components to LAB FRAME
    tVec_d -- (3, 1) ndarray, the translation vector connecting LAB to DETECTOR
    tVec_s -- (3, 1) ndarray, the translation vector connecting LAB to SAMPLE
    tVec_c -- (3, 1) ndarray, the translation vector connecting SAMPLE to CRYSTAL

    Optional Keyword Arguments:
    beamVec -- (1, 3) mdarray containing the incident beam direction components in the LAB FRAME
    etaVec  -- (1, 3) mdarray containing the reference azimuth direction components in the LAB FRAME

    Outputs:
    (n, 2) ndarray containing the (tTh, eta) pairs associated with each (x, y)
    (n, 3) ndarray containing the associated G vector directions in the LAB FRAME
    associated with gVecs
*/
static PyObject * detectorXYToGvec(PyObject * self, PyObject * args)
{
  PyArrayObject *xy_det, *rMat_d, *rMat_s,
		        *tVec_d, *tVec_s, *tVec_c,
                *beamVec, *etaVec;
  PyArrayObject *tTh, *eta, *gVec_l;
  PyObject *inner_tuple, *outer_tuple;

  int dxy, drd, drs, dtd, dts, dtc, dbv, dev;
  npy_intp npts, dims[2];

  double *xy_Ptr, *rMat_d_Ptr, *rMat_s_Ptr,
         *tVec_d_Ptr, *tVec_s_Ptr, *tVec_c_Ptr,
         *beamVec_Ptr, *etaVec_Ptr;
  double *tTh_Ptr, *eta_Ptr, *gVec_l_Ptr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OOOOOOOO",
			 &xy_det,
			 &rMat_d, &rMat_s,
			 &tVec_d, &tVec_s, &tVec_c,
			 &beamVec, &etaVec)) return(NULL);
  if ( xy_det  == NULL || rMat_d == NULL || rMat_s == NULL ||
       tVec_d  == NULL || tVec_s == NULL || tVec_c == NULL ||
       beamVec == NULL || etaVec == NULL ) return(NULL);

  /* Verify shape of input arrays */
  dxy = PyArray_NDIM(xy_det);
  drd = PyArray_NDIM(rMat_d);
  drs = PyArray_NDIM(rMat_s);
  dtd = PyArray_NDIM(tVec_d);
  dts = PyArray_NDIM(tVec_s);
  dtc = PyArray_NDIM(tVec_c);
  dbv = PyArray_NDIM(beamVec);
  dev = PyArray_NDIM(etaVec);
  assert( dxy == 2 && drd == 2 && drs == 2 &&
	  dtd == 1 && dts == 1 && dtc == 1 &&
	  dbv == 1 && dev == 1);

  /* Verify dimensions of input arrays */
  npts = PyArray_DIMS(xy_det)[0];

  assert( PyArray_DIMS(xy_det)[1]  == 2 );
  assert( PyArray_DIMS(rMat_d)[0]  == 3 && PyArray_DIMS(rMat_d)[1] == 3 );
  assert( PyArray_DIMS(rMat_s)[0]  == 3 && PyArray_DIMS(rMat_s)[1] == 3 );
  assert( PyArray_DIMS(tVec_d)[0]  == 3 );
  assert( PyArray_DIMS(tVec_s)[0]  == 3 );
  assert( PyArray_DIMS(tVec_c)[0]  == 3 );
  assert( PyArray_DIMS(beamVec)[0] == 3 );
  assert( PyArray_DIMS(etaVec)[0]  == 3 );

  /* Allocate arrays for return values */
  dims[0] = npts; dims[1] = 3;
  gVec_l = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  tTh    = (PyArrayObject*)PyArray_EMPTY(1,&npts,NPY_DOUBLE,0);
  eta    = (PyArrayObject*)PyArray_EMPTY(1,&npts,NPY_DOUBLE,0);

  /* Grab data pointers into various arrays */
  xy_Ptr      = (double*)PyArray_DATA(xy_det);
  gVec_l_Ptr  = (double*)PyArray_DATA(gVec_l);

  tTh_Ptr     = (double*)PyArray_DATA(tTh);
  eta_Ptr     = (double*)PyArray_DATA(eta);

  rMat_d_Ptr  = (double*)PyArray_DATA(rMat_d);
  rMat_s_Ptr  = (double*)PyArray_DATA(rMat_s);

  tVec_d_Ptr  = (double*)PyArray_DATA(tVec_d);
  tVec_s_Ptr  = (double*)PyArray_DATA(tVec_s);
  tVec_c_Ptr  = (double*)PyArray_DATA(tVec_c);

  beamVec_Ptr = (double*)PyArray_DATA(beamVec);
  etaVec_Ptr  = (double*)PyArray_DATA(etaVec);

  /* Call the computational routine */
  detectorXYToGvec_cfunc(npts, xy_Ptr,
			 rMat_d_Ptr, rMat_s_Ptr,
			 tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
			 beamVec_Ptr, etaVec_Ptr,
			 tTh_Ptr, eta_Ptr, gVec_l_Ptr);

  /* Build and return the nested data structure */
  /* Note that Py_BuildValue with 'O' increases reference count */
  inner_tuple = Py_BuildValue("OO",tTh,eta);
  outer_tuple = Py_BuildValue("OO", inner_tuple, gVec_l);
  Py_DECREF(inner_tuple);
  Py_DECREF(tTh);
  Py_DECREF(eta);
  Py_DECREF(gVec_l);
  return outer_tuple;
}

/*
    Takes a list cartesian (x, y) pairs in the detector coordinates and calculates
    the associated reciprocal lattice (G) vectors and (bragg angle, azimuth) pairs
    with respect to the specified beam and azimth (eta) reference directions

    Required Arguments:
    xy_det -- (n, 2) ndarray or list-like input of n detector (x, y) points
    rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components to LAB FRAME
    rMat_s -- (n, 3, 3) ndarray, the COB taking SAMPLE FRAME components to LAB FRAME
    tVec_d -- (3, 1) ndarray, the translation vector connecting LAB to DETECTOR
    tVec_s -- (3, 1) ndarray, the translation vector connecting LAB to SAMPLE
    tVec_c -- (3, 1) ndarray, the translation vector connecting SAMPLE to CRYSTAL

    Optional Keyword Arguments:
    beamVec -- (1, 3) mdarray containing the incident beam direction components in the LAB FRAME
    etaVec  -- (1, 3) mdarray containing the reference azimuth direction components in the LAB FRAME

    Outputs:
    (n, 2) ndarray containing the (tTh, eta) pairs associated with each (x, y)
    (n, 3) ndarray containing the associated G vector directions in the LAB FRAME
    associated with gVecs
*/
static PyObject * detectorXYToGvecArray(PyObject * self, PyObject * args)
{
  PyArrayObject *xy_det, *rMat_d, *rMat_s,
		*tVec_d, *tVec_s, *tVec_c,
                *beamVec, *etaVec;
  PyArrayObject *tTh, *eta, *gVec_l;
  PyObject *inner_tuple, *outer_tuple;

  int dxy, drd, drs, dtd, dts, dtc, dbv, dev;
  npy_intp npts, dims[2];

  double *xy_Ptr, *rMat_d_Ptr, *rMat_s_Ptr,
         *tVec_d_Ptr, *tVec_s_Ptr, *tVec_c_Ptr,
         *beamVec_Ptr, *etaVec_Ptr;
  double *tTh_Ptr, *eta_Ptr, *gVec_l_Ptr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OOOOOOOO",
			 &xy_det,
			 &rMat_d, &rMat_s,
			 &tVec_d, &tVec_s, &tVec_c,
			 &beamVec, &etaVec)) return(NULL);
  if ( xy_det  == NULL || rMat_d == NULL || rMat_s == NULL ||
       tVec_d  == NULL || tVec_s == NULL || tVec_c == NULL ||
       beamVec == NULL || etaVec == NULL ) return(NULL);

  /* Verify shape of input arrays */
  dxy = PyArray_NDIM(xy_det);
  drd = PyArray_NDIM(rMat_d);
  drs = PyArray_NDIM(rMat_s);
  dtd = PyArray_NDIM(tVec_d);
  dts = PyArray_NDIM(tVec_s);
  dtc = PyArray_NDIM(tVec_c);
  dbv = PyArray_NDIM(beamVec);
  dev = PyArray_NDIM(etaVec);
  assert( dxy == 2 && drd == 2 && drs == 2 &&
	  dtd == 1 && dts == 1 && dtc == 1 &&
	  dbv == 1 && dev == 1);

  /* Verify dimensions of input arrays */
  npts = PyArray_DIMS(xy_det)[0];
  if (npts != PyArray_DIM(rMat_s, 0)) {
    PyErr_Format(PyExc_ValueError, "xy_det and rMat_s length mismatch %d vs %d",
                 (int)PyArray_DIM(xy_det, 0), (int)PyArray_DIM(rMat_s, 0));
    return NULL;
  }

  assert( PyArray_DIMS(xy_det)[1]  == 2 );
  assert( PyArray_DIMS(rMat_d)[0]  == 3 && PyArray_DIMS(rMat_d)[1] == 3 );
  assert( PyArray_DIMS(rMat_s)[0]  == 3 && PyArray_DIMS(rMat_s)[1] == 3 );
  assert( PyArray_DIMS(tVec_d)[0]  == 3 );
  assert( PyArray_DIMS(tVec_s)[0]  == 3 );
  assert( PyArray_DIMS(tVec_c)[0]  == 3 );
  assert( PyArray_DIMS(beamVec)[0] == 3 );
  assert( PyArray_DIMS(etaVec)[0]  == 3 );

  /* Allocate arrays for return values */
  dims[0] = npts; dims[1] = 3;
  gVec_l = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  tTh    = (PyArrayObject*)PyArray_EMPTY(1,&npts,NPY_DOUBLE,0);
  eta    = (PyArrayObject*)PyArray_EMPTY(1,&npts,NPY_DOUBLE,0);

  /* Grab data pointers into various arrays */
  xy_Ptr      = (double*)PyArray_DATA(xy_det);
  gVec_l_Ptr  = (double*)PyArray_DATA(gVec_l);

  tTh_Ptr     = (double*)PyArray_DATA(tTh);
  eta_Ptr     = (double*)PyArray_DATA(eta);

  rMat_d_Ptr  = (double*)PyArray_DATA(rMat_d);
  rMat_s_Ptr  = (double*)PyArray_DATA(rMat_s);

  tVec_d_Ptr  = (double*)PyArray_DATA(tVec_d);
  tVec_s_Ptr  = (double*)PyArray_DATA(tVec_s);
  tVec_c_Ptr  = (double*)PyArray_DATA(tVec_c);

  beamVec_Ptr = (double*)PyArray_DATA(beamVec);
  etaVec_Ptr  = (double*)PyArray_DATA(etaVec);

  /* Call the computational routine */
  detectorXYToGvecArray_cfunc(npts, xy_Ptr,
                              rMat_d_Ptr, rMat_s_Ptr,
                              tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
                              beamVec_Ptr, etaVec_Ptr,
                              tTh_Ptr, eta_Ptr, gVec_l_Ptr);

  /* Build and return the nested data structure */
  /* Note that Py_BuildValue with 'O' increases reference count */
  inner_tuple = Py_BuildValue("OO",tTh,eta);
  outer_tuple = Py_BuildValue("OO", inner_tuple, gVec_l);
  Py_DECREF(inner_tuple);
  Py_DECREF(tTh);
  Py_DECREF(eta);
  Py_DECREF(gVec_l);
  return outer_tuple;
}

static PyObject * oscillAnglesOfHKLs(PyObject * self, PyObject * args)
{
  PyArrayObject *hkls, *rMat_c, *bMat,
		*vInv_s, *beamVec, *etaVec;
  PyFloatObject *chi, *wavelength;
  PyArrayObject *oangs0, *oangs1;
  PyObject *return_tuple;

  int dhkls, drc, dbm, dvi, dbv, dev;
  npy_intp npts, dims[2];

  double *hkls_Ptr, chi_d,
	 *rMat_c_Ptr, *bMat_Ptr, wavelen_d,
	 *vInv_s_Ptr, *beamVec_Ptr, *etaVec_Ptr;
  double *oangs0_Ptr, *oangs1_Ptr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OOOOOOOO",
			 &hkls, &chi,
			 &rMat_c, &bMat, &wavelength,
			 &vInv_s, &beamVec, &etaVec)) return(NULL);
  if ( hkls    == NULL || chi == NULL ||
       rMat_c  == NULL || bMat == NULL || wavelength == NULL ||
       vInv_s  == NULL || beamVec == NULL || etaVec == NULL ) return(NULL);

  /* Verify shape of input arrays */
  dhkls = PyArray_NDIM(hkls);
  drc   = PyArray_NDIM(rMat_c);
  dbm   = PyArray_NDIM(bMat);
  dvi   = PyArray_NDIM(vInv_s);
  dbv   = PyArray_NDIM(beamVec);
  dev   = PyArray_NDIM(etaVec);
  assert( dhkls == 2 && drc == 2 && dbm == 2 &&
	  dvi   == 1 && dbv == 1 && dev == 1);

  /* Verify dimensions of input arrays */
  npts = PyArray_DIMS(hkls)[0];

  assert( PyArray_DIMS(hkls)[1]    == 3 );
  assert( PyArray_DIMS(rMat_c)[0]  == 3 && PyArray_DIMS(rMat_c)[1] == 3 );
  assert( PyArray_DIMS(bMat)[0]    == 3 && PyArray_DIMS(bMat)[1]   == 3 );
  assert( PyArray_DIMS(vInv_s)[0]  == 6 );
  assert( PyArray_DIMS(beamVec)[0] == 3 );
  assert( PyArray_DIMS(etaVec)[0]  == 3 );

  /* Allocate arrays for return values */
  dims[0] = npts; dims[1] = 3;
  oangs0 = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);
  oangs1 = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  /* Grab data pointers into various arrays */
  hkls_Ptr    = (double*)PyArray_DATA(hkls);

  chi_d       = PyFloat_AsDouble((PyObject*)chi);
  wavelen_d   = PyFloat_AsDouble((PyObject*)wavelength);

  rMat_c_Ptr  = (double*)PyArray_DATA(rMat_c);
  bMat_Ptr    = (double*)PyArray_DATA(bMat);

  vInv_s_Ptr  = (double*)PyArray_DATA(vInv_s);

  beamVec_Ptr = (double*)PyArray_DATA(beamVec);
  etaVec_Ptr  = (double*)PyArray_DATA(etaVec);

  oangs0_Ptr  = (double*)PyArray_DATA(oangs0);
  oangs1_Ptr  = (double*)PyArray_DATA(oangs1);

  /* Call the computational routine */
  oscillAnglesOfHKLs_cfunc(npts, hkls_Ptr, chi_d,
			   rMat_c_Ptr, bMat_Ptr, wavelen_d,
			   vInv_s_Ptr, beamVec_Ptr, etaVec_Ptr,
			   oangs0_Ptr, oangs1_Ptr);

  /* Build and return the list data structure */
  return_tuple = Py_BuildValue("OO",oangs0,oangs1);
  Py_DECREF(oangs1);
  Py_DECREF(oangs0);

  return return_tuple;
}

/******************************************************************************/
/* Utility Funtions */

static PyObject * unitRowVector(PyObject * self, PyObject * args)
{
  PyArrayObject *vecIn, *vecOut;
  double *cIn, *cOut;
  int d;
  npy_intp n;

  if ( !PyArg_ParseTuple(args,"O", &vecIn)) return(NULL);
  if ( vecIn  == NULL ) return(NULL);

  assert( PyArray_ISCONTIGUOUS(vecIn) );
  assert( PyArray_ISALIGNED(vecIn) );

  d = PyArray_NDIM(vecIn);

  assert(d == 1);

  n = PyArray_DIMS(vecIn)[0];

  vecOut = (PyArrayObject*)PyArray_EMPTY(d,PyArray_DIMS(vecIn),NPY_DOUBLE,0);

  cIn  = (double*)PyArray_DATA(vecIn);
  cOut = (double*)PyArray_DATA(vecOut);

  unitRowVector_cfunc(n,cIn,cOut);

  return((PyObject*)vecOut);
}

static PyObject * unitRowVectors(PyObject *self, PyObject *args)
{
  PyArrayObject *vecIn, *vecOut;
  double *cIn, *cOut;
  int d;
  npy_intp m,n;

  if ( !PyArg_ParseTuple(args,"O", &vecIn)) return(NULL);
  if ( vecIn  == NULL ) return(NULL);

  assert( PyArray_ISCONTIGUOUS(vecIn) );
  assert( PyArray_ISALIGNED(vecIn) );

  d = PyArray_NDIM(vecIn);

  assert(d == 2);

  m = PyArray_DIMS(vecIn)[0];
  n = PyArray_DIMS(vecIn)[1];

  vecOut = (PyArrayObject*)PyArray_EMPTY(d,PyArray_DIMS(vecIn),NPY_DOUBLE,0);

  cIn  = (double*)PyArray_DATA(vecIn);
  cOut = (double*)PyArray_DATA(vecOut);

  unitRowVectors_cfunc(m,n,cIn,cOut);

  return((PyObject*)vecOut);
}

static PyObject * makeDetectorRotMat(PyObject * self, PyObject * args)
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
  makeDetectorRotMat_cfunc(tPtr,rPtr);

  return((PyObject*)rMat);
}

static PyObject * makeOscillRotMat(PyObject * self, PyObject * args)
{
  PyArrayObject *oscillAngles, *rMat;
  int doa;
  npy_intp no, dims[2];
  double *oPtr, *rPtr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"O", &oscillAngles)) return(NULL);
  if ( oscillAngles == NULL ) return(NULL);

  /* Verify shape of input arrays */
  doa = PyArray_NDIM(oscillAngles);
  assert( doa == 1 );

  /* Verify dimensions of input arrays */
  no = PyArray_DIMS(oscillAngles)[0];
  assert( no == 2 );

  /* Allocate the result matrix with appropriate dimensions and type */
  dims[0] = 3; dims[1] = 3;
  rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  /* Grab pointers to the various data arrays */
  oPtr = (double*)PyArray_DATA(oscillAngles);
  rPtr = (double*)PyArray_DATA(rMat);

  /* Call the actual function */
  makeOscillRotMat_cfunc(oPtr[0], oPtr[1], rPtr);

  return((PyObject*)rMat);
}

static PyObject * makeOscillRotMatArray(PyObject * self, PyObject * args)
{
  PyObject *chiObj;
  double chi;
  PyArrayObject *omeArray, *rMat;
  int doa;
  npy_intp i, no, dims[3];
  double *oPtr, *rPtr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OO", &chiObj, &omeArray)) return(NULL);
  if ( chiObj == NULL ) return(NULL);
  if ( omeArray == NULL ) return(NULL);

  /* Get chi */
  chi = PyFloat_AsDouble(chiObj);
  if (chi == -1 && PyErr_Occurred()) return(NULL);

  /* Verify shape of input arrays */
  doa = PyArray_NDIM(omeArray);
  assert( doa == 1 );

  /* Verify dimensions of input arrays */
  no = PyArray_DIMS(omeArray)[0];

  /* Allocate the result matrix with appropriate dimensions and type */
  dims[0] = no; dims[1] = 3; dims[2] = 3;
  rMat = (PyArrayObject*)PyArray_EMPTY(3,dims,NPY_DOUBLE,0);

  /* Grab pointers to the various data arrays */
  oPtr = (double*)PyArray_DATA(omeArray);
  rPtr = (double*)PyArray_DATA(rMat);

  /* Call the actual function repeatedly */
  for (i = 0; i < no; ++i) {
      makeOscillRotMat_cfunc(chi, oPtr[i], rPtr + i*9);
  }

  return((PyObject*)rMat);
}

static PyObject * makeRotMatOfExpMap(PyObject * self, PyObject * args)
{
  PyArrayObject *expMap, *rMat;
  int de;
  npy_intp ne, dims[2];
  double *ePtr, *rPtr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"O", &expMap)) return(NULL);
  if ( expMap == NULL ) return(NULL);

  /* Verify shape of input arrays */
  de = PyArray_NDIM(expMap);
  assert( de == 1 );

  /* Verify dimensions of input arrays */
  ne = PyArray_DIMS(expMap)[0];
  assert( ne == 3 );

  /* Allocate the result matrix with appropriate dimensions and type */
  dims[0] = 3; dims[1] = 3;
  rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  /* Grab pointers to the various data arrays */
  ePtr = (double*)PyArray_DATA(expMap);
  rPtr = (double*)PyArray_DATA(rMat);

  /* Call the actual function */
  makeRotMatOfExpMap_cfunc(ePtr,rPtr);

  return((PyObject*)rMat);
}

static PyObject * makeRotMatOfQuat(PyObject * self, PyObject * args)
{
  PyArrayObject *quat, *rMat;
  int nq, ne, de;
  npy_intp dims2[2]={3, 3}, dims3[3];
  double *qPtr, *rPtr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"O", &quat)) return(NULL);
  if ( quat == NULL ) return(NULL);

  /* Verify shape of input arrays */
  de = PyArray_NDIM(quat);
  if (de == 1) {
    ne = PyArray_DIMS(quat)[0];
    assert( ne == 4 );
    nq = 1;
    /* Allocate the result matrix with appropriate dimensions and type */
    rMat = (PyArrayObject*)PyArray_EMPTY(2,dims2,NPY_DOUBLE,0);
  } else {
    assert( de == 2 );
    nq = PyArray_DIMS(quat)[0];
    ne = PyArray_DIMS(quat)[1];
    assert( ne == 4 );
    dims3[0] = nq; dims3[1] = 3; dims3[2] = 3;
    /* Allocate the result matrix with appropriate dimensions and type */
    rMat = (PyArrayObject*)PyArray_EMPTY(3,dims3,NPY_DOUBLE,0);
  }

  /* Grab pointers to the various data arrays */
  qPtr = (double*)PyArray_DATA(quat);
  rPtr = (double*)PyArray_DATA(rMat);

  /* Call the actual function */
  makeRotMatOfQuat_cfunc(nq, qPtr, rPtr);

  return((PyObject*)rMat);
}

static PyObject * makeBinaryRotMat(PyObject * self, PyObject * args)
{
  PyArrayObject *axis, *rMat;
  int da;
  npy_intp na, dims[2];
  double *aPtr, *rPtr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"O", &axis)) return(NULL);
  if ( axis  == NULL ) return(NULL);

  /* Verify shape of input arrays */
  da = PyArray_NDIM(axis);
  assert( da == 1 );

  /* Verify dimensions of input arrays */
  na = PyArray_DIMS(axis)[0];
  assert( na == 3 );

  /* Allocate the result matrix with appropriate dimensions and type */
  dims[0] = 3; dims[1] = 3;
  rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  /* Grab pointers to the various data arrays */
  aPtr = (double*)PyArray_DATA(axis);
  rPtr = (double*)PyArray_DATA(rMat);

  /* Call the actual function */
  makeBinaryRotMat_cfunc(aPtr,rPtr);

  return((PyObject*)rMat);
}

static PyObject * makeEtaFrameRotMat(PyObject * self, PyObject * args)
{
  PyArrayObject *bHat, *eHat, *rMat;
  int db, de;
  npy_intp nb, ne, dims[2];
  double *bPtr, *ePtr, *rPtr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"OO", &bHat,&eHat)) return(NULL);
  if ( bHat  == NULL || eHat == NULL ) return(NULL);

  /* Verify shape of input arrays */
  db = PyArray_NDIM(bHat);
  de = PyArray_NDIM(eHat);
  assert( db == 1 && de == 1);

  /* Verify dimensions of input arrays */
  nb = PyArray_DIMS(bHat)[0];
  ne = PyArray_DIMS(eHat)[0];
  assert( nb == 3 && ne == 3 );

  /* Allocate the result matrix with appropriate dimensions and type */
  dims[0] = 3; dims[1] = 3;
  rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

  /* Grab pointers to the various data arrays */
  bPtr = (double*)PyArray_DATA(bHat);
  ePtr = (double*)PyArray_DATA(eHat);
  rPtr = (double*)PyArray_DATA(rMat);

  /* Call the actual function */
  makeEtaFrameRotMat_cfunc(bPtr,ePtr,rPtr);

  return((PyObject*)rMat);
}

static PyObject * validateAngleRanges(PyObject * self, PyObject * args)
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
  validateAngleRanges_cfunc(na,aPtr,nmin,minPtr,maxPtr,rPtr,ccwVal);

  return((PyObject*)reflInRange);
}

static PyObject * rotate_vecs_about_axis(PyObject * self, PyObject * args)
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
  rotate_vecs_about_axis_cfunc(na,aPtr,nax0,axesPtr,nv0,vecsPtr,rPtr);

  return((PyObject*)rVecs);
}

static PyObject * quat_distance(PyObject * self, PyObject * args)
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
  dist = quat_distance_cfunc(nsym,q1Ptr,q2Ptr,qsymPtr);
  if (dist < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory");
    return NULL;
  }
  return(PyFloat_FromDouble(dist));
}

static PyObject * homochoricOfQuat(PyObject * self, PyObject * args)
{
  PyArrayObject *quat, *hVec;
  int nq, dq, ne;
  npy_intp dims[2];
  double *qPtr, *hPtr;

  /* Parse arguments */
  if ( !PyArg_ParseTuple(args,"O", &quat)) return(NULL);
  if ( quat == NULL ) return(NULL);

  /* Verify shape of input arrays */
  dq = PyArray_NDIM(quat);
  if (dq == 1) {
    ne = PyArray_DIMS(quat)[0];
    assert( ne == 4 );
    nq = 1;
    dims[0] = nq; dims[1] = 3;
    /* Allocate the result matrix with appropriate dimensions and type */
    hVec = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);
  } else {
    assert( dq == 2 );
    nq = PyArray_DIMS(quat)[0];
    ne = PyArray_DIMS(quat)[1];
    assert( ne == 4 );
    dims[0] = nq; dims[1] = 3;
    /* Allocate the result matrix with appropriate dimensions and type */
    hVec = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);
  }

  /* Grab pointers to the various data arrays */
  qPtr = (double*)PyArray_DATA(quat);
  hPtr = (double*)PyArray_DATA(hVec);

  /* Call the actual function */
  homochoricOfQuat_cfunc(nq, qPtr, hPtr);

  return((PyObject*)hVec);
}
