
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


static void
gvec_to_xy_single(double *gVec_c, double *rMat_d, double *rMat_sc,
                           double *tVec_d, double *bHat_l, double *nVec_l,
                           double num, double *P0_l, double *result)
{
    int j, k;
    double bDot, ztol, denom, u;
    double gHat_c[3], gVec_l[3], dVec_l[3], P2_l[3], P2_d[3];
    double brMat[9];

    ztol = epsf;

    /* Compute unit reciprocal lattice vector in crystal frame w/o translation */
    unit_row_vector(3, gVec_c, gHat_c);

    /*
	 * Compute unit reciprocal lattice vector in lab frame
	 * and dot with beam vector
	 */
    bDot = 0.0;
    for (j=0; j<3; j++) {
        gVec_l[j] = 0.0;
        for (k=0; k<3; k++)
            gVec_l[j] += rMat_sc[3*j+k]*gHat_c[k];

        bDot -= bHat_l[j]*gVec_l[j];
    }

    if ( bDot >= ztol && bDot <= 1.0-ztol ) {
        /*
		 * If we are here diffraction is possible so increment
		 * the number of admissable vectors
		 */
        make_binary_rmat(gVec_l, brMat);

        denom = 0.0;
        for (j=0; j<3; j++) {
            dVec_l[j] = 0.0;
            for (k=0; k<3; k++)
                dVec_l[j] -= brMat[3*j+k]*bHat_l[k];

            denom += nVec_l[j]*dVec_l[j];
        }

        if ( denom < -ztol ) {

            u = num/denom;

            for (j=0; j<3; j++)
                P2_l[j] = P0_l[j]+u*dVec_l[j];

            for (j=0; j<2; j++) {
                P2_d[j] = 0.0;
                for (k=0; k<3; k++)
                    P2_d[j] += rMat_d[3*k+j]*(P2_l[k]-tVec_d[k]);
                result[j] = P2_d[j];
            }
        } else {
            result[0] = NAN;
            result[1] = NAN;
        }

    } else {
        result[0] = NAN;
        result[1] = NAN;
    }
}

XRD_CFUNCTION void
gvec_to_xy(size_t npts, double *gVec_c, double *rMat_d,
           double *rMat_s, double *rMat_c, double *tVec_d,
           double *tVec_s, double *tVec_c, double *beamVec,
           double * result)
{
    size_t i, j, k, l;
    double num;
    double nVec_l[3], bHat_l[3], P0_l[3], P3_l[3];
    double rMat_sc[9];

    /* Normalize the beam vector */
    unit_row_vector(3, beamVec, bHat_l);

    /* Initialize the detector normal and frame origins */
    num = 0.0;
    for (j=0; j<3; j++) {
        nVec_l[j] = 0.0;
        P0_l[j]   = tVec_s[j];

        for (k=0; k<3; k++) {
            nVec_l[j] += rMat_d[3*j+k]*Zl[k];
            P0_l[j]   += rMat_s[3*j+k]*tVec_c[k];
        }

        P3_l[j] = tVec_d[j];

        num += nVec_l[j]*(P3_l[j]-P0_l[j]);
    }

    /* Compute the matrix product of rMat_s and rMat_c */
    for (j=0; j<3; j++) {
        for (k=0; k<3; k++) {
            rMat_sc[3*j+k] = 0.0;
            for (l=0; l<3; l++) {
                rMat_sc[3*j+k] += rMat_s[3*j+l]*rMat_c[3*l+k];
            }
        }
    }

    for (i=0L; i<npts; i++) {
        gvec_to_xy_single(&gVec_c[3*i], rMat_d, rMat_sc, tVec_d,
                          bHat_l, nVec_l, num,
                          P0_l, &result[2*i]);
    }
}

/*
 * The only difference between this and the non-Array version
 * is that rMat_s is an array of matrices of length npts instead
 * of a single matrix.
 */
XRD_CFUNCTION void
gvec_to_xy_array(size_t npts, double *gVec_c, double *rMat_d,
                 double *rMat_s, double *rMat_c, double *tVec_d,
                 double *tVec_s, double *tVec_c, double *beamVec,
                 double * result)
{
    size_t i, j, k, l;

    double num;
    double nVec_l[3], bHat_l[3], P0_l[3], P3_l[3];
    double rMat_sc[9];

    /* Normalize the beam vector */
    unit_row_vector(3,beamVec,bHat_l);

    for (i=0L; i<npts; i++) {
        /* Initialize the detector normal and frame origins */
        num = 0.0;
        for (j=0; j<3; j++) {
            nVec_l[j] = 0.0;
            P0_l[j]   = tVec_s[j];

            for (k=0; k<3; k++) {
                nVec_l[j] += rMat_d[3*j+k]*Zl[k];
                P0_l[j]   += rMat_s[9*i + 3*j+k]*tVec_c[k];
            }

            P3_l[j] = tVec_d[j];

            num += nVec_l[j]*(P3_l[j]-P0_l[j]);
        }

        /* Compute the matrix product of rMat_s and rMat_c */
        for (j=0; j<3; j++) {
            for (k=0; k<3; k++) {
                rMat_sc[3*j+k] = 0.0;
                for (l=0; l<3; l++) {
                    rMat_sc[3*j+k] += rMat_s[9*i + 3*j+l]*rMat_c[3*l+k];
                }
            }
        }

        gvec_to_xy_single(&gVec_c[3*i], rMat_d, rMat_sc, tVec_d,
                          bHat_l, nVec_l, num,
                          P0_l, &result[2*i]);
    }
}

#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_gvecToDetectorXY =
    "c module implementation of gvec_to_xy (single sample).\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER const char *docstring_gvecToDetectorXYArray =
    "c module implementation of gvec_to_xy (multi sample).\n"
    "Please use the Python wrapper.\n";

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
XRD_PYTHON_WRAPPER PyObject *
python_gvecToDetectorXY(PyObject * self, PyObject * args)
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
    gvec_to_xy(npts, gVec_c_Ptr,
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
XRD_PYTHON_WRAPPER PyObject *
python_gvecToDetectorXYArray(PyObject * self, PyObject * args)
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
    gvec_to_xy_array(npts, gVec_c_Ptr,
                     rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
                     tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
                     beamVec_Ptr,
                     result_Ptr);

    /* Build and return the nested data structure */
    return((PyObject*)result);
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
