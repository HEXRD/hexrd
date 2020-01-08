
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION void
xy_to_gvec(size_t npts, double *xy, double *rMat_d, double *rMat_s,
           double *tVec_d, double *tVec_s, double *tVec_c,
           double *beamVec, double *etaVec,
           double *tTh, double *eta, double *gVec_l)
{
    size_t i, j, k;
    double nrm, phi, bVec[3], tVec1[3], tVec2[3], dHat_l[3], n_g[3];
    double rMat_e[9];

    /* Fill rMat_e */
    make_beam_rmat(beamVec, etaVec, rMat_e);

    /* Normalize the beam vector */
    nrm = 0.0;
    for (j=0; j<3; j++) {
        nrm += beamVec[j]*beamVec[j];
    }
    nrm = sqrt(nrm);
    if ( nrm > epsf ) {
        for (j=0; j<3; j++)
            bVec[j] = beamVec[j]/nrm;
    } else {
        for (j=0; j<3; j++)
            bVec[j] = beamVec[j];
    }

    /* Compute shift vector */
    for (j=0; j<3; j++) {
        tVec1[j] = tVec_d[j]-tVec_s[j];
        for (k=0; k<3; k++) {
            tVec1[j] -= rMat_s[3*j+k]*tVec_c[k];
        }
    }

    for (i=0; i<npts; i++) {
        /* Compute dHat_l vector */
        nrm = 0.0;
        for (j=0; j<3; j++) {
            dHat_l[j] = tVec1[j];
            for (k=0; k<2; k++) {
                dHat_l[j] += rMat_d[3*j+k]*xy[2*i+k];
            }
            nrm += dHat_l[j]*dHat_l[j];
        }
        if ( nrm > epsf ) {
            for (j=0; j<3; j++) {
                dHat_l[j] /= sqrt(nrm);
            }
        }

        /* Compute tTh */
        nrm = 0.0;
        for (j=0; j<3; j++) {
            nrm += bVec[j]*dHat_l[j];
        }
        tTh[i] = acos(nrm);

        /* Compute eta */
        for (j=0; j<2; j++) {
            tVec2[j] = 0.0;
            for (k=0; k<3; k++) {
                tVec2[j] += rMat_e[3*k+j]*dHat_l[k];
            }
        }
        eta[i] = atan2(tVec2[1],tVec2[0]);

        /* Compute n_g vector */
        nrm = 0.0;
        for (j=0; j<3; j++) {
            n_g[j] = bVec[(j+1)%3]*dHat_l[(j+2)%3]-bVec[(j+2)%3]*dHat_l[(j+1)%3];
            nrm += n_g[j]*n_g[j];
        }
        nrm = sqrt(nrm);
        for (j=0; j<3; j++) {
            n_g[j] /= nrm;
        }

        /* Rotate dHat_l vector */
        phi = 0.5*(M_PI-tTh[i]);
        rotate_vecs_about_axis(1, &phi, 1, n_g, 1, dHat_l, &gVec_l[3*i]);
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_detectorXYToGvec =
    "c module implementation of xy_to_gvec.\n"
    "Please use the Python wrapper.\n";

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
XRD_PYTHON_WRAPPER PyObject *
python_detectorXYToGvec(PyObject * self, PyObject * args)
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
    xy_to_gvec(npts, xy_Ptr,
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

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
