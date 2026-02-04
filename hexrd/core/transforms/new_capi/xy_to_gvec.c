static void
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

static const char *docstring_detectorXYToGvec =
    "c module implementation of xy_to_gvec.\n"
    "Please use the Python wrapper.\n";

/*
  Takes a list cartesian (x, y) pairs in the detector coordinates and calculates
  the associated reciprocal lattice (G) vectors and (bragg angle, azimuth) pairs
  with respect to the specified beam and azimth (eta) reference directions.
  Note the wrapper that calls this expects:

  Required Arguments:
  xy_det  -- (n, 2) ndarray or list-like input of n detector (x, y) points
  rMat_d  -- (3, 3) ndarray, the COB taking DETECTOR FRAME components to LAB FRAME
  rMat_s  -- (3, 3) ndarray, the COB taking SAMPLE FRAME components to LAB FRAME
  tVec_d  -- (3, 1) ndarray, the translation vector connecting LAB to DETECTOR
  tVec_s  -- (3, 1) ndarray, the translation vector connecting LAB to SAMPLE
  tVec_c  -- (3, 1) ndarray, the translation vector connecting SAMPLE to CRYSTAL
  beamVec -- (1, 3) ndarray, the incident beam direction components in the LAB FRAME
  etaVec  -- (3, 3) ndarray
 
  Outputs:
  tuple ((tTh, eta), gvec_l) where:
  tTh    -- (n,) ndarray with the tTh associated with each (x, y)
  eta    -- (n,) ndarray with the eta associated with each (x, y)
  gvec_l -- (n, 3) ndarray containing the associated G vector directions in the
         LAB FRAME associated with gVecs
*/
static PyObject *
python_detectorXYToGvec(PyObject * self, PyObject * args)
{
    /* Right now, the Python wrapper guarantees that:
       xy_det is at least 2d
       optional parameter are always passed in, with its defaults
     */
    nah_array xy_det = { NULL, "xy_det", NAH_TYPE_DP_FP, { 2, NAH_DIM_ANY }};
    nah_array rmat_d = { NULL, "rmat_d", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array rmat_s = { NULL, "rmat_s", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array tvec_d = { NULL, "tvec_d", NAH_TYPE_DP_FP, { 3 }};
    nah_array tvec_s = { NULL, "tvec_s", NAH_TYPE_DP_FP, { 3 }};
    nah_array tvec_c = { NULL, "tvec_c", NAH_TYPE_DP_FP, { 3 }};
    nah_array beam_vec = { NULL, "beam_vec", NAH_TYPE_DP_FP, { 3 }};
    nah_array eta_vec = { NULL, "eta_Vec", NAH_TYPE_DP_FP, { 3 }};
    npy_intp npts, dims[2];
    PyObject *inner_tuple = NULL, *outer_tuple = NULL;
    PyArrayObject *gvec_l = NULL;
    PyArrayObject *tTh = NULL;
    PyArrayObject *eta = NULL;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"O&O&O&O&O&O&O&O&",
                          nah_array_converter, &xy_det,
                          nah_array_converter, &rmat_d,
                          nah_array_converter, &rmat_s,
                          nah_array_converter, &tvec_d,
                          nah_array_converter, &tvec_s,
                          nah_array_converter, &tvec_c,
                          nah_array_converter, &beam_vec,
                          nah_array_converter, &eta_vec))
        return NULL;

    /* Allocate arrays and tuples for return values */
    npts = PyArray_DIM(xy_det.pyarray, 0);
    dims[0] = npts;
    dims[1] = 3;
    gvec_l = (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (!gvec_l)
        goto fail_alloc;
    
    tTh    = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    if (!tTh)
        goto fail_alloc;
    
    eta    = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_DOUBLE,0);
    if (!eta)
        goto fail_alloc;
    
    inner_tuple = Py_BuildValue("OO", tTh, eta);
    if (!inner_tuple)
        goto fail_alloc;
    
    outer_tuple = Py_BuildValue("OO", inner_tuple, gvec_l);
    if (!outer_tuple)
        goto fail_alloc;
    
    /* Call the computational routine */
    xy_to_gvec(npts,
               (double *)PyArray_DATA(xy_det.pyarray),
               (double *)PyArray_DATA(rmat_d.pyarray),
               (double *)PyArray_DATA(rmat_s.pyarray),
               (double *)PyArray_DATA(tvec_d.pyarray),
               (double *)PyArray_DATA(tvec_s.pyarray),
               (double *)PyArray_DATA(tvec_c.pyarray),
               (double *)PyArray_DATA(beam_vec.pyarray),
               (double *)PyArray_DATA(eta_vec.pyarray),
               (double *)PyArray_DATA(tTh),
               (double *)PyArray_DATA(eta),
               (double *)PyArray_DATA(gvec_l));

    /*
      At this point, no allocation may fail. Release all redundant references
      (as the outer_tuple will hold its references).

      Beyond this point the cleanup after fail_alloc shouldn't, but clearing
      the associated variables makes the code less error-prone.
    */
    Py_DECREF(inner_tuple); inner_tuple = NULL;
    Py_DECREF(tTh); tTh = NULL;
    Py_DECREF(eta); eta = NULL;
    Py_DECREF(gvec_l); gvec_l = NULL;

    return outer_tuple;
    
 fail_alloc:
    Py_XDECREF(outer_tuple);
    Py_XDECREF(inner_tuple);
    Py_XDECREF(eta);
    Py_XDECREF(tTh);
    Py_XDECREF(gvec_l);
    
    return PyErr_NoMemory();
}

