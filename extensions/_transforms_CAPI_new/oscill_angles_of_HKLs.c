
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION void
oscill_angles_of_HKLs(size_t npts, double * hkls, double chi,
                      double * rMat_c, double * bMat, double wavelength,
                      double * vInv_s, double * beamVec, double * etaVec,
                      double * oangs0, double * oangs1)
{
    size_t i, j, k;
    bool crc = false;

    double gVec_e[3], gHat_c[3], gHat_s[3], bHat_l[3], eHat_l[3], oVec[2];
    double tVec0[3], tmpVec[3];
    double rMat_e[9], rMat_s[9];
    double a, b, c, sintht, cchi, schi;
    double abMag, phaseAng, rhs, rhsAng;
    double nrm0;

    /* Normalize the beam vector */
    nrm0 = 0.0;
    for (j=0; j<3; j++) {
        nrm0 += beamVec[j]*beamVec[j];
    }
    nrm0 = sqrt(nrm0);
    if ( nrm0 > epsf ) {
        for (j=0; j<3; j++)
            bHat_l[j] = beamVec[j]/nrm0;
    } else {
        for (j=0; j<3; j++)
            bHat_l[j] = beamVec[j];
    }

    /* Normalize the eta vector */
    nrm0 = 0.0;
    for (j=0; j<3; j++) {
        nrm0 += etaVec[j]*etaVec[j];
    }
    nrm0 = sqrt(nrm0);
    if ( nrm0 > epsf ) {
        for (j=0; j<3; j++)
            eHat_l[j] = etaVec[j]/nrm0;
    } else {
        for (j=0; j<3; j++)
            eHat_l[j] = etaVec[j];
    }

    /* Check for consistent reference coordiantes */
    nrm0 = 0.0;
    for (j=0; j<3; j++) {
        nrm0 += bHat_l[j]*eHat_l[j];
    }
    if ( fabs(nrm0) < 1.0-sqrt_epsf ) crc = true;

    /* Compute the sine and cosine of the oscillation axis tilt */
    cchi = cos(chi);
    schi = sin(chi);

    for (i=0; i<npts; i++) {

        /* Compute gVec_c */
        for (j=0; j<3; j++) {
            gHat_c[j] = 0.0;
            for (k=0; k<3; k++) {
                gHat_c[j] += bMat[3*j+k]*hkls[3L*i+k];
            }
        }

        /* Apply rMat_c to get gVec_s */
        for (j=0; j<3; j++) {
            gHat_s[j] = 0.0;
            for (k=0; k<3; k++) {
                gHat_s[j] += rMat_c[3*j+k]*gHat_c[k];
            }
        }

        /* Apply vInv_s to gVec_s and store in tmpVec*/
        tmpVec[0] = vInv_s[0]*gHat_s[0] + (vInv_s[5]*gHat_s[1] + vInv_s[4]*gHat_s[2])/sqrt(2.);
        tmpVec[1] = vInv_s[1]*gHat_s[1] + (vInv_s[5]*gHat_s[0] + vInv_s[3]*gHat_s[2])/sqrt(2.);
        tmpVec[2] = vInv_s[2]*gHat_s[2] + (vInv_s[4]*gHat_s[0] + vInv_s[3]*gHat_s[1])/sqrt(2.);

        /* Apply rMat_c.T to get stretched gVec_c and store norm in nrm0*/
        nrm0 = 0.0;
        for (j=0; j<3; j++) {
            gHat_c[j] = 0.0;
            for (k=0; k<3; k++) {
                gHat_c[j] += rMat_c[j+3*k]*tmpVec[k];
            }
            nrm0 += gHat_c[j]*gHat_c[j];
        }
        nrm0 = sqrt(nrm0);

        /* Normalize both gHat_c and gHat_s */
        if ( nrm0 > epsf ) {
            for (j=0; j<3; j++) {
                gHat_c[j] /= nrm0;
                gHat_s[j]  = tmpVec[j]/nrm0;
            }
        }

        /* Compute the sine of the Bragg angle */
        sintht = 0.5*wavelength*nrm0;

        /* Compute the coefficients of the harmonic equation */
        a = gHat_s[2]*bHat_l[0] + schi*gHat_s[0]*bHat_l[1] - cchi*gHat_s[0]*bHat_l[2];
        b = gHat_s[0]*bHat_l[0] - schi*gHat_s[2]*bHat_l[1] + cchi*gHat_s[2]*bHat_l[2];
        c =            - sintht - cchi*gHat_s[1]*bHat_l[1] - schi*gHat_s[1]*bHat_l[2];

        /* Form solution */
        abMag    = sqrt(a*a + b*b); assert( abMag > 0.0 );
        phaseAng = atan2(b,a);
        rhs      = c/abMag;

        if ( fabs(rhs) > 1.0 ) {
            for (j=0; j<3; j++)
                oangs0[3L*i+j] = NAN;
            for (j=0; j<3; j++)
                oangs1[3L*i+j] = NAN;
            continue;
        }

        rhsAng   = asin(rhs);

        /* Write ome angles */
        oangs0[3L*i+2] =        rhsAng - phaseAng;
        oangs1[3L*i+2] = M_PI - rhsAng - phaseAng;

        if ( crc ) {
            make_beam_rmat(bHat_l,eHat_l,rMat_e);

            oVec[0] = chi;

            oVec[1] = oangs0[3L*i+2];
            make_sample_rmat(oVec[0], oVec[1], rMat_s);

            for (j=0; j<3; j++) {
                tVec0[j] = 0.0;
                for (k=0; k<3; k++) {
                    tVec0[j] += rMat_s[3*j+k]*gHat_s[k];
                }
            }
            for (j=0; j<2; j++) {
                gVec_e[j] = 0.0;
                for (k=0; k<3; k++) {
                    gVec_e[j] += rMat_e[3*k+j]*tVec0[k];
                }
            }
            oangs0[3L*i+1] = atan2(gVec_e[1],gVec_e[0]);

            oVec[1] = oangs1[3L*i+2];
            make_sample_rmat(oVec[0], oVec[1], rMat_s);

            for (j=0; j<3; j++) {
                tVec0[j] = 0.0;
                for (k=0; k<3; k++) {
                    tVec0[j] += rMat_s[3*j+k]*gHat_s[k];
                }
            }
            for (j=0; j<2; j++) {
                gVec_e[j] = 0.0;
                for (k=0; k<3; k++) {
                    gVec_e[j] += rMat_e[3*k+j]*tVec0[k];
                }
            }
            oangs1[3L*i+1] = atan2(gVec_e[1],gVec_e[0]);

            oangs0[3L*i+0] = 2.0*asin(sintht);
            oangs1[3L*i+0] = oangs0[3L*i+0];
        }
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_oscillAnglesOfHKLs =
    "c module implementation of solve_omega.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_oscillAnglesOfHKLs(PyObject * self, PyObject * args)
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
    oscill_angles_of_HKLs(npts, hkls_Ptr, chi_d,
                          rMat_c_Ptr, bMat_Ptr, wavelen_d,
                          vInv_s_Ptr, beamVec_Ptr, etaVec_Ptr,
                          oangs0_Ptr, oangs1_Ptr);

    /* Build and return the list data structure */
    return_tuple = Py_BuildValue("OO",oangs0,oangs1);
    Py_DECREF(oangs1);
    Py_DECREF(oangs0);

    return return_tuple;
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
