
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#  include "ndargs_helper.h"
#endif


static void
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

static const char *docstring_oscillAnglesOfHKLs =
    "c module implementation of solve_omega.\n"
    "Please use the Python wrapper.\n";

static PyObject *
python_oscillAnglesOfHKLs(PyObject * self, PyObject * args)
{
    nah_array hkls = { NULL, "hkls", NAH_TYPE_DP_FP, { 3, NAH_DIM_ANY }};
    nah_array rMat_c = { NULL, "rMat_c", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array bMat = { NULL, "bMat", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array vInv = { NULL, "vInv", NAH_TYPE_DP_FP, { 6 }};
    nah_array beamVec = { NULL, "beamVec", NAH_TYPE_DP_FP, { 3 }};
    nah_array etaVec = { NULL, "etaVec", NAH_TYPE_DP_FP, { 3 }};
    double chi, wavelen;
    PyArrayObject *oangs0 = NULL, *oangs1 = NULL;
    PyObject *result = NULL;
    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"O&dO&O&dO&O&O&",
                          nah_array_converter, &hkls,
                          &chi,
                          nah_array_converter, &rMat_c,
                          nah_array_converter, &bMat,
                          &wavelen,
                          nah_array_converter, &vInv,
                          nah_array_converter, &beamVec,
                          nah_array_converter, &etaVec))
        return NULL;

    /* Allocate arrays for return values */
    oangs0 = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(hkls.pyarray),
                                           PyArray_SHAPE(hkls.pyarray),
                                           NPY_DOUBLE, 0);
    if (!oangs0)
        goto fail_alloc;
    
    oangs1 = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(hkls.pyarray),
                                           PyArray_SHAPE(hkls.pyarray),
                                           NPY_DOUBLE, 0);
    if (!oangs1)
        goto fail_alloc;

    /* result is actually a tuple of oangs0 and oangs1 */
    result = Py_BuildValue("OO", oangs0, oangs1);
    if (!result)
        goto fail_alloc;
    

    /* Call the computational routine */
    oscill_angles_of_HKLs(PyArray_DIM(hkls.pyarray, 0),
                          (double *)PyArray_DATA(hkls.pyarray),
                          chi,
                          (double *)PyArray_DATA(rMat_c.pyarray),
                          (double *)PyArray_DATA(bMat.pyarray),
                          wavelen,
                          (double *)PyArray_DATA(vInv.pyarray),
                          (double *)PyArray_DATA(beamVec.pyarray),
                          (double *)PyArray_DATA(etaVec.pyarray),
                          (double *)PyArray_DATA(oangs0),
                          (double *)PyArray_DATA(oangs1));

    /* release the internal references to the arrays, set the
       pointers to NULL just for sanity */
    Py_DECREF(oangs1); oangs1 = NULL;
    Py_DECREF(oangs0); oangs0 = NULL;

    return result;
    
 fail_alloc:
    Py_XDECREF(result);
    Py_XDECREF(oangs0);
    Py_XDECREF(oangs1);
    
    return PyErr_NoMemory();
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
