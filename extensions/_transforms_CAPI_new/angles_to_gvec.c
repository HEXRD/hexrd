
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION void
angles_to_gvec(size_t nvecs, double * angs,
               double * bHat_l, double * eHat_l,
               double chi, double * rMat_c,
               double * gVec_c)
{
    /*
     *  takes an angle spec (2*theta, eta, omega) for nvecs g-vectors and
     *  returns the unit g-vector components in the crystal frame
     *
     *  For unit g-vector in the lab frame, spec rMat_c = Identity and
     *  overwrite the omega values with zeros
     */
    size_t i, j, k, l;
    double rMat_e[9], rMat_s[9], rMat_ctst[9];
    double gVec_e[3], gVec_l[3], gVec_c_tmp[3];

    /* Need eta frame cob matrix (could omit for standard setting) */
    make_beam_rmat(bHat_l, eHat_l, rMat_e);

    /* make vector array */
    for (i=0; i<nvecs; i++) {
        /* components in BEAM frame */
        gVec_e[0] = cos(0.5*angs[3*i]) * cos(angs[3*i+1]);
        gVec_e[1] = cos(0.5*angs[3*i]) * sin(angs[3*i+1]);
        gVec_e[2] = sin(0.5*angs[3*i]);

        /* take from beam frame to lab frame */
        for (j=0; j<3; j++) {
            gVec_l[j] = 0.0;
            for (k=0; k<3; k++) {
                gVec_l[j] += rMat_e[3*j+k]*gVec_e[k];
            }
        }

        /* need pointwise rMat_s according to omega */
        make_sample_rmat(chi, angs[3*i+2], rMat_s);

        /* Compute dot(rMat_c.T, rMat_s.T) and hit against gVec_l */
        for (j=0; j<3; j++) {
            for (k=0; k<3; k++) {
                rMat_ctst[3*j+k] = 0.0;
                for (l=0; l<3; l++) {
                    rMat_ctst[3*j+k] += rMat_c[3*l+j]*rMat_s[3*k+l];
                }
            }
            gVec_c_tmp[j] = 0.0;
            for (k=0; k<3; k++) {
                gVec_c_tmp[j] += rMat_ctst[3*j+k]*gVec_l[k];
            }
            gVec_c[3*i+j] = gVec_c_tmp[j];
        }
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_anglesToGVec =
    "c module implementation of angles_to_gvec.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_anglesToGVec(PyObject * self, PyObject * args)
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
    angles_to_gvec(nvecs, angs_ptr,
                   bHat_l_ptr, eHat_l_ptr,
                   chi, rMat_c_ptr,
                   gVec_c_ptr);

    /* Build and return the nested data structure */
    return((PyObject*)gVec_c);
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
