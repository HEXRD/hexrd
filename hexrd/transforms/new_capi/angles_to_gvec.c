
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#  include "ndargs_helper.h"
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
    /*
      API interface in Python is:
      angs, beam_vec=None, eta_vec=None, chi=None, rmat_c=None

      currently, defaults are handled by the wrapper, so the C-module
      is always called with all arguments. Wrapper always guarantees
      that the input array is 2d.
      TODO: handle defaults here?
    */
    nah_array angs = { NULL, "angs", NAH_TYPE_DP_FP, { 3, NAH_DIM_ANY }};
    nah_array beam_vec = { NULL, "beam_vec", NAH_TYPE_DP_FP, { 3 }};
    nah_array eta_vec = { NULL, "eta_vec", NAH_TYPE_DP_FP, { 3 }};
    nah_array rmat_c = { NULL, "rmat_c", NAH_TYPE_DP_FP, { 3, 3 }};
    PyArrayObject *result = NULL;
    double chi;

    if ( !PyArg_ParseTuple(args,"O&O&O&dO&",
                           nah_array_converter, &angs,
                           nah_array_converter, &beam_vec,
                           nah_array_converter, &eta_vec,
                           &chi,
                           nah_array_converter, &rmat_c))
        return NULL;

    /* Allocate C-style array for return data */
    result = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(angs.pyarray),
                                           PyArray_SHAPE(angs.pyarray),
                                           NPY_DOUBLE,
                                           0);

    /* Call the actual function */
    angles_to_gvec(PyArray_DIM(angs.pyarray, 0),
                   (double *)PyArray_DATA(angs.pyarray),
                   (double *)PyArray_DATA(beam_vec.pyarray),
                   (double *)PyArray_DATA(eta_vec.pyarray),
                   chi,
                   (double *)PyArray_DATA(rmat_c.pyarray),
                   (double *)PyArray_DATA(result));

    /* Build and return the nested data structure */
    return((PyObject*)result);
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
