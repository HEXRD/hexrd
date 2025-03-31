#ifndef XRD_TRANSFORMS_PROTOTYPES_H
#define XRD_TRANSFORMS_PROTOTYPES_H

#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_types.h"
#endif

XRD_CFUNCTION void
angles_to_gvec(size_t nvecs, double *angs, double *bHat_l, double *eHat_l,
               double chi, double *rMat_c, double *gVec_c);

XRD_CFUNCTION void
angles_to_dvec(size_t nvecs, double *angs, double *bHat_l, double *eHat_l,
               double chi, double *rMat_c, double * gVec_c);


#define GV2XY_SINGLE_RMAT_S 1
XRD_CFUNCTION void
gvec_to_xy(size_t npts, const double *gVec_c_Ptr, const double *rMat_d_Ptr,
           const double *rMat_s_Ptr, const double *rMat_c_Ptr,
           const double *tVec_d_Ptr, const double *tVec_s_Ptr,
           const double *tVec_c_Ptr, const double *beamVec_Ptr,
           double * restrict result_Ptr, unsigned int flags);

XRD_CFUNCTION void
xy_to_gvec(size_t npts, double *xy, double *rMat_d, double *rMat_s,
           double *tVec_d, double *tVec_s, double *tVec_c,
           double *beamVec, double *etaVec, double *tTh, double *eta,
           double *gVec_l);

XRD_CFUNCTION void
oscill_angles_of_HKLs(size_t npts, double *hkls, double chi, double *rMat_c,
                      double *bMat, double wavelength, double *vInv_s,
                      double *beamVec, double *etaVec, double *oangs0,
                      double *oangs1);

/* this should probably be just a utility function in util... */
XRD_CFUNCTION int 
unit_row_vector(size_t n, double *cIn, double *cOut);

XRD_CFUNCTION void
unit_row_vectors(size_t m, size_t n, double *cIn, double *cOut);

/*
XRD_CFUNCTION void
make_detector_rmat(double *tPtr, double *rPtr);
*/

XRD_CFUNCTION void
make_sample_rmat(double chi, const double ome, double *result_rmat);

XRD_CFUNCTION void
make_sample_rmat_array(double chi, const double *ome_ptr, size_t ome_count, double *result_ptr);

XRD_CFUNCTION void
make_rmat_of_expmap(double *ePtr, double *rPtr);

XRD_CFUNCTION void
make_binary_rmat(const double *aPtr, double * restrict rPtr);

#define TF_MAKE_BEAM_RMAT_ERR_BEAM_ZERO 1
#define TF_MAKE_BEAM_RMAT_ERR_COLLINEAR 2
XRD_CFUNCTION int 
make_beam_rmat(double *bPtr, double *ePtr, double *rPtr); /* aka make_eta_frame_rotmat */

XRD_CFUNCTION void
validate_angle_ranges(size_t na, double *aPtr, size_t nr, double *minPtr,
                      double *maxPtr, bool *rPtr, int ccw);

XRD_CFUNCTION void
rotate_vecs_about_axis(size_t na, double *angles, size_t nax, double *axes,
                       size_t nv, double * vecs, double * rVecs);

XRD_CFUNCTION double
quat_distance(size_t nsym, double *q1, double *q2, double *qsym);


#endif /* XRD_TRANSFORMS_PROTOTYPES_H */
