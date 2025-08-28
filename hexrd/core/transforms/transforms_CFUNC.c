#define _USE_MATH_DEFINES
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "transforms_CFUNC.h"

/*
 * Microsoft's C compiler, when running in C mode, does not support the inline
 * keyword. However it does support an __inline one.
 *
 * So if compiling with MSC, just use __inline as inline
 */
#if defined(_MSC_VER)
#   define inline __inline
#endif

/*
 * For now, disable C99 codepaths
 */
#define USE_C99_CODE 0
#if ! defined(USE_C99_CODE)
#   if defined(__STDC__)
#       if (__STD_VERSION__ >= 199901L)
#           define USE_C99_CODE 1
#       else
#           define USE_C99_CODE 0
#       endif
#   endif
#endif

#if ! USE_C99_CODE
/*
 * Just remove any "restrict" keyword that may be present.
 */
#define restrict
#endif

static double epsf      = 2.2e-16;
static double sqrt_epsf = 1.5e-8;
static double Zl[3] = {0.0,0.0,1.0};


/******************************************************************************/
/* Functions */
#if USE_C99_CODE
static inline
double *
v3_v3s_inplace_add(double *dst_src1,
                   const double *src2, int stride)
{
    dst_src1[0] += src2[0];
    dst_src1[1] += src2[1*stride];
    dst_src1[2] += src2[2*stride];
    return dst_src1;
}

static inline
double *
v3_v3s_add(const double *src1,
           const double *src2, int stride,
           double * restrict dst)
{
    dst[0] = src1[0] + src2[0];
    dst[1] = src1[1] + src2[1*stride];
    dst[2] = src1[2] + src2[2*stride];

    return dst;
}

static inline
double *
v3_v3s_inplace_sub(double *dst_src1,
                   const double *src2, int stride)
{
    dst_src1[0] -= src2[0];
    dst_src1[1] -= src2[1*stride];
    dst_src1[2] -= src2[2*stride];
    return dst_src1;
}

static inline
double *
v3_v3s_sub(const double *src1,
           const double *src2, int stride,
           double * restrict dst)
{
    dst[0] = src1[0] - src2[0];
    dst[1] = src1[1] - src2[1*stride];
    dst[2] = src1[2] - src2[2*stride];

    return dst;
}

static inline
double *
v3_inplace_normalize(double * restrict v)
{
    double sqr_norm = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];

    if (sqr_norm > epsf) {
        double normalize_factor = 1./sqrt(sqr_norm);
        v[0] *= normalize_factor;
        v[1] *= normalize_factor;
        v[2] *= normalize_factor;
    }

    return v;
}

static inline
double *
v3_normalize(const double *in,
             double * restrict out)
{
    double in0 = in[0], in1 = in[1], in2 = in[2];
    double sqr_norm = in0*in0 + in1*in1 + in2*in2;

    if (sqr_norm > epsf) {
        double normalize_factor = 1./sqrt(sqr_norm);
        out[0] = in0 * normalize_factor;
        out[1] = in1 * normalize_factor;
        out[2] = in2 * normalize_factor;
    } else {
        out[0] = in0;
        out[1] = in1;
        out[2] = in2;
    }

    return out;
}

static inline
double *
m33_inplace_transpose(double * restrict m)
{
    double e1 = m[1];
    double e2 = m[2];
    double e5 = m[5];
    m[1] = m[3];
    m[2] = m[6];
    m[5] = m[7];
    m[3] = e1;
    m[6] = e2;
    m[7] = e5;

    return m;
}

static inline
double *
m33_transpose(const double *m,
              double * restrict dst)
{
    dst[0] = m[0]; dst[1] = m[3]; dst[2] = m[6];
    dst[3] = m[1]; dst[4] = m[4]; dst[5] = m[7];
    dst[7] = m[2]; dst[8] = m[5]; dst[9] = m[9];

    return dst;
}

static inline
double
v3_v3s_dot(const double *v1,
           const double *v2, int stride)
{
    return v1[0]*v2[0] + v1[1]*v2[stride] + v1[2]*v2[2*stride];
}


/* 3x3 matrix by strided 3 vector product -------------------------------------
   hopefully a constant stride will be optimized
 */
static inline
double *
m33_v3s_multiply(const double *m,
                 const double *v, int stride,
                 double * restrict dst)
{
    dst[0] = m[0]*v[0] + m[1]*v[stride] + m[2]*v[2*stride];
    dst[1] = m[3]*v[0] + m[4]*v[stride] + m[5]*v[2*stride];
    dst[2] = m[6]*v[0] + m[7]*v[stride] + m[8]*v[2*stride];

    return dst;
}

/* transposed 3x3 matrix by strided 3 vector product --------------------------
 */
static inline
double *
v3s_m33t_multiply(const double *v, int stride,
                  const double *m,
                  double * restrict dst)
{
    double v0 = v[0]; double v1 = v[stride]; double v2 = v[2*stride];
    dst[0] = v0*m[0] + v1*m[1] + v2*m[2];
    dst[1] = v0*m[3] + v1*m[4] + v2*m[5];
    dst[2] = v0*m[6] + v1*m[7] + v2*m[8];

    return dst;
}

static inline
double *
v3s_m33_multiply(const double *v, int stride,
                 const double *m,
                 double * restrict dst)
{
    double v0 = v[0]; double v1 = v[stride]; double v2 = v[2*stride];
    dst[0] = v0*m[0] + v1*m[3] + v2*m[6];
    dst[1] = v0*m[1] + v1*m[4] + v2*m[7];
    dst[2] = v0*m[2] + v1*m[5] + v2*m[8];

    return dst;
}

static inline
double *
m33t_v3s_multiply(const double *m,
                  const double *v, int stride,
                  double * restrict dst)
{
    dst[0] = m[0]*v[0] + m[3]*v[stride] + m[6]*v[2*stride];
    dst[1] = m[1]*v[0] + m[4]*v[stride] + m[7]*v[2*stride];
    dst[2] = m[2]*v[0] + m[5]*v[stride] + m[8]*v[2*stride];

    return dst;
}

static inline
double *
m33_m33_multiply(const double *src1,
                 const double *src2,
                 double * restrict dst)
{
    v3s_m33_multiply(src1 + 0, 1, src2, dst+0);
    v3s_m33_multiply(src1 + 3, 1, src2, dst+3);
    v3s_m33_multiply(src1 + 6, 1, src2, dst+6);

    return dst;
}

static inline
double *
m33t_m33_multiply(const double *src1,
                  const double *src2,
                  double * restrict dst)
{
    v3s_m33_multiply(src1 + 0, 3, src2, dst+0);
    v3s_m33_multiply(src1 + 1, 3, src2, dst+3);
    v3s_m33_multiply(src1 + 2, 3, src2, dst+6);

    return dst;
}

static inline
double *
m33_m33t_multiply(const double *src1,
                  const double *src2,
                  double * restrict dst)
{
    return m33_inplace_transpose(m33t_m33_multiply(src2, src1, dst));
}

static inline
double *
m33t_m33t_multiply(const double *src1,
                   const double *src2,
                   double * restrict dst)
{
    return m33_inplace_transpose(m33_m33_multiply(src2, src1, dst));
}

#endif

#if USE_C99_CODE
static inline
void anglesToGvec_single(double *v3_ang, double *m33_e,
                         double chi, double *m33_c,
                         double * restrict v3_c)
{
    double v3_g[3], v3_tmp1[3], v3_tmp2[3], m33_s[9], m33_ctst[9];

    /* build g */
    double cx = cos(0.5*v3_ang[0]);
    double sx = sin(0.5*v3_ang[0]);
    double cy = cos(v3_ang[1]);
    double sy = sin(v3_ang[1]);
    v3_g[0] = cx*cy;
    v3_g[1] = cx*sy;
    v3_g[2] = sx;

    /* build S */
    makeOscillRotMat_cfunc(chi, v3_ang[2], m33_s);

    /* beam frame to lab frame */
    /* eval the chain:
       C.T _dot_ S.T _dot_ E _dot_ g
     */
    m33_v3s_multiply (m33_e, v3_g,    1, v3_tmp1); /* E _dot_ g */
    m33t_v3s_multiply(m33_s, v3_tmp1, 1, v3_tmp2); /* S.T _dot_ E _dot_ g */
    m33t_v3s_multiply(m33_c, v3_tmp2, 1, v3_c); /* the whole lot */
}

void anglesToGvec_cfunc(long int nvecs, double * angs,
            double * bHat_l, double * eHat_l,
            double chi, double * rMat_c,
            double * gVec_c)
{
    double m33_e[9];

    makeEtaFrameRotMat_cfunc(bHat_l, eHat_l, m33_e);

    for (int i = 0; i<nvecs; i++) {
        double * ang = angs + 3*i;
        double * restrict v3_c = gVec_c + 3*i;

        anglesToGvec_single(ang, m33_e, chi, rMat_c, v3_c);
    }
}

#else
void anglesToGvec_cfunc(long int nvecs, double * angs,
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
    int i, j, k, l;
    double rMat_e[9], rMat_s[9], rMat_ctst[9];
    double gVec_e[3], gVec_l[3], gVec_c_tmp[3];

    /* Need eta frame cob matrix (could omit for standard setting) */
    makeEtaFrameRotMat_cfunc(bHat_l, eHat_l, rMat_e);

    /* make vector array */
    for (i = 0; i < nvecs; i++) {
        /* components in BEAM frame */
        gVec_e[0] = cos(0.5*angs[3*i]) * cos(angs[3*i+1]);
        gVec_e[1] = cos(0.5*angs[3*i]) * sin(angs[3*i+1]);
        gVec_e[2] = sin(0.5*angs[3*i]);

        /* take from beam frame to lab frame */
        for (j = 0; j < 3; j++) {
            gVec_l[j] = 0.0;
            for (k = 0; k < 3; k++) {
                gVec_l[j] += rMat_e[3*j+k]*gVec_e[k];
            }
        }

        /* need pointwise rMat_s according to omega */
        makeOscillRotMat_cfunc(chi, angs[3*i+2], rMat_s);

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
#endif

void anglesToDvec_cfunc(long int nvecs, double * angs,
			double * bHat_l, double * eHat_l,
			double chi, double * rMat_c,
			double * gVec_c)
{
    /*
     *  takes an angle spec (2*theta, eta, omega) for nvecs g-vectors and
     *  returns the unit d-vector components in the crystal frame
     *
     *  For unit d-vector in the lab frame, spec rMat_c = Identity and
     *  overwrite the omega values with zeros
     */
    int i, j, k, l;

    double rMat_e[9], rMat_s[9], rMat_ctst[9];
    double gVec_e[3], gVec_l[3], gVec_c_tmp[3];

    /* Need eta frame cob matrix (could omit for standard setting) */
    makeEtaFrameRotMat_cfunc(bHat_l, eHat_l, rMat_e);

    /* make vector array */
    for (i=0; i<nvecs; i++) {
	double c0 = cos(angs[3*i]);
	double c1 = cos(angs[3*i+1]);
	double s0 = sin(angs[3*i]);
	double s1 = sin(angs[3*i+1]);
	
	/* components in BEAM frame */
        gVec_e[0] = s0*c1;
        gVec_e[1] = s0*s1;
        gVec_e[2] = -c0;

        /* take from BEAM frame to LAB frame */
        for (j=0; j<3; j++) {
            gVec_l[j] = 0.0;
            for (k=0; k<3; k++) {
                gVec_l[j] += rMat_e[3*j+k]*gVec_e[k];
            }
        }

        /* need pointwise rMat_s according to omega */
        makeOscillRotMat_cfunc(chi, angs[3*i+2], rMat_s);

        /* compute dot(rMat_c.T, rMat_s.T) and hit against gVec_l */
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

#if USE_C99_CODE
static inline
void gvecToDetectorXYOne_cfunc(double * gVec_c, double * rMat_d,
                   double * rMat_sc, double * tVec_d,
                   double * bHat_l,
                   double * nVec_l, double num, double * P0_l,
                               double * restrict result)
{
    /*
     * Compute unit reciprocal lattice vector in crystal frame w/o
     * translation
     */
    double gHat_c[3];
    v3_normalize(gVec_c, gHat_c);

    /*
     * Compute unit reciprocal lattice vector in lab frame and dot with beam
     * vector
     */
    double gVec_l[3];
    m33_v3s_multiply(rMat_sc, gHat_c, 1, gVec_l);

    double bDot = -v3_v3s_dot(bHat_l, gVec_l, 1);
    double ztol = epsf;
    if ( bDot >= ztol && bDot <= 1.0-ztol ) {
        /*
         * If we are here diffraction is possible so increment the number of
         * admissable vectors
         */
        double brMat[9];
        makeBinaryRotMat_cfunc(gVec_l, brMat);

        double dVec_l[3];
        m33_v3s_multiply(brMat, bHat_l, 1, dVec_l);
        double denom = v3_v3s_dot(nVec_l, dVec_l, 1);

        if (denom > ztol) {
            double u = num/denom;
            double v3_tmp[3];

            /* v3_tmp = P0_l + u*dVec_l - tVec_d */
            for (int j=0; j<3; j++)
                v3_tmp[j] = P0_l[j] + u*dVec_l[j] - tVec_d[j];

            result[0] = v3_v3s_dot(v3_tmp, rMat_d + 0, 3);
            result[1] = v3_v3s_dot(v3_tmp, rMat_d + 1, 3);

            /* result when computation can be finished */
            return;
        }
    }

    /* default result when computation can't be finished */
    result[0] = NAN;
    result[1] = NAN;
}

/*
 * The only difference between this and the non-Array version
 * is that rMat_s is an array of matrices of length npts instead
 * of a single matrix.
 */
void gvecToDetectorXYArray_cfunc(long int npts, double * gVec_c_array,
                                 double * rMat_d, double * rMat_s_array, double * rMat_c,
                                 double * tVec_d, double * tVec_s, double * tVec_c,
                                 double * beamVec, double * result_array)
{
    /* Normalize the beam vector */
    double bHat_l[3];
    v3_normalize(beamVec, bHat_l);
    double nVec_l[3];
    m33_v3s_multiply(rMat_d, Zl, 1, nVec_l);

    for (size_t i = 0; i < npts; i++) {
        double *rMat_s = rMat_s_array + 9*i;
        double *gVec_c = gVec_c_array + 3*i;
        double * restrict result = result_array + 2*i;
        /* Initialize the detector normal and frame origins */

        double P0_l[3];
        m33_v3s_multiply(rMat_s, tVec_c, 1, P0_l);
        v3_v3s_inplace_add(P0_l, tVec_s, 1);

        double P3_l_minus_P0_l[3];
        v3_v3s_sub(tVec_d, P0_l, 1, P3_l_minus_P0_l);
        double num = v3_v3s_dot(nVec_l, P3_l_minus_P0_l, 1);

        double gHat_c[3];
        v3_normalize(gVec_c, gHat_c);
        /*
        double rMat_sc[9];
        m33_m33_multiply(rMat_s, rMat_c, rMat_sc);
        double gVec_l[3];
        m33_v3s_multiply(rMat_sc, gHat_c, 1, gVec_l);
        */
        double tmp_vec[3], gVec_l[3];
        m33_v3s_multiply(rMat_c, gHat_c, 1, tmp_vec);
        m33_v3s_multiply(rMat_s, tmp_vec, 1, gVec_l);

        double bDot = -v3_v3s_dot(bHat_l, gVec_l, 1);
        double ztol = epsf;

        if (bDot < ztol || bDot > 1.0-ztol) {
            result[0] = NAN; result[1] = NAN;
            continue;
        }

        double brMat[9];
        makeBinaryRotMat_cfunc(gVec_l, brMat);

        double dVec_l[3];
        m33_v3s_multiply(brMat, bHat_l, 1, dVec_l);
        double denom = v3_v3s_dot(nVec_l, dVec_l, 1);
        if (denom < ztol) {
            result[0] = NAN; result[1] = NAN;
            continue;
        }

        double u = num/denom;
        double v3_tmp[3];
        for (int j=0; j < 3; j++)
            v3_tmp[j] = u*dVec_l[j] - P3_l_minus_P0_l[j];

        result[0] = v3_v3s_dot(v3_tmp, rMat_d + 0, 3);
        result[1] = v3_v3s_dot(v3_tmp, rMat_d + 1, 3);
    }
}

#else
void gvecToDetectorXYOne_cfunc(double * gVec_c, double * rMat_d,
                               double * rMat_sc, double * tVec_d,
                               double * bHat_l,
                               double * nVec_l, double num, double * P0_l,
                               double * result)
{
    int j, k;
    double bDot, ztol, denom, u;
    double gHat_c[3], gVec_l[3], dVec_l[3], P2_l[3], P2_d[3];
    double brMat[9];

    ztol = epsf;

    /* Compute unit reciprocal lattice vector in crystal frame w/o
       translation */
    unitRowVector_cfunc(3, gVec_c, gHat_c);

    /* Compute unit reciprocal lattice vector in lab frame and dot with beam
       vector */
    bDot = 0.0;
    for (j=0; j<3; j++) {
        gVec_l[j] = 0.0;
        for (k=0; k<3; k++)
            gVec_l[j] += rMat_sc[3*j+k]*gHat_c[k];

        bDot -= bHat_l[j]*gVec_l[j];
    }

    if ( bDot >= ztol && bDot <= 1.0-ztol ) {
        /* If we are here diffraction is possible so increment the number of
           admissable vectors */
        makeBinaryRotMat_cfunc(gVec_l, brMat);

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
            /* result when computation can be finished */
            return;
        }
    }
    /* default result when computation can't be finished */
    result[0] = NAN;
    result[1] = NAN;
}

/*
 * The only difference between this and the non-Array version
 * is that rMat_s is an array of matrices of length npts instead
 * of a single matrix.
 */
void gvecToDetectorXYArray_cfunc(long int npts, double * gVec_c,
                                 double * rMat_d, double * rMat_s,
                                 double * rMat_c, double * tVec_d,
                                 double * tVec_s, double * tVec_c,
                                 double * beamVec, double * result)
{
    long int i;
    int j, k, l;
    double num;
    double nVec_l[3], bHat_l[3], P0_l[3], P3_l[3];
    double rMat_sc[9];

    /* Normalize the beam vector */
    unitRowVector_cfunc(3,beamVec,bHat_l);

    for (i=0L; i < npts; i++) {
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

        gvecToDetectorXYOne_cfunc(gVec_c + 3*i, rMat_d, rMat_sc,
                                  tVec_d, bHat_l, nVec_l, num,
                                  P0_l, result + 2*i);
    }
}

#endif
void gvecToDetectorXY_cfunc(long int npts, double * gVec_c,
                 double * rMat_d, double * rMat_s, double * rMat_c,
                 double * tVec_d, double * tVec_s, double * tVec_c,
                 double * beamVec, double * result)
{
  long int i;
  int j, k, l;

  double num;
  double nVec_l[3], bHat_l[3], P0_l[3], P3_l[3];
  double rMat_sc[9];

  /* Normalize the beam vector */
  unitRowVector_cfunc(3,beamVec,bHat_l);

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
    gvecToDetectorXYOne_cfunc(&gVec_c[3*i], rMat_d, rMat_sc, tVec_d,
                  bHat_l, nVec_l, num,
                  P0_l, &result[2*i]);
  }
}


/*
 * detectorXYToGVec
 * ----------------
 * Two versions, one for an array of points and a single rMat_s, other for
 * array of points and array of rMat_s
 */
static inline void
detectorXYToGVecOne_cfunc(const double *xy, /* source point, just one */
                          const double *rMat_d,
                          const double *rMat_e,
                          const double *tVec1,
                          const double *bVec,
                          double *tTh_out, /* out, scalar */
                          double *eta_out, /* out, scalar */
                          double *gVec_l_out /* out, vector 3 */
                          )
{
    int j, k;
    double nrm, b_dot_dHat_l, tTh, eta, phi;
    double dHat_l[3], tVec2[2], n_g[3];


    /* Compute dHat_l vector */
    nrm = 0.0;
    for (j=0; j<3; j++) {
        double acc = tVec1[j];
        dHat_l[j] = tVec1[j];
        for (k=0; k<2; k++) {
            acc += rMat_d[3*j+k]*xy[k];
        }
        nrm += acc*acc;
        dHat_l[j] = acc;
    }

    if ( nrm > epsf ) {
        double nrm_factor = 1.0/sqrt(nrm);
        for (j=0; j<3; j++) {
            dHat_l[j] *= nrm_factor;
        }
    }

    /* Compute tTh */
    b_dot_dHat_l = 0.0;
    for (j=0; j<3; j++) {
        b_dot_dHat_l += bVec[j]*dHat_l[j];
    }
    tTh = acos(b_dot_dHat_l);

    /* Compute eta */
    for (j=0; j<2; j++) {
        tVec2[j] = 0.0;
        for (k=0; k<3; k++) {
            tVec2[j] += rMat_e[3*k+j]*dHat_l[k];
        }
    }
    eta = atan2(tVec2[1], tVec2[0]);

    /* Compute n_g vector */
    nrm = 0.0;
    for (j=0; j<3; j++) {
        double val;
        int j1 = j < 2 ? j+1 : 0;
        int j2 = j > 0 ? j-1 : 2;
        val = bVec[j1] * dHat_l[j2] - bVec[j2] * dHat_l[j1];
        nrm += val*val;
        n_g[j] = val;
    }
    if ( nrm > epsf ) {
        double nrm_factor = 1.0/sqrt(nrm);
        for (j=0; j<3; j++) {
            n_g[j] *= nrm_factor;
        }
    }

    /* Rotate dHat_l vector */
    phi = 0.5*(M_PI-tTh);
    *tTh_out = tTh;
    *eta_out = eta;
    rotate_vecs_about_axis_cfunc(1, &phi, 1, n_g, 1, dHat_l, gVec_l_out);
}

void detectorXYToGvec_cfunc(long int npts, double * xy,
                double * rMat_d, double * rMat_s,
                double * tVec_d, double * tVec_s, double * tVec_c,
                double * beamVec, double * etaVec,
                double * tTh, double * eta, double * gVec_l)
{
  long int i;
  int j, k;
    double nrm, bVec[3], tVec1[3];
  double rMat_e[9];

  /* Fill rMat_e */
  makeEtaFrameRotMat_cfunc(beamVec,etaVec,rMat_e);

  /* Normalize the beam vector */
  nrm = 0.0;
  for (j=0; j<3; j++) {
    nrm += beamVec[j]*beamVec[j];
  }

  if ( nrm > epsf ) {
        double nrm_factor = 1.0/sqrt(nrm);
    for (j=0; j<3; j++)
            bVec[j] = beamVec[j]*nrm_factor;
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
        detectorXYToGVecOne_cfunc(xy+2*i, rMat_d, rMat_e, tVec1, bVec, tTh + i, eta + i, gVec_l + 3*i);
    }
}

/*
 * In this version, rMat_s is an array
 */
void detectorXYToGvecArray_cfunc(long int npts, double * xy,
                                 double * rMat_d, double * rMat_s,
                                 double * tVec_d, double * tVec_s, double * tVec_c,
                                 double * beamVec, double * etaVec,
                                 double * tTh, double * eta, double * gVec_l)
{
    long int i;
    int j, k;
    double nrm, bVec[3], tVec1[3];
    double rMat_e[9];

    /* Fill rMat_e */
    makeEtaFrameRotMat_cfunc(beamVec, etaVec, rMat_e);

    /* Normalize the beam vector */
    nrm = 0.0;
    for (j=0; j<3; j++) {
        nrm += beamVec[j]*beamVec[j];
      }

    if ( nrm > epsf ) {
        double nrm_factor = 1.0/sqrt(nrm);
        for (j=0; j<3; j++)
            bVec[j] = beamVec[j]*nrm_factor;
    } else {
        for (j=0; j<3; j++)
            bVec[j] = beamVec[j];
      }

    for (j=0; j<3; j++) {
        tVec1[j] = tVec_d[j]-tVec_s[j];
      for (k=0; k<3; k++) {
            tVec1[j] -= rMat_s[3*j+k]*tVec_c[k];
      }
    }

    for (i=0; i<npts; i++) {
        /* Compute shift vector */
    for (j=0; j<3; j++) {
            tVec1[j] = tVec_d[j]-tVec_s[j];
            for (k=0; k<3; k++) {
                tVec1[j] -= rMat_s[3*j+k]*tVec_c[k];
    }
    }
        detectorXYToGVecOne_cfunc(xy+2*i, rMat_d, rMat_e, tVec1, bVec,
                                  tTh + i, eta + i, gVec_l + 3*i);
  }
}

void oscillAnglesOfHKLs_cfunc(long int npts, double * hkls, double chi,
                  double * rMat_c, double * bMat, double wavelength,
                  double * vInv_s, double * beamVec, double * etaVec,
                  double * oangs0, double * oangs1)
{
  long int i;
  int j, k;
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
      makeEtaFrameRotMat_cfunc(bHat_l,eHat_l,rMat_e);

      oVec[0] = chi;

      oVec[1] = oangs0[3L*i+2];
      makeOscillRotMat_cfunc(oVec[0], oVec[1], rMat_s);

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
      makeOscillRotMat_cfunc(oVec[0], oVec[1], rMat_s);

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

/******************************************************************************/
/* Utility Funtions */

void unitRowVector_cfunc(int n, double * cIn, double * cOut)
{
  int j;
  double nrm;

  nrm = 0.0;
  for (j=0; j<n; j++) {
    nrm += cIn[j]*cIn[j];
  }
  nrm = sqrt(nrm);
  if ( nrm > epsf ) {
    for (j=0; j<n; j++) {
      cOut[j] = cIn[j]/nrm;
    }
  } else {
    for (j=0; j<n; j++) {
      cOut[j] = cIn[j];
    }
  }
}

void unitRowVectors_cfunc(int m, int n, double * cIn, double * cOut)
{
  int i,j;
  double nrm;

  for (i=0; i<m; i++) {
    nrm = 0.0;
    for (j=0; j<n; j++) {
      nrm += cIn[n*i+j]*cIn[n*i+j];
    }
    nrm = sqrt(nrm);
    if ( nrm > epsf ) {
      for (j=0; j<n; j++) {
    cOut[n*i+j] = cIn[n*i+j]/nrm;
      }
    } else {
      for (j=0; j<n; j++) {
    cOut[n*i+j] = cIn[n*i+j];
      }
    }
  }
}

void makeDetectorRotMat_cfunc(double * tPtr, double * rPtr)
{
  int i;
  double c[3],s[3];

  for (i=0; i<3; i++) {
    c[i] = cos(tPtr[i]);
    s[i] = sin(tPtr[i]);
  }

  rPtr[0] =  c[1]*c[2];
  rPtr[1] =  s[0]*s[1]*c[2]-c[0]*s[2];
  rPtr[2] =  c[0]*s[1]*c[2]+s[0]*s[2];
  rPtr[3] =  c[1]*s[2];
  rPtr[4] =  s[0]*s[1]*s[2]+c[0]*c[2];
  rPtr[5] =  c[0]*s[1]*s[2]-s[0]*c[2];
  rPtr[6] = -s[1];
  rPtr[7] =  s[0]*c[1];
  rPtr[8] =  c[0]*c[1];
}

void makeOscillRotMat_cfunc(double chi, double ome, double * rPtr)
{
  double c[2],s[2];

  c[0] = cos(chi);
  s[0] = sin(chi);
  c[1] = cos(ome);
  s[1] = sin(ome);

  rPtr[0] =  c[1];
  rPtr[1] =  0.0;
  rPtr[2] =  s[1];
  rPtr[3] =  s[0]*s[1];
  rPtr[4] =  c[0];
  rPtr[5] = -s[0]*c[1];
  rPtr[6] = -c[0]*s[1];
  rPtr[7] =  s[0];
  rPtr[8] =  c[0]*c[1];
}

void makeRotMatOfExpMap_cfunc(double * ePtr, double * rPtr)
{
  int i;
  double c, s, phi;

  for (i=0; i<9; i++) {
    if ( i%4 != 0 )
      rPtr[i] = 0.0;
    else
      rPtr[i] = 1.0;
  }

  phi = sqrt(ePtr[0]*ePtr[0]+ePtr[1]*ePtr[1]+ePtr[2]*ePtr[2]);

  if ( phi > epsf ) {
    s = sin(phi)/phi;
    c = (1.0-cos(phi))/(phi*phi);

    rPtr[1] -= s*ePtr[2];
    rPtr[2] += s*ePtr[1];
    rPtr[3] += s*ePtr[2];
    rPtr[5] -= s*ePtr[0];
    rPtr[6] -= s*ePtr[1];
    rPtr[7] += s*ePtr[0];

    rPtr[1] += c*ePtr[0]*ePtr[1];
    rPtr[2] += c*ePtr[0]*ePtr[2];
    rPtr[3] += c*ePtr[1]*ePtr[0];
    rPtr[5] += c*ePtr[1]*ePtr[2];
    rPtr[6] += c*ePtr[2]*ePtr[0];
    rPtr[7] += c*ePtr[2]*ePtr[1];

    rPtr[0] -= c*(ePtr[1]*ePtr[1]+ePtr[2]*ePtr[2]);
    rPtr[4] -= c*(ePtr[2]*ePtr[2]+ePtr[0]*ePtr[0]);
    rPtr[8] -= c*(ePtr[0]*ePtr[0]+ePtr[1]*ePtr[1]);
  }
}

void makeRotMatOfQuat_cfunc(int nq, double * qPtr, double * rPtr)
{
  int i, j;
  double c, s, phi, n[3]={0.0,0.0,0.0};

  for (i=0; i<nq; i++) {
    phi = 2. * acos(qPtr[4*i+0]);

    if (phi > epsf) {
      n[0] = (1. / sin(0.5*phi)) * qPtr[4*i+1];
      n[1] = (1. / sin(0.5*phi)) * qPtr[4*i+2];
      n[2] = (1. / sin(0.5*phi)) * qPtr[4*i+3];

      s = sin(phi);
      c = cos(phi);

      rPtr[9*i+0] = c + n[0]*n[0]*(1. - c);
      rPtr[9*i+1] = n[0]*n[1]*(1. - c) - n[2]*s;
      rPtr[9*i+2] = n[0]*n[2]*(1. - c) + n[1]*s;
      rPtr[9*i+3] = n[1]*n[0]*(1. - c) + n[2]*s;
      rPtr[9*i+4] = c + n[1]*n[1]*(1. - c);
      rPtr[9*i+5] = n[1]*n[2]*(1. - c) - n[0]*s;
      rPtr[9*i+6] = n[2]*n[0]*(1. - c) - n[1]*s;
      rPtr[9*i+7] = n[2]*n[1]*(1. - c) + n[0]*s;
      rPtr[9*i+8] = c + n[2]*n[2]*(1. - c);
    }
    else {
      for (j=0; j<9; j++) {
    if ( j%4 == 0 )
      rPtr[9*i+j] = 1.0;
    else
      rPtr[9*i+j] = 0.0;
      }
    }
  }
}

void makeBinaryRotMat_cfunc(double * aPtr, double * rPtr)
{
  int i, j;

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      rPtr[3*i+j] = 2.0*aPtr[i]*aPtr[j];
    }
    rPtr[3*i+i] -= 1.0;
  }
}

void makeEtaFrameRotMat_cfunc(double * bPtr, double * ePtr, double * rPtr)
{
  /*
   * This function generates a COB matrix that takes components in BEAM frame to
   * LAB frame
   *
   * NOTE: the beam and eta vectors MUST NOT BE COLINEAR!!!!
   */
  int i;

  double yPtr[3], bHat[3], yHat[3], xHat[3];

  /* find Y as e ^ b */
  yPtr[0] = ePtr[1]*bPtr[2] - bPtr[1]*ePtr[2];
  yPtr[1] = ePtr[2]*bPtr[0] - bPtr[2]*ePtr[0];
  yPtr[2] = ePtr[0]*bPtr[1] - bPtr[0]*ePtr[1];

  /* Normalize beam (-Z) and Y vectors */
  unitRowVector_cfunc(3, bPtr, bHat);
  unitRowVector_cfunc(3, yPtr, yHat);

  /* Find X as b ^ Y */
  xHat[0] = bHat[1]*yHat[2] - yHat[1]*bHat[2];
  xHat[1] = bHat[2]*yHat[0] - yHat[2]*bHat[0];
  xHat[2] = bHat[0]*yHat[1] - yHat[0]*bHat[1];

  /* Assign columns */
  /* Assign Y column */
  for (i=0; i<3; i++)
  {
      rPtr[3*i+0] =  xHat[i];
      rPtr[3*i+1] =  yHat[i];
      rPtr[3*i+2] = -bHat[i];
  }
}

void validateAngleRanges_old_cfunc(int na, double * aPtr, int nr, double * minPtr, double * maxPtr, bool * rPtr)
{
  int i, j;
  double thetaMax, theta;

  /* Each angle should only be examined once. Any more is a waste of time.  */
  for (i=0; i<na; i++) {

    /* Ensure there's no match to begin with */
    rPtr[i] = false;

    for (j=0; j<nr; j++) {

      /* Since the angle values themselves are unimportant we will
     redefine them so that the start of the range is zero.  The
     end of the range will then be between zero and two pi.  It
     will then be quite easy to determine if the angle of interest
     is in the range or not. */

      thetaMax = maxPtr[j] - minPtr[j];
      theta    = aPtr[i] - minPtr[j];

      while ( thetaMax < 0.0 )
    thetaMax += 2.0*M_PI;
      while ( thetaMax > 2.0*M_PI )
    thetaMax -= 2.0*M_PI;

      while ( theta < 0.0 )
    theta += 2.0*M_PI;
      while ( theta > 2.0*M_PI )
    theta -= 2.0*M_PI;

      if ( theta > -sqrt_epsf && theta < thetaMax + sqrt_epsf ) {
    rPtr[i] = true;

    /* No need to check other ranges */
    break;
      }
    }
  }
}

void validateAngleRanges_cfunc(int na, double * aPtr, int nr,
                   double * minPtr, double * maxPtr,
                   bool * rPtr, int ccw)
{
  int i, j;
  double thetaMax, theta;
  double *startPtr, *stopPtr;

  if ( ccw ) {
    startPtr = minPtr;
    stopPtr  = maxPtr;
  } else {
    startPtr = maxPtr;
    stopPtr  = minPtr;
  }

  /* Each angle should only be examined once. Any more is a waste of time.  */
  for (i=0; i<na; i++) {

    /* Ensure there's no match to begin with */
    rPtr[i] = false;

    for (j=0; j<nr; j++) {

      /* Since the angle values themselves are unimportant we will
     redefine them so that the start of the range is zero.  The
     end of the range will then be between zero and two pi.  It
     will then be quite easy to determine if the angle of interest
     is in the range or not. */

      thetaMax = stopPtr[j] - startPtr[j];
      theta    = aPtr[i] - startPtr[j];

      while ( thetaMax < 0.0 )
    thetaMax += 2.0*M_PI;
      while ( thetaMax > 2.0*M_PI )
    thetaMax -= 2.0*M_PI;

      /* Check for an empty range */
      if ( fabs(thetaMax) < sqrt_epsf ) {
    rPtr[i] = true;

    /* No need to check other ranges */
    break;
      }

      /* Check for a range which spans a full circle */
      if ( fabs(thetaMax-2.0*M_PI) < sqrt_epsf ) {

    /* Double check the initial range */
    if ( (ccw && maxPtr[j] > minPtr[j]) || ((!ccw) && maxPtr[j] < minPtr[j]) ) {
      rPtr[i] = true;

      /* No need to check other ranges */
      break;
    }
      }

      while ( theta < 0.0 )
    theta += 2.0*M_PI;
      while ( theta > 2.0*M_PI )
    theta -= 2.0*M_PI;

      if ( theta >= -sqrt_epsf && theta <= thetaMax+sqrt_epsf ) {
    rPtr[i] = true;

    /* No need to check other ranges */
    break;
      }
    }
  }
}


void rotate_vecs_about_axis_cfunc(long int na, double * angles,
                  long int nax, double * axes,
                  long int nv, double * vecs,
                  double * rVecs)
{
  int i, j, sa, sax;
  double c, s, nrm, proj, aCrossV[3];

  if ( na == 1 ) sa = 0;
  else sa = 1;
  if ( nax == 1 ) sax = 0;
  else sax = 3;

  for (i=0; i<nv; i++) {

    /* Rotate using the Rodrigues' Rotation Formula */
    c = cos(angles[sa*i]);
    s = sin(angles[sa*i]);

    /* Compute projection of vec along axis */
    proj = 0.0;
    for (j=0; j<3; j++)
      proj += axes[sax*i+j]*vecs[3*i+j];

    /* Compute norm of axis */
    if ( nax > 1 || i == 0 ) {
      nrm = 0.0;
      for (j=0; j<3; j++)
    nrm += axes[sax*i+j]*axes[sax*i+j];
      nrm = sqrt(nrm);
    }

    /* Compute projection of vec along axis */
    proj = 0.0;
    for (j=0; j<3; j++)
      proj += axes[sax*i+j]*vecs[3*i+j];

    /* Compute the cross product of the axis with vec */
    for (j=0; j<3; j++)
      aCrossV[j] = axes[sax*i+(j+1)%3]*vecs[3*i+(j+2)%3]-axes[sax*i+(j+2)%3]*vecs[3*i+(j+1)%3];

    /* Combine the three terms to compute the rotated vector */
    for (j=0; j<3; j++) {
      rVecs[3*i+j] = c*vecs[3*i+j]+(s/nrm)*aCrossV[j]+(1.0-c)*proj*axes[sax*i+j]/(nrm*nrm);
    }
  }
}

double quat_distance_cfunc(int nsym, double * q1, double * q2, double * qsym)
{
  int i;
  double q0, q0_max = 0.0, dist = 0.0;
  double *q2s;

  if ( NULL == (q2s = (double *)malloc(4*nsym*sizeof(double))) ) {
      printf("malloc failed\n");
      return(-1);
  }

  /* For each symmetry in qsym compute its inner product with q2 */
  for (i=0; i<nsym; i++) {
    q2s[4*i+0] = q2[0]*qsym[4*i+0] - q2[1]*qsym[4*i+1] - q2[2]*qsym[4*i+2] - q2[3]*qsym[4*i+3];
    q2s[4*i+1] = q2[1]*qsym[4*i+0] + q2[0]*qsym[4*i+1] - q2[3]*qsym[4*i+2] + q2[2]*qsym[4*i+3];
    q2s[4*i+2] = q2[2]*qsym[4*i+0] + q2[3]*qsym[4*i+1] + q2[0]*qsym[4*i+2] - q2[1]*qsym[4*i+3];
    q2s[4*i+3] = q2[3]*qsym[4*i+0] - q2[2]*qsym[4*i+1] + q2[1]*qsym[4*i+2] + q2[0]*qsym[4*i+3];
  }

  /* For each symmetric equivalent q2 compute its inner product with inv(q1) */
  for (i=0; i<nsym; i++) {
    q0 = q1[0]*q2s[4*i+0] + q1[1]*q2s[4*i+1] + q1[2]*q2s[4*i+2] + q1[3]*q2s[4*i+3];
    if ( fabs(q0) > q0_max ) {
      q0_max = fabs(q0);
    }
  }

  if ( q0_max <= 1.0 )
    dist = 2.0*acos(q0_max);
  else if ( q0_max - 1. < 1e-12 )
    /* in case of quats loaded from single precision file */
    dist = 0.;
  else
    dist = NAN;

  free(q2s);

  return(dist);
}

void homochoricOfQuat_cfunc(int nq, double * qPtr, double * hPtr)
{
  int i;
  double arg, f, s, phi;

  for (i=0; i<nq; i++) {
    phi = 2. * acos(qPtr[4*i+0]);

    if (phi > epsf) {
      arg = 0.75*(phi - sin(phi));
      if (arg < 0.) {
    f = -pow(-arg, 1./3.);
      } else {
    f = pow(arg, 1./3.);
      }
      s = 1. / sin(0.5*phi);

      hPtr[3*i+0] = f * s * qPtr[4*i+1];
      hPtr[3*i+1] = f * s * qPtr[4*i+2];
      hPtr[3*i+2] = f * s * qPtr[4*i+3];
    }
    else {
      hPtr[3*i+0] = 0.;
      hPtr[3*i+1] = 0.;
      hPtr[3*i+2] = 0.;
    }
  }
}
