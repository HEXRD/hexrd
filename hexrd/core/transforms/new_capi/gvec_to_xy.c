
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#  include "ndargs_helper.h"
#endif

#define GV2XY_GROUP_SIZE 128
#define GVEC_TO_XY_FUNC gvec_to_xy_vect

struct planar_detector_struct {
    double plane[4];
    double x[4];
    double y[4];
};

/*
  computes the diffraction vector for the G vector ```gvec```
  Assumes the inciding beam vector is the Z unit vector {0,0,1}
*/
static inline
void
diffract_z(const double *gvec, double * restrict diffracted)
{
    diffracted[0] = 2*gvec[0]*gvec[2];
    diffracted[1] = 2*gvec[1]*gvec[2];
    diffracted[2] = 2*gvec[2]*gvec[2] - 1.0;
}

static inline
void
diffract(const double *beam, const double *vec, double * restrict diffracted)
{
    double bm02 = vec[0]*vec[2]*2.0;
    double bm12 = vec[1]*vec[2]*2.0;
    double bm22 = vec[2]*vec[2]*2.0 - 1.0;

    /* if beam is not null, we use a beam other than the trivial one {0,0,1} */
    if (!beam)
    { /* use beam {0, 0, 1} */
        diffracted[0] = bm02;
        diffracted[1] = bm12;
        diffracted[2] = bm22;
    }
    else
    {
        double bm00 = vec[0]*vec[0]*2.0 - 1.0;
        double bm11 = vec[1]*vec[1]*2.0 - 1.0;
        double bm01 = vec[0]*vec[1]*2.0;
        double b0 = beam[0];
        double b1 = beam[1];
        double b2 = beam[2];

        diffracted[0] = bm02*b2 + bm01*b1 + bm00*b0;
        diffracted[1] = bm12*b2 + bm11*b1 + bm01*b0;
        diffracted[2] = bm22*b2 + bm12*b1 + bm02*b0;
    }
}

/*
  compute diffraction for <count> G vectors for a <beam> direction. Place
  diffracted vectors in <diffracted>. If <beam> is NULL, use { 0, 0, 1 } as
  beam.
 */
static void
diffract_array(const double *beam, const double *vectors,
               double * restrict diffracted, size_t count)
{
    size_t i;

    if (beam)
    {
        double b0 = beam[0];
        double b1 = beam[1];
        double b2 = beam[2];
        for (i = 0; i < count; i++)
        {
            double v0 = vectors[3*i+0];
            double v1 = vectors[3*i+1];
            double v2 = vectors[3*i+2];
            double bm00 = v0*v0*2.0 - 1.0;
            double bm11 = v1*v1*2.0 - 1.0;
            double bm22 = v2*v2*2.0 - 1.0;
            double bm01 = v0*v1*2.0;
            double bm02 = v0*v2*2.0;
            double bm12 = v1*v2*2.0;

            diffracted[3*i+0] = bm02*b2 + bm01*b1 + bm00*b0;
            diffracted[3*i+1] = bm12*b2 + bm11*b1 + bm01*b0;
            diffracted[3*i+2] = bm22*b2 + bm12*b1 + bm02*b0;
        }

    }
    else
    {
        for (i = 0; i < count; i++)
        {
            double v0 = vectors[3*i+0];
            double v1 = vectors[3*i+1];
            double v2 = vectors[3*i+2];
            double bm02 = v0*v2*2.0;
            double bm12 = v1*v2*2.0;
            double bm22 = v2*v2*2.0 - 1.0;

            diffracted[3*i+0] = bm02;
            diffracted[3*i+1] = bm12;
            diffracted[3*i+2] = bm22;
        }
    }
}

/*
  rotate <count> 3-vectors pointed by <vectors> by <rot_mat>, place results
  in <rotated>.
  rotate_vector_array_v: vectors are vectorized, rot_mat remains constant.
  rotate_vector_array_rv: vectors and rot_mat are vectorized.
*/
static void
rotate_vector_array_v(const double *rot_mat, const double *vectors,
                    double * restrict rotated, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
    {
        m33_v3s_multiply(rot_mat, vectors + 3*i, 1, rotated + 3*i);
    }
}

static void
rotate_vector_array_rv(const double *rot_mat, const double *vectors,
                    double * restrict rotated, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
    {
        m33_v3s_multiply(rot_mat+9*i, vectors + 3*i, 1, rotated + 3*i);
    }
}


/*
  apply a full transform of <rotation> and <translation> to <vector>, placing
  results in <transformed>.
  This particular routine does <count> transforms, where <rotation> changes for
  each iteration. <translation> and <vector> remain constant.
 */
static void
transform_vector_array_r(const double *translation, const double *rotation,
                         const double *vectors, double * restrict transformed,
                         size_t count)
{
    size_t i;
    double t0 = translation[0], t1 = translation[1], t2 = translation[2];
    double v0 = vectors[0], v1 = vectors[1], v2 = vectors[2];
    for (i = 0; i < count; i++)
    {
        double r00 = rotation[0+9*i], r01 = rotation[1+9*i], r02 = rotation[2+9*i];
        double r10 = rotation[3+9*i], r11 = rotation[4+9*i], r12 = rotation[5+9*i];
        double r20 = rotation[6+9*i], r21 = rotation[7+9*i], r22 = rotation[8+9*i];

        transformed[3*i+0] = t0 + r00*v0 + r01*v1 + r02*v2;
        transformed[3*i+1] = t1 + r10*v0 + r11*v1 + r12*v2;
        transformed[3*i+2] = t2 + r20*v0 + r21*v1 + r22*v2;
    }
}

static void
vdot_array_v(const double *vector, const double *vector_array,
             double * restrict results, size_t count)
{
    size_t i;
    double v0 = vector[0], v1 = vector[1], v2 = vector[2];

    for (i = 0; i < count; i++)
    {
        double va0 = vector_array[3*i+0], va1 = vector_array[3*i+1], va2 = vector_array[3*i+2];
        results[i] = v0*va0 + v1*va1 + v2*va2;
    }
}

/*
  compute result = base - factor*vectors, for <count> iterations.

  in the svv version:
  <base>  remains constant.
  <factor> one entry per iteration of double
  <vector> one entry per iteration of double
*/
static void
submul_array_svv(const double *base, const double *factor, const double *vector,
                 double * restrict results, size_t count)
{
    size_t i;
    double b = *base;

    for (i = 0; i<count; i++)
    {
        double t = factor[i], x = vector[i];
        results[i] = b - t*x;
    }
}

/*
  given a numerator and denominator for the ray-plane intersection, compute the
  resulting factors <t> that solve the ray-plane intersection for <count>
  elements.

  It the factor <t> is negative, the resulting ray will never collide, so return
  a NAN.

  In the _sv version of the function, a single numerator is provided and <count>
  denominators are present.
 */
static void
ray_factor_array_sv(const double *num, const double *denom,
                   double * restrict result, size_t count)
{
    size_t i;
    double nume = *num;
    for (i = 0; i < count; i++)
    {
        double tmp = nume/denom[i];
        result[i] = tmp < 0.0 ? tmp : NAN;
    }
}

static void
project_on_planar_detector_vsv(const struct planar_detector_struct *pd_data,
                               const double *ray_origin,
                               const double *ray_vector,
                               double * restrict results,
                               size_t count)
{
    /* note: this is written in a way that that compilers seem to do a nice
       job using vectorization.
       Operands are placed into local variables. They are only loaded from
       memory once when they enter in "scope". This seems to make compilers
       more eager to vectorize.
    */
    size_t i;
    double orig0 = ray_origin[0], orig1 = ray_origin[1], orig2 = ray_origin[2];
    const double *x = pd_data->x, *y = pd_data->y, *plane = pd_data->plane;
    double plane0 = plane[0], plane1 = plane[1], plane2 = plane[2], plane3 = plane[3];
    double x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
    double y0 = y[0], y1 = y[1], y2 = y[2], y3 = y[3];
    double x_base = x3 + x0*orig0 + x1*orig1 + x2*orig2;
    double y_base = y3 + y0*orig0 + y1*orig1 + y2*orig2;
    double num = plane3 + plane0*orig0 + plane1*orig1 + plane2*orig2;

    for (i=0; i<count; i++)
    {
        double vect0 = ray_vector[i*3+0], vect1 = ray_vector[i*3+1], vect2 = ray_vector[i*3+2];
        double denom = plane0*vect0 + plane1*vect1 + plane2*vect2;
        double x_vect = x0*vect0 + x1*vect1 + x2*vect2;
        double y_vect = y0*vect0 + y1*vect1 + y2*vect2;
        double t = num/denom;
        double factor = t < 0.0 ? t : NAN;

        results[2*i+0] = x_base - factor*x_vect;
        results[2*i+1] = y_base - factor*y_vect;
    }
}

static void
project_on_planar_detector_vvv(const struct planar_detector_struct *pd_data,
                               const double *ray_origin,
                               const double *ray_vector,
                               double * restrict results,
                               size_t count)
{
    size_t i;
    const double *x = pd_data->x, *y = pd_data->y, *plane = pd_data->plane;
    double plane0 = plane[0], plane1 = plane[1], plane2 = plane[2], plane3 = plane[3];
    double x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
    double y0 = y[0], y1 = y[1], y2 = y[2], y3 = y[3];


    for (i=0; i<count; i++)
    {
        double orig0 = ray_origin[i*3+0], orig1 = ray_origin[i*3+1], orig2 = ray_origin[i*3+2];
        double vect0 = ray_vector[i*3+0], vect1 = ray_vector[i*3+1], vect2 = ray_vector[i*3+2];
        double num = plane3 + plane0*orig0 + plane1*orig1 + plane2*orig2;
        double x_base = x3 + x0*orig0 + x1*orig1 + x2*orig2;
        double y_base = y3 + y0*orig0 + y1*orig1 + y2*orig2;
        double denom = plane0*vect0 + plane1*vect1 + plane2*vect2;
        double x_vect = x0*vect0 + x1*vect1 + x2*vect2;
        double y_vect = y0*vect0 + y1*vect1 + y2*vect2;
        double t = num/denom;
        double factor = t < 0.0 ? t : NAN;

        results[2*i+0] = x_base - factor*x_vect;
        results[2*i+1] = y_base - factor*y_vect;
    }
}

/*
  returns N, where N solves N*vect+origin laying in the plane defined by plane.

  origin -> point where the ray starts.
  vect -> direction vector for the ray.
  plane -> vector[4] for A B C D in the plane formula Ax + By + Cz + D = 0. Note
           this means that (A, B, C) is the plane normal.

  returns 1 on intersection, 0 if there is no intersection. In case of
  intersection, the intersecting point will be filled in
  collision_point. Otherwise collision_point will be left untouched.

  Note this code does not handle division by 0. on IEEE-754 arithmetic t may
  become an infinity in that case. The result would be infinity in the resulting
  collision point. As the result is going to be clipped at some point, this
  should not be a problem.
 */
static inline
int
ray_plane_intersect(const double *origin, const double *vect,
                    const double *plane, double * restrict collision_point)
{
    double t, num, denom;
    num = plane[3] - v3_v3s_dot(plane, origin, 1);
    denom = v3_v3s_dot(plane, vect, 1);
    t = num / denom;
    if (t < 0.0)
        return 0;
    v3s_s_v3_muladd(vect, 1, t, origin, collision_point);
    return 1;
}


static void
gvec_to_xy(size_t npts, const double *gVec_cs,
           const double *rMat_d, const double *rMat_ss, const double *rMat_c,
           const double *tVec_d, const double *tVec_s, const double *tVec_c,
           const double *beamVec,
           double * restrict result, unsigned int flags)
{
    size_t i;
    double plane[4], tVec_sc[3], tVec_ds[3];
    double ray_origin[3];
    double rMat_sc[9];
    int use_single_rMat_s = flags & GV2XY_SINGLE_RMAT_S;

    /*
      compute detector normal in LAB (nVec_l)
       The normal will just be the Z column vector of rMat_d
     */
    v3s_copy(rMat_d + 2, 3, plane);
    plane[3] = 0.0;

    /* tVec_ds is the translation from DETECTOR to SAMPLE in LAB frame*/
    v3_v3s_sub(tVec_s, tVec_d, 1, tVec_ds);

    /*
       This loop generates the ray, checks collision with detector plane,
       obtains the collision point and projects it into DETECTOR frame.
       Collision detection is made in LAB coordinates, but centered around
       the detector position.

       Ray ORIGIN will be the input "tVec_c" in LAB frame centered on the
       detector.

       Ray VECTOR will be the diffracted direction based on the associated
       gVec_c and the beam vector (hardcoded to z unit vector {0,0,1}) for
       speed.

       Note that some code inside the loop for the single rMat_s and the
       multiple rMat_s is shared, however, having the "if" outside the loop
       seems to improve performance a bit.
    */

    if (use_single_rMat_s) {
        const double *rMat_s = rMat_ss; /* only one */
        /* All gvecs use the same rMat_s */
        /* ray origin can be already computed */
        m33_v3s_multiply(rMat_s, tVec_c, 1, tVec_sc);
        v3_v3s_add(tVec_ds, tVec_sc, 1, ray_origin);

        /* precompute rMat_sc. */
        m33_m33_multiply(rMat_s, rMat_c, rMat_sc);

        for (i=0L; i < npts; i++) {
            double gVec_l[3], ray_vector[3], point[3];
            const double *gVec_c = gVec_cs + 3*i;

            m33_v3s_multiply(rMat_sc, gVec_c, 1, gVec_l);
            diffract(beamVec, gVec_l, ray_vector);
            if (ray_plane_intersect(ray_origin, ray_vector, plane, point))
            { /* intersect: project the result (only x, y -> z is always 0.0) */
                result[2*i] = v3_v3s_dot(point, rMat_d, 3);
                result[2*i+1] = v3_v3s_dot(point, rMat_d+1, 3);
            }
            else
            { /* no intersection, NAN the result */
                result[2*i] = NAN;
                result[2*i + 1] = NAN;
            }
        }
    }
    else
    {
        for (i=0L; i<npts; i++) {
            double gVec_sam[3], gVec_lab[3], ray_vector[3], point[3];
            const double *gVec_cry = gVec_cs + 3*i;
            const double *rMat_s = rMat_ss + 9*i;

            /*
              tVec_dc <= tVec_s + rMat_s x tVec_c
              tVec_sc is transform from SAMPLE to CRYSTAL
              tVec_dc is transform from DETECTOR to CRYSTAL
            */
            m33_v3s_multiply(rMat_s, tVec_c, 1, tVec_sc);
            v3_v3s_add(tVec_ds, tVec_sc, 1, ray_origin);

            /* Compute gVec in lab frame */
            m33_v3s_multiply(rMat_c, gVec_cry, 1, gVec_sam); /* gVec in SAMPLE frame */
            m33_v3s_multiply(rMat_s, gVec_sam, 1, gVec_lab); /* gVec in LAB frame */
            diffract(beamVec, gVec_lab, ray_vector);
            if (0 == ray_plane_intersect(ray_origin, ray_vector, plane, point))
            {
                /* no intersection */
                result[2*i] = NAN;
                result[2*i + 1] = NAN;
                continue;
            }
            /*
              project into DETECTOR coordinates. Only ```x``` and ```y``` as
              ```z``` will always be 0.0
            */
            result[2*i] = v3_v3s_dot(point, rMat_d, 3);
            result[2*i+1] = v3_v3s_dot(point, rMat_d+1, 3);

        }
    }
}

static void
rays_to_planar_detector (const void *detector_data,
                         const double *ray_origin, size_t ray_origin_count,
                         const double *ray_vector, size_t ray_vector_count,
                         double * restrict projected_xy, size_t projected_xy_count)

{
    const struct planar_detector_struct *pd = (struct planar_detector_struct *)detector_data;

    if (ray_origin_count == 1)
    {
        project_on_planar_detector_vsv(pd, ray_origin, ray_vector, projected_xy,
                                       ray_vector_count);
    }
    else
    {
        project_on_planar_detector_vvv(pd, ray_origin, ray_vector, projected_xy,
                                       ray_vector_count);
    }
}

typedef void (*rays_to_detector_func) (const void *detector_data,
                                       const double *ray_origin, size_t ray_origin_count,
                                       const double *ray_vector, size_t ray_vector_count,
                                       double * restrict projected_xy, size_t projected_xy_count);
/* experimental:
 */
static void
gvec_to_xy_detector(size_t npts, const double *gVec_cs,
                    const double *rMat_ss, const double *rMat_c,
                    const double *tVec_s, const double *tVec_c,
                    const double *beamVec,
                    double * restrict result, unsigned int flags,
                    rays_to_detector_func to_detector, const void *to_detector_data)
{
    size_t i;
    int use_single_rMat_s = flags & GV2XY_SINGLE_RMAT_S;

    /*
       This loop generates the ray, checks collision with detector plane,
       obtains the collision point and projects it into DETECTOR frame.
       Collision detection is made in LAB coordinates, but centered around
       the detector position.

       Ray ORIGIN will be the input "tVec_c" in LAB frame centered on the
       detector.

       Ray VECTOR will be the diffracted direction based on the associated
       gVec_c and the beam vector (hardcoded to z unit vector {0,0,1}) for
       speed.

       Note that some code inside the loop for the single rMat_s and the
       multiple rMat_s is shared, however, having the "if" outside the loop
       seems to improve performance a bit.
    */

    if (use_single_rMat_s)
    {
        #define INNER_SIZE GV2XY_GROUP_SIZE
        double ray_origin[3]; /* only one origin if rMat_s is constant */
        double rMat_sc[9];
        const double *rMat_s = rMat_ss; /* only one */
        double ray_vector[INNER_SIZE*3];
        double gVec_l[INNER_SIZE*3];

        /* compute ray origin at LAB frame */
        transform_vector_array_r(tVec_s, rMat_s, tVec_c, ray_origin, 1);
        /*        m33_v3s_multiply(rMat_s, tVec_c, 1, tVec_sc);
                  v3_v3s_add(tVec_s, tVec_sc, 1, ray_origin);
        */

        /* loop full... */
        m33_m33_multiply(rMat_s, rMat_c, rMat_sc);
        for (i = 0; i + INNER_SIZE - 1 < npts; i += INNER_SIZE)
        {
            rotate_vector_array_v(rMat_sc, gVec_cs + 3*i, gVec_l, INNER_SIZE);
            diffract_array(beamVec, gVec_l, ray_vector, INNER_SIZE);
            to_detector(to_detector_data,
                        ray_origin, 1,
                        ray_vector, INNER_SIZE,
                        result + 2*i, INNER_SIZE);
        }

        if (i < npts)
        {
            size_t missing = npts-i;
            rotate_vector_array_v(rMat_sc, gVec_cs + 3*i, gVec_l, missing);
            diffract_array(beamVec, gVec_l, ray_vector, missing);
            to_detector(to_detector_data,
                        ray_origin, 1,
                        ray_vector, missing,
                        result + 2*i, missing);
        }
        #undef INNER_SIZE
    }
    else
    {
        #define INNER_SIZE GV2XY_GROUP_SIZE
        double ray_origin[INNER_SIZE*3], ray_vector[INNER_SIZE*3];
        double gVec_sam[INNER_SIZE*3], gVec_lab[INNER_SIZE*3];

        for (i = 0; i + INNER_SIZE - 1 < npts; i += INNER_SIZE)
        {
            const double *rMat_s = rMat_ss + 9*i;
            const double *gVec_c = gVec_cs + 3*i;
            /* compute ray origins */
            transform_vector_array_r(tVec_s, rMat_s, tVec_c, ray_origin, INNER_SIZE);

            /* compute ray vectors */
            rotate_vector_array_v(rMat_c, gVec_c, gVec_sam, INNER_SIZE);
            rotate_vector_array_rv(rMat_s, gVec_sam, gVec_lab, INNER_SIZE);
            diffract_array(beamVec, gVec_lab, ray_vector, INNER_SIZE);
            to_detector(to_detector_data,
                        ray_origin, INNER_SIZE,
                        ray_vector, INNER_SIZE,
                        result + 2*i, INNER_SIZE);
        }

        if (i < npts)
        {
            const double *rMat_s = rMat_ss + 9*i;
            const double *gVec_c = gVec_cs + 3*i;
            size_t missing = npts-i;
            /* compute ray origins */
            transform_vector_array_r(tVec_s, rMat_s, tVec_c, ray_origin, missing);

            /* compute ray vectors */
            rotate_vector_array_v(rMat_c, gVec_c, gVec_sam, missing);
            rotate_vector_array_rv(rMat_s, gVec_sam, gVec_lab, missing);
            diffract_array(beamVec, gVec_lab, ray_vector, missing);
            to_detector(to_detector_data,
                        ray_origin, missing,
                        ray_vector, missing,
                        result + 2*i, missing);
        }
        #undef INNER_SIZE
    }
}

static void
gvec_to_xy_vect(size_t npts, const double *gVec_cs,
                const double *rMat_d, const double *rMat_ss, const double *rMat_c,
                const double *tVec_d, const double *tVec_s, const double *tVec_c,
                const double *beam,
                double * restrict result, unsigned int flags)
{
    struct planar_detector_struct planar_detector_data;

    /* build the planar detector struct */
    v3s_copy(rMat_d + 2, 3, planar_detector_data.plane);
    planar_detector_data.plane[3] = -v3_v3s_dot(tVec_d, rMat_d + 2, 3);
    v3s_copy(rMat_d + 0, 3, planar_detector_data.x);
    planar_detector_data.x[3] = -v3_v3s_dot(tVec_d, rMat_d + 0, 3);
    v3s_copy(rMat_d + 1, 3, planar_detector_data.y);
    planar_detector_data.y[3] = -v3_v3s_dot(tVec_d, rMat_d + 1, 3);

    gvec_to_xy_detector(npts, gVec_cs, rMat_ss, rMat_c, tVec_s, tVec_c, beam,
                        result, flags, rays_to_planar_detector, &planar_detector_data);
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


/* normalize the beam vector.
 * in is the beam vector input (either null or a 3 element vector)
 * work is a work buffer where the result is to be stored.
 * if the normalized beam vector results in a Z vector, NULL is returned.
 */
static inline double *
normalize_beam(double *in, double *work)
{
    double beam_sqrnorm;
    if (NULL == in)
        return NULL;

    beam_sqrnorm = v3_v3s_dot(in, in, 1);
    if (beam_sqrnorm > (epsf*epsf))
    {
        double beam_z;
        double beam_rnorm = -1.0/sqrt(beam_sqrnorm);
        beam_z = in[2] * beam_rnorm;
        if (fabs(beam_z - 1.0) < epsf)
        {
            /* consider that beam is {0, 0, 1}, activate fast path by using a
               NULL beam */
            return NULL;
        }
        else
        {
            /* normalize the beam vector in full... */
            work[0] = in[0] * beam_rnorm;
            work[1] = in[1] * beam_rnorm;
            work[2] = beam_z;

            return work;
        }
    }
    /* this case maybe should result in an error. Is is basically getting a
     * 0 beam vector
     */
    return in;
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
XRD_PYTHON_WRAPPER PyObject *
python_gvecToDetectorXY(PyObject * self, PyObject * args)
{
    nah_array gVec_c = { NULL, "gvec_c", NAH_TYPE_DP_FP, { 3, NAH_DIM_ANY }};
    nah_array rMat_d = { NULL, "rmat_d", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array rMat_s = { NULL, "rmat_s", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array rMat_c = { NULL, "rmat_c", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array tVec_d = { NULL, "tvec_d", NAH_TYPE_DP_FP, { 3 }};
    nah_array tVec_s = { NULL, "tvec_s", NAH_TYPE_DP_FP, { 3 }};
    nah_array tVec_c = { NULL, "tvec_c", NAH_TYPE_DP_FP, { 3 }};
    nah_array beamVec = { NULL, "beam_vec", NAH_TYPE_NONE | NAH_TYPE_DP_FP, { 3 }};
    PyArrayObject *result;

    npy_intp npts, dims[2];
    double beam[3], *beam_ptr;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"O&O&O&O&O&O&O&|O&",
                          nah_array_converter, &gVec_c,
                          nah_array_converter, &rMat_d,
                          nah_array_converter, &rMat_s,
                          nah_array_converter, &rMat_c,
                          nah_array_converter, &tVec_d,
                          nah_array_converter, &tVec_s,
                          nah_array_converter, &tVec_c,
                          nah_array_converter, &beamVec))
        return NULL;

    /* number of gvectors comes from gvec_c outer dimension */
    npts = PyArray_DIM(gVec_c.pyarray, 0);

    /* Allocate C-style array for return data */
    dims[0] = npts;
    dims[1] = 2;
    result = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);
    if (!result)
        goto fail_alloc;

    if (beamVec.pyarray)
        beam_ptr = normalize_beam((double *)PyArray_DATA(beamVec.pyarray), beam);
    else
        beam_ptr = NULL;

    /* Call the computational routine */
    GVEC_TO_XY_FUNC(npts,
                    (double *)PyArray_DATA(gVec_c.pyarray),
                    (double *)PyArray_DATA(rMat_d.pyarray),
                    (double *)PyArray_DATA(rMat_s.pyarray),
                    (double *)PyArray_DATA(rMat_c.pyarray),
                    (double *)PyArray_DATA(tVec_d.pyarray),
                    (double *)PyArray_DATA(tVec_s.pyarray),
                    (double *)PyArray_DATA(tVec_c.pyarray),
                    beam_ptr,
                    (double *)PyArray_DATA(result),
                    GV2XY_SINGLE_RMAT_S);

    /* Build and return the nested data structure */
    return (PyObject *)result;

 fail_alloc:
    Py_XDECREF(result);

    return PyErr_NoMemory();
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
    nah_array gVec_c = { NULL, "gVec_c", NAH_TYPE_DP_FP, { 3, NAH_DIM_ANY }};
    nah_array rMat_d = { NULL, "rMat_d", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array rMat_s = { NULL, "rMat_s", NAH_TYPE_DP_FP, { 3, 3, NAH_DIM_ANY }};
    nah_array rMat_c = { NULL, "rMat_c", NAH_TYPE_DP_FP, { 3, 3 }};
    nah_array tVec_d = { NULL, "tVec_d", NAH_TYPE_DP_FP, { 3 }};
    nah_array tVec_s = { NULL, "tVec_s", NAH_TYPE_DP_FP, { 3 }};
    nah_array tVec_c = { NULL, "tVec_c", NAH_TYPE_DP_FP, { 3 }};
    nah_array beamVec = { NULL, "beamVec", NAH_TYPE_NONE | NAH_TYPE_DP_FP, { 3 }};
    PyArrayObject *result;
    npy_intp npts, dims[2];
    double beam[3], *beam_ptr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"O&O&O&O&O&O&O&|O&",
                           nah_array_converter, &gVec_c,
                           nah_array_converter, &rMat_d,
                           nah_array_converter, &rMat_s,
                           nah_array_converter, &rMat_c,
                           nah_array_converter, &tVec_d,
                           nah_array_converter, &tVec_s,
                           nah_array_converter, &tVec_c,
                           nah_array_converter, &beamVec))
        return(NULL);

    /* Verify dimensions of input arrays */
    npts = PyArray_DIM(gVec_c.pyarray, 0);

    if (npts != PyArray_DIM(rMat_s.pyarray, 0))
    {
        PyErr_Format(PyExc_ValueError,
                     "gVec_c and rMat_s length mismatch %zd vs %zd",
                     PyArray_DIM(gVec_c.pyarray, 0),
                     PyArray_DIM(rMat_s.pyarray, 0));
        return NULL;
    }

    /* Allocate C-style array for return data */
    dims[0] = npts;
    dims[1] = 2;
    result = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);
    if (!result)
        goto fail_alloc;

    if (beamVec.pyarray)
        beam_ptr = normalize_beam((double *)PyArray_DATA(beamVec.pyarray), beam);
    else
        beam_ptr = NULL;

    /* Call the computational routine */
    GVEC_TO_XY_FUNC(npts,
                    (double *)PyArray_DATA(gVec_c.pyarray),
                    (double *)PyArray_DATA(rMat_d.pyarray),
                    (double *)PyArray_DATA(rMat_s.pyarray),
                    (double *)PyArray_DATA(rMat_c.pyarray),
                    (double *)PyArray_DATA(tVec_d.pyarray),
                    (double *)PyArray_DATA(tVec_s.pyarray),
                    (double *)PyArray_DATA(tVec_c.pyarray),
                    beam_ptr,
                    (double *)PyArray_DATA(result),
                    0);

    /* Build and return the nested data structure */
    return (PyObject *)result;

 fail_alloc:
    Py_XDECREF(result);

    return PyErr_NoMemory();
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
