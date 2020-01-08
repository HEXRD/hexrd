#ifndef XRD_TRANSFORMS_TYPES_H
#define XRD_TRANSFORMS_TYPES_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>

#ifndef NAN
/* in most cases NAN will be defined in math.h */
static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
#  define NAN (*(const float *) __nan)
#endif

#ifdef _WIN32
  #define false 0
  #define true 1
  #define bool int
#else
  #include <stdbool.h>
#endif

static double epsf = DBL_EPSILON;

/* maybe this shouldn't be needed... */
static double sqrt_epsf = 1.5e-8;
/* this doesn't belong here... really */
static double Zl[3] = { 0.0, 0.0, 1.0};

typedef struct {
    double e[3];
} vector3;

typedef struct {
    double e[9];
} matrix33;


#endif /* XRD_TRANSFORM_TYPES_H */
