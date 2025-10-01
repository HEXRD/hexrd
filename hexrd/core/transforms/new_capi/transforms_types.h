#ifndef XRD_TRANSFORMS_TYPES_H
#define XRD_TRANSFORMS_TYPES_H

#define _USE_MATH_DEFINES
#include <stddef.h>
#include <math.h>
#include <float.h>

#ifndef NAN
/* in most cases NAN will be defined in math.h */
/* note this relies on being little endian */
static const unsigned long __nan[2] = {0, 0x7ff80000};
#  define NAN (*(const float *) __nan)
#endif

#if defined(__STDC_VERSION__) || __STDC_VERSION__ >= 199901L
/* C99 is supported */
#  include <stdbool.h>
#else
#  define false 0
#  define true 1
#  define bool char

#  if defined(_MSC_VER)
#    define inline __inline
#  endif

#  define restrict
#endif

static double epsf = DBL_EPSILON;

/* maybe this shouldn't be needed... */
static double sqrt_epsf = 1.5e-8;

/* this doesn't belong here... really
   Seems the hardcoded Z vector for the lab frame. Maybe shouldn't even exist */
static double Zl[3] = { 0.0, 0.0, 1.0};

#endif /* XRD_TRANSFORM_TYPES_H */
