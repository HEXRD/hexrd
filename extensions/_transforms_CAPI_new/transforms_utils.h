#ifndef TRANSFORMS_UTILS_H
#define TRANSFORMS_UTILS_H

#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_types.h"
#endif


static double
vector3_dot(const vector3* lhs, const vector3* rhs)
{
    return lhs->e[0]*rhs->e[0] + lhs->e[1]*rhs->e[1] + lhs->e[2]*rhs->e[2];
}

static void
vector3_normalized(const vector3* in, vector3* restrict out)
{
    int i;
    double sqr_norm = vector3_dot(in, in);

    if (sqr_norm > epsf) {
        double recip_norm = 1.0/sqrt(sqr_norm);
        for (i=0; i<3; ++i)
            out->e[i] = in->e[i] * recip_norm;
    } else {
        *out = *in;
        /*        for (i=0; i<3; ++i)
                  out[i] = in[i];*/
    }
}

static
void matrix33_set_identity(matrix33* restrict out)
{
    int i;
    for (i = 0; i < 9; ++i)
        out->e[i] = (i%4 == 0)? 1.0: 0.0;
}


#endif /* TRANSFORMS_UTILS_H */

