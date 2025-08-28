import numpy as np

# Functions to use for reference in testing and the like. These are made to be
# easily testable and clear. They will be use to test against in unit tests.
# They may be slow and not vectorized.


def intersect_ray_plane(ro, rv, p):
    '''
    ray-plane intersection

    Parameters
    ----------
    ro: array[3]
        point of origin of the ray (ray origin).

    rv: array[3]
        direction vector of the ray (ray vector).

    p: array[4]
        (A, B, C, D) coeficients for the plane formula (Ax+By+Cz+D=0).

    Returns
    -------
    t : scalar
        t  where ray intersects with plane, such as (ro + t*rv) lies within p.

    Notes
    -----
    If t is negative, the intersection happens 'behind' the origin point.

    In the case where (A,B,C) -the plane normal- is orthogonal to rv, no
    intersection will happen (or the ray is fully on the plane). In that case,
    t will be non-finite.

    Vector rv needs not to be an unit vector, result will be scaled accordingly
    based on the modulo of rv. However, a 0 vector will result in non-finite
    result.

    The code is based on the segment-plane intersection in  [1]_

    .. [1] Ericson. (2005). Real-Time Collision Detection, pp 176.
           Morgan Kaufmann.
    '''
    assert ro.shape == (3,)
    assert rv.shape == (3,)
    assert p.shape == (4,)

    normal = p[:3]
    D = p[3]
    # disable divide by 0 and invalid as the division in t can cause both.
    # the behavior of the function actually relies in IEEE754 with a division
    # by 0 generating the appropriate infinity, or a NAN if it is a 0/0.
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (D - normal @ ro) / (normal @ rv)

    return t
