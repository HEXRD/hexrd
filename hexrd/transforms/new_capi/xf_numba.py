#! /usr/bin/env python
# =============================================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# =============================================================================
# ??? do we want to set np.seterr(invalid='ignore') to avoid nan warnings?
# -*- coding: utf-8 -*-

"""Tranforms module implementation using numba.

Currently, this implementation contains code for the following functions:

- angles_to_gvec
- angles_to_dvec

- row_norm
- unit_vector
- make_rmat_of_expmap
- make_beam_rmat
"""
import numpy as np
from numpy import float_ as npfloat
from numpy import int_ as npint

from . import constants as cnst
from .transforms_definitions import xf_api, get_signature
from .xf_numpy import _beam_to_crystal

try:
    import numba
except ImportError:
    # Numba is an optional dependency. Any code relying on numba should be
    # optional
    raise ImportError("xf_numba not available: numba not installed")

# Use the following decorator instead of numba.jit for interface functions.
# This is so we can patch certain features.
def xfapi_jit(fn):
    out = numba.jit(fn)
    out.__signature__ = get_signature(fn)

    return out


@numba.njit
def _angles_to_gvec_helper(angs, out=None):
    """
    angs are vstacked [2*theta, eta, omega], although omega is optional

    This should be equivalent to the one-liner numpy version:
    out = np.vstack([[np.cos(0.5*angs[:, 0]) * np.cos(angs[:, 1])],
                     [np.cos(0.5*angs[:, 0]) * np.sin(angs[:, 1])],
                     [np.sin(0.5*angs[:, 0])]])

    although much faster
    """
    count, dim = angs.shape
    out = out if out is not None else np.empty((count, 3), dtype=angs.dtype)

    for i in range(count):
        ca0 = np.cos(0.5*angs[i, 0])
        sa0 = np.sin(0.5*angs[i, 0])
        ca1 = np.cos(angs[i, 1])
        sa1 = np.sin(angs[i, 1])
        out[i, 0] = ca0 * ca1
        out[i, 1] = ca0 * sa1
        out[i, 2] = sa0

    return out


@numba.njit
def _angles_to_dvec_helper(angs, out=None):
    """
    angs are vstacked [2*theta, eta, omega], although omega is optional

    This shoud be equivalent to the one-liner numpy version:
    out = np.vstack([[np.sin(angs[:, 0]) * np.cos(angs[:, 1])],
                     [np.sin(angs[:, 0]) * np.sin(angs[:, 1])],
                     [-np.cos(angs[:, 0])]])

    although much faster
    """
    _, dim = angs.shape
    out = out if out is not None else np.empty((dim, 3), dtype=angs.dtype)
    for i in range(len(angs)):
        ca0 = np.cos(angs[i, 0])
        sa0 = np.sin(angs[i, 0])
        ca1 = np.cos(angs[i, 1])
        sa1 = np.sin(angs[i, 1])
        out[i, 0] = sa0 * ca1
        out[i, 1] = sa0 * sa1
        out[i, 2] = -ca0

    return out

@numba.njit
def _rmat_s_helper(chi=None, omes=None, out=None):
    """
    simple utility for calculating sample rotation matrices based on
    standard definition for HEDM

    chi is a single value, 0.0 by default
    omes is either a 1d array or None.
         If None the code should be equivalent to a single ome of value 0.0

    out is a preallocated output array. No check is done about it having the
        proper size. If None a new array will be allocated. The expected size
        of the array is as many 3x3 matrices as omes (n, 3, 3).
    """
    if chi is not None:
        cx = np.cos(chi)
        sx = np.sin(chi)
    else:
        cx = 1.0
        sx = 0.0

    if omes is not None:
        # omes is an array (vector): output is as many rotation matrices as omes entries.
        n = len(omes)
        out = out if out is not None else np.empty((n,3,3), dtype=omes.dtype)

        if chi is not None:
            # ome is array and chi is a value... compute output
            cx = np.cos(chi)
            sx = np.sin(chi)
            for i in range(n):
                cw = np.cos(omes[i])
                sw = np.sin(omes[i])
                out[i, 0, 0] =     cw;  out[i, 0, 1] = 0.;  out[i, 0, 2] =     sw
                out[i, 1, 0] =  sx*sw;  out[i, 1, 1] = cx;  out[i, 1, 2] = -sx*cw
                out[i, 2, 0] = -cx*sw;  out[i, 2, 1] = sx;  out[i, 2, 2] =  cx*cw
        else:
            # omes is array and chi is None -> equivalent to chi=0.0, but shortcut computations.
            # cx IS 1.0, sx IS 0.0
            for i in range(n):
                cw = np.cos(omes[i])
                sw = np.sin(omes[i])
                out[i, 0, 0] =  cw;  out[i, 0, 1] = 0.;  out[i, 0, 2] = sw
                out[i, 1, 0] =  0.;  out[i, 1, 1] = 1.;  out[i, 1, 2] = 0.
                out[i, 2, 0] = -sw;  out[i, 2, 1] = 0.;  out[i, 2, 2] = cw
    else:
        # omes is None, results should be equivalent to an array with a single element 0.0
        out = out if out is not None else np.empty((1, 3, 3))
        if chi is not None:
            # ome is 0.0. cw is 1.0 and sw is 0.0
            cx = np.cos(chi)
            sx = np.sin(chi)
            out[0, 0, 0] = 1.;  out[0, 0, 1] = 0.;  out[0, 0, 2] =  0.
            out[0, 1, 0] = 0.;  out[0, 1, 1] = cx;  out[0, 1, 2] = -sx
            out[0, 2, 0] = 0.;  out[0, 2, 1] = sx;  out[0, 2, 2] =  cx
        else:
            # both omes and chi are None... return a single identity matrix.
            out[0, 0, 0] = 1.;  out[0, 0, 1] = 0.;  out[0, 0, 2] = 0.
            out[0, 1, 0] = 0.;  out[0, 1, 1] = 1.;  out[0, 1, 2] = 0.
            out[0, 2, 0] = 0.;  out[0, 2, 1] = 0.;  out[0, 2, 2] = 1.


    return out


@xf_api
def angles_to_gvec(angs,
                   beam_vec=None, eta_vec=None,
                   chi=None, rmat_c=None):
    """Note about this implementation:
    This used to take rmat_b instead of the pair beam_vec, eta_vec. So it may require
    some checking.
    """
    orig_ndim = angs.ndim
    angs = np.atleast_2d(angs)
    nvecs, dim = angs.shape

    # make vectors in beam frame
    gvec_b = _angles_to_gvec_helper(angs[:,0:2])

    # _rmat_s_helper could return None to mean "Identity" when chi and ome are None.
    omes = angs[:, 2] if dim > 2 else None
    if chi is not None or omes is not None:
        rmat_s = _rmat_s_helper(chi=chi, omes=omes)
    else:
        rmat_s = None

    # apply defaults to beam_vec and eta_vec.
    # TODO: use a default rmat when beam_vec and eta_vec are None so computations
    #       can be avoided?
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    eta_vec = eta_vec if eta_vec is not None else cnst.beam_vec
    rmat_b = make_beam_rmat(beam_vec, eta_vec)

    out = _beam_to_crystal(gvec_b,
                           rmat_b=rmat_b, rmat_s=rmat_s, rmat_c=rmat_c)
    return out[0] if orig_ndim == 1 else out


@xf_api
def angles_to_dvec(angs,
                   beam_vec=None, eta_vec=None,
                   chi=None, rmat_c=None):
    """Note about this implementation:

    This used to take rmat_b instead of the pair beam_vec, eta_vec. So it may
    require some checking.
    """
    angs = np.atleast_2d(angs)
    nvecs, dim = angs.shape

    # make vectors in beam frame
    dvec_b = _angles_to_dvec_helper(angs[:,0:2])

    # calculate rmat_s
    omes = angs[:, 2] if dim>2 else None
    if chi is not None or omes is not None:
        rmat_s = _rmat_s_helper(chi=chi, omes=omes)
    else:
        rmat_s = None

    # apply defaults to beam_vec and eta_vec.
    # TODO: use a default rmat when beam_vec and eta_vec are None so computations
    #       can be avoided?
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    eta_vec = eta_vec if eta_vec is not None else cnst.beam_vec
    rmat_b = make_beam_rmat(beam_vec, eta_vec)

    return _beam_to_crystal(dvec_b,
                            rmat_b=rmat_b, rmat_s=rmat_s, rmat_c=rmat_c)


# this could be a gufunc... (n)->()
@numba.njit
def _row_norm(a, out=None):
    n, dim = a.shape
    out = out if out is not None else np.empty(n, dtype=a.dtype)
    for i in range(n):
        nrm = 0.0
        for j in range(dim):
            x = a[i, j]
            nrm += x*x
        out[i] = np.sqrt(nrm)

    return out


# this and _unit_vector_single would be better as a gufunc.
@numba.njit
def _unit_vector_single(a, out=None):
    out = out if out is not None else np.empty_like(a)

    n = len(a)
    sqr_norm = a[0]*a[0]
    for i in range(1, n):
        sqr_norm += a[i]*a[i]

        # prevent divide by zero
    if sqr_norm > cnst.epsf:
        recip_norm = 1.0 / np.sqrt(sqr_norm)
        out[:] = a[:] * recip_norm
    else:
        out[:] = a[:]

    return out


@numba.njit
def _unit_vector_multi(a, out=None):
    out = out if out is not None else np.empty_like(a)

    n, dim = a.shape
    for i in range(n):
        #_unit_vector_single(a[i], out=out[i])
        sqr_norm = a[i, 0] * a[i, 0]

        for j in range(1, dim):
            sqr_norm += a[i, j]*a[i, j]

        if sqr_norm > cnst.epsf:
            recip_norm = 1.0 / np.sqrt(sqr_norm)
            out[i,:] = a[i,:] * recip_norm
        else:
            out[i,:] = a[i,:]

    return out

@xf_api
def row_norm(vec_in):
    """
    return row-wise norms for a list of vectors
    """
    # TODO: leave this to a PRECONDITION in the xf_api?
    if vec_in.ndim == 1:
        out = _row_norm(np.atleast_2d(vec_in))[0]
    elif vec_in.ndim == 2:
        out = _row_norm(vec_in)
    else:
        raise ValueError(
            "incorrect shape: arg must be  1-d or 2-d, yours is %d"
            % (len(vec_in.shape)))

    return out


@xf_api
def unit_vector(vec_in):
    """
    normalize array of column vectors (hstacked, axis = 0)
    """
    if vec_in.ndim == 1:
        out = _unit_vector_single(vec_in)
    elif vec_in.ndim == 2:
        out = _unit_vector_multi(vec_in)
    else:
        raise ValueError(
            "incorrect arg shape; must be 1-d or 2-d, yours is %d-d"
            % (vec_in.ndim)
        )
    return out


@numba.njit
def _make_rmat_of_expmap(x, out=None):
    """
    TODO:

    Test effectiveness of two options:

    1) avoid conditional inside for loop and use np.divide to return NaN
       for the phi = 0 cases, and deal with it later; or
    2) catch phi = 0 cases inside the loop and just return squeezed answer
    """
    n = len(x)
    out = out if out is not None else np.empty((n,3,3), dtype=x.dtype)
    for i in range(n):
        phi = np.sqrt(x[i, 0]*x[i, 0] + x[i, 1]*x[i, 1] + x[i, 2]*x[i, 2])
        if phi <= cnst.sqrt_epsf:
            out[i, 0, 0] = 1.;  out[i, 0, 1] = 0.;  out[i, 0, 2] = 0.
            out[i, 1, 0] = 0.;  out[i, 1, 1] = 1.;  out[i, 1, 2] = 0.
            out[i, 2, 0] = 0.;  out[i, 2, 1] = 0.;  out[i, 2, 2] = 1.
        else:
            f1 = np.sin(phi)/phi
            f2 = (1. - np.cos(phi)) / (phi*phi)

            out[i, 0, 0] = 1. - f2*(x[i, 2]*x[i, 2] + x[i, 1]*x[i, 1])
            out[i, 0, 1] = f2*x[i, 1]*x[i, 0] - f1*x[i, 2]
            out[i, 0, 2] = f1*x[i, 1] + f2*x[i, 2]*x[i, 0]

            out[i, 1, 0] = f1*x[i, 2] + f2*x[i, 1]*x[i, 0]
            out[i, 1, 1] = 1. - f2*(x[i, 2]*x[i, 2] + x[i, 0]*x[i, 0])
            out[i, 1, 2] = f2*x[i, 2]*x[i, 1] - f1*x[i, 0]

            out[i, 2, 0] = f2*x[i, 2]*x[i, 0] - f1*x[i, 1]
            out[i, 2, 1] = f1*x[i, 0] + f2*x[i, 2]*x[i, 1]
            out[i, 2, 2] = 1. - f2*(x[i, 1]*x[i, 1] + x[i, 0]*x[i, 0])

    return out


"""
if the help above was set up to return nans...

def make_rmat_of_expmap(exp_map):
    exp_map = np.atleast_2d(exp_map)
    rmats = np.empty((len(exp_map), 3, 3))
    _make_rmat_of_expmap(exp_map, rmats)
    chk = np.isnan(rmats)
    if np.any(chk):
        rmats[chk] = np.tile(
            [1., 0., 0., 0., 1., 0., 0., 0., 1.], np.sum(chk)/9
            )
    return rmats
"""

@xf_api
def make_rmat_of_expmap(exp_map):
    exp_map = np.atleast_2d(exp_map)
    rmats = _make_rmat_of_expmap(exp_map)
    return np.squeeze(rmats)


@xf_api
@xfapi_jit
def make_beam_rmat(bvec_l, evec_l):
    # bvec_l and evec_l CANNOT have 0 magnitude!
    # must catch this case as well as colinear bhat_l/ehat_l elsewhere...
    bvec_mag = np.sqrt(bvec_l[0]**2 + bvec_l[1]**2 + bvec_l[2]**2)

    if bvec_mag < cnst.sqrt_epsf:
        raise RuntimeError("bvec_l MUST NOT be ZERO!")
        pass

    # assign Ze as -bhat_l
    Ze0 = -bvec_l[0] / bvec_mag
    Ze1 = -bvec_l[1] / bvec_mag
    Ze2 = -bvec_l[2] / bvec_mag

    # find Ye as Ze ^ ehat_l
    Ye0 = Ze1*evec_l[2] - evec_l[1]*Ze2
    Ye1 = Ze2*evec_l[0] - evec_l[2]*Ze0
    Ye2 = Ze0*evec_l[1] - evec_l[0]*Ze1

    Ye_mag = np.sqrt(Ye0**2 + Ye1**2 + Ye2**2)
    if Ye_mag < cnst.sqrt_epsf:
        raise RuntimeError("bvec_l and evec_l MUST NOT be collinear!")
        pass

    out = np.empty((3,3), dtype=bvec_l.dtype)
    Ye0 /= Ye_mag
    Ye1 /= Ye_mag
    Ye2 /= Ye_mag

    # find Xe as Ye ^ Ze
    Xe0 = Ye1*Ze2 - Ze1*Ye2
    Xe1 = Ye2*Ze0 - Ze2*Ye0
    Xe2 = Ye0*Ze1 - Ze0*Ye1


    out[0, 0] = Xe0
    out[0, 1] = Ye0
    out[0, 2] = Ze0

    out[1, 0] = Xe1
    out[1, 1] = Ye1
    out[1, 2] = Ze1

    out[2, 0] = Xe2
    out[2, 1] = Ye2
    out[2, 2] = Ze2

    return out


