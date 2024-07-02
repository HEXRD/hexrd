# -*- coding: utf-8 -*-
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

import numpy as np

from numpy.linalg import svd

from scipy import sparse

from hexrd.utils.decorators import numba_njit_if_available
from hexrd import constants
from hexrd.constants import USE_NUMBA
if USE_NUMBA:
    import numba
    from numba import prange

# module variables
sqr6i = 1./np.sqrt(6.)
sqr3i = 1./np.sqrt(3.)
sqr2i = 1./np.sqrt(2.)
sqr2 = np.sqrt(2.)
sqr3 = np.sqrt(3.)
sqr2b3 = np.sqrt(2./3.)

fpTol = constants.epsf  # 2.220446049250313e-16
vTol = 100*fpTol


def columnNorm(a):
    """
    normalize array of column vectors (hstacked, axis = 0)
    """
    if len(a.shape) > 2:
        raise RuntimeError(
            "incorrect shape: arg must be 1-d or 2-d, yours is %d"
            % (len(a.shape))
        )

    cnrma = np.sqrt(np.sum(np.asarray(a)**2, axis=0))

    return cnrma


def rowNorm(a):
    """
    normalize array of row vectors (vstacked, axis = 1)
    """
    if len(a.shape) > 2:
        raise RuntimeError(
            "incorrect shape: arg must be 1-d or 2-d, yours is %d"
            % (len(a.shape))
        )

    cnrma = np.sqrt(np.sum(np.asarray(a)**2, axis=1))

    return cnrma


def unitVector(a):
    """
    normalize array of column vectors (hstacked, axis = 0)
    """
    assert a.ndim in [1, 2], \
        "incorrect arg shape; must be 1-d or 2-d, yours is %d-d" % (a.ndim)

    ztol = constants.ten_epsf

    m = a.shape[0]
    n = 1

    nrm = np.tile(np.sqrt(np.sum(np.asarray(a)**2, axis=0)), (m, n))

    # prevent divide by zero
    zchk = nrm <= ztol
    nrm[zchk] = 1.0

    nrma = a/nrm

    return nrma


def nullSpace(A, tol=vTol):
    """
    computes the null space of the real matrix A
    """
    assert A.ndim == 2, \
        'input must be 2-d; yours is %d-d' % (A.ndim)

    n, m = A.shape

    if n > m:
        return nullSpace(A.T, tol).T

    U, S, V = svd(A)

    S = np.hstack([S, np.zeros(m - n)])

    null_mask = (S <= tol)
    null_space = V[null_mask, :]

    return null_space


def blockSparseOfMatArray(matArray):
    """
    blockSparseOfMatArray

    Constructs a block diagonal sparse matrix (csc format) from a
    (p, m, n) ndarray of p (m, n) arrays

    ...maybe optional args to pick format type?
    """

    # if isinstance(args[0], str):
    #    a = args[0]
    # if a == 'csc': ...

    if len(matArray.shape) != 3:
        raise RuntimeError("input array is not the correct shape!")

    p = matArray.shape[0]
    m = matArray.shape[1]
    n = matArray.shape[2]

    mn = m*n
    jmax = p*n
    imax = p*m
    ntot = p*m*n

    rl = np.asarray(list(range(p)), 'int')
    rm = np.asarray(list(range(m)), 'int')
    rjmax = np.asarray(list(range(jmax)), 'int')

    sij = matArray.transpose(0, 2, 1).reshape(1, ntot).squeeze()
    j = np.reshape(np.tile(rjmax, (m, 1)).T, (1, ntot))
    i = np.reshape(np.tile(rm, (1, jmax)), (1, ntot)) + \
        np.reshape(np.tile(m*rl, (mn, 1)).T, (1, ntot))

    ij = np.concatenate((i, j), axis=0)

    # syntax as of scipy-0.7.0
    # csc_matrix((data, indices, indptr), shape=(M, N))
    smat = sparse.csc_matrix((sij, ij), shape=(imax, jmax))

    return smat


def symmToVecMV(A, scale=True):
    """
    convert from symmetric matrix to Mandel-Voigt vector
    representation (JVB)
    """
    if scale:
        fac = sqr2
    else:
        fac = 1.
    mvvec = np.zeros(6, dtype='float64')
    mvvec[0] = A[0, 0]
    mvvec[1] = A[1, 1]
    mvvec[2] = A[2, 2]
    mvvec[3] = fac * A[1, 2]
    mvvec[4] = fac * A[0, 2]
    mvvec[5] = fac * A[0, 1]
    return mvvec


def vecMVToSymm(A, scale=True):
    """
    convert from Mandel-Voigt vector to symmetric matrix
    representation (JVB)
    """
    if scale:
        fac = sqr2
    else:
        fac = 1.
    symm = np.zeros((3, 3), dtype='float64')
    symm[0, 0] = A[0]
    symm[1, 1] = A[1]
    symm[2, 2] = A[2]
    symm[1, 2] = A[3] / fac
    symm[0, 2] = A[4] / fac
    symm[0, 1] = A[5] / fac
    symm[2, 1] = A[3] / fac
    symm[2, 0] = A[4] / fac
    symm[1, 0] = A[5] / fac
    return symm


def vecMVCOBMatrix(R):
    """
    GenerateS array of 6 x 6 basis transformation matrices for the
    Mandel-Voigt tensor representation in 3-D given by:

    [A] = [[A_11, A_12, A_13],
           [A_12, A_22, A_23],
           [A_13, A_23, A_33]]

    {A} = [A_11, A_22, A_33, sqrt(2)*A_23, sqrt(2)*A_13, sqrt(2)*A_12]

    where the operation :math:`R*A*R.T` (in tensor notation) is obtained by
    the matrix-vector product [T]*{A}.

    USAGE

        T = vecMVCOBMatrix(R)

    INPUTS

        1) R is (3, 3) an ndarray representing a change of basis matrix

    OUTPUTS

        1) T is (6, 6), an ndarray of transformation matrices as
           described above

    NOTES

        1) Compoments of symmetric 4th-rank tensors transform in a
           manner analogous to symmetric 2nd-rank tensors in full
           matrix notation.

    SEE ALSO

    symmToVecMV, vecMVToSymm, quatToMat
    """
    rdim = len(R.shape)
    if rdim == 2:
        nrot = 1
        R = np.tile(R, (1, 1, 1))
    elif rdim == 3:
        nrot = R.shape[0]
    else:
        raise RuntimeError(
            "R array must be (3, 3) or (n, 3, 3); input has dimension %d"
            % (rdim)
        )

    T = np.zeros((nrot, 6, 6), dtype='float64')

    for i in range(3):
        # Other two i values
        i1, i2 = [k for k in range(3) if k != i]
        for j in range(3):
            # Other two j values
            j1, j2 = [k for k in range(3) if k != j]

            T[:, i, j] = R[:, i, j] ** 2
            T[:, i, j + 3] = sqr2 * R[:, i, j1] * R[:, i, j2]
            T[:, i + 3, j] = sqr2 * R[:, i1, j] * R[:, i2, j]
            T[:, i + 3, j + 3] = (
                R[:, i1, j1] * R[:, i2, j2] + R[:, i1, j2] * R[:, i2, j1]
            )

    if nrot == 1:
        T = T.squeeze()

    return T


def nrmlProjOfVecMV(vec):
    """
    Gives vstacked p x 6 array to To perform n' * A * n as [N]*{A} for
    p hstacked input 3-vectors using the Mandel-Voigt convention.

    Nvec = normalProjectionOfMV(vec)

    *) the input vector array need not be normalized; it is performed in place

    """
    # normalize in place... col vectors!
    n = unitVector(vec)

    nmat = np.array(
        [n[0, :]**2,
         n[1, :]**2,
         n[2, :]**2,
         sqr2 * n[1, :] * n[2, :],
         sqr2 * n[0, :] * n[2, :],
         sqr2 * n[0, :] * n[1, :]],
        dtype='float64'
    )

    return nmat.T


def rankOneMatrix(vec1, *args):
    """
    Create rank one matrices (dyadics) from vectors.

      r1mat = rankOneMatrix(vec1)
      r1mat = rankOneMatrix(vec1, vec2)

      vec1 is m1 x n, an array of n hstacked m1 vectors
      vec2 is m2 x n, (optional) another array of n hstacked m2 vectors

      r1mat is n x m1 x m2, an array of n rank one matrices
                   formed as c1*c2' from columns c1 and c2

      With one argument, the second vector is taken to
      the same as the first.

      Notes:

      *)  This routine loops on the dimension m, assuming this
          is much smaller than the number of points, n.
    """
    if len(vec1.shape) > 2:
        raise RuntimeError("input vec1 is the wrong shape")

    if (len(args) == 0):
        vec2 = vec1.copy()
    else:
        vec2 = args[0]
        if len(vec1.shape) > 2:
            raise RuntimeError("input vec2 is the wrong shape")

    m1, n1 = np.asmatrix(vec1).shape
    m2, n2 = np.asmatrix(vec2).shape

    if (n1 != n2):
        raise RuntimeError("Number of vectors differ in arguments.")

    m1m2 = m1 * m2

    r1mat = np.zeros((m1m2, n1), dtype='float64')

    mrange = np.asarray(list(range(m1)), dtype='int')

    for i in range(m2):
        r1mat[mrange, :] = vec1 * np.tile(vec2[i, :], (m1, 1))
        mrange = mrange + m1

    r1mat = np.reshape(r1mat.T, (n1, m2, m1)).transpose(0, 2, 1)
    return r1mat.squeeze()


def skew(A):
    """
    skew-symmetric decomposition of n square (m, m) ndarrays.  Result
    is a (squeezed) (n, m, m) ndarray
    """
    A = np.asarray(A)

    if A.ndim == 2:
        m = A.shape[0]
        n = A.shape[1]
        if m != n:
            raise RuntimeError(
                "this function only works for square arrays; yours is (%d, %d)"
                % (m, n)
            )
        A.resize(1, m, n)
    elif A.ndim == 3:
        m = A.shape[1]
        n = A.shape[2]
        if m != n:
            raise RuntimeError("this function only works for square arrays")
    else:
        raise RuntimeError("this function only works for square arrays")

    return np.squeeze(0.5*(A - A.transpose(0, 2, 1)))


def symm(A):
    """
    symmetric decomposition of n square (m, m) ndarrays.  Result
    is a (squeezed) (n, m, m) ndarray.
    """
    A = np.asarray(A)

    if A.ndim == 2:
        m = A.shape[0]
        n = A.shape[1]
        if m != n:
            raise RuntimeError(
                "this function only works for square arrays; yours is (%d, %d)"
                % (m, n)
            )
        A.resize(1, m, n)
    elif A.ndim == 3:
        m = A.shape[1]
        n = A.shape[2]
        if m != n:
            raise RuntimeError("this function only works for square arrays")
    else:
        raise RuntimeError("this function only works for square arrays")

    return np.squeeze(0.5*(A + A.transpose(0, 2, 1)))


def skewMatrixOfVector(w):
    """
    skewMatrixOfVector(w)

    given a (3, n) ndarray, w,  of n hstacked axial vectors, computes
    the associated skew matrices and stores them in an (n, 3, 3)
    ndarray.  Result is (3, 3) for w.shape = (3, 1) or (3, ).

    See also: vectorOfSkewMatrix
    """
    dims = w.ndim
    stackdim = 0
    if dims == 1:
        if len(w) != 3:
            raise RuntimeError('input is not a 3-d vector')
        else:
            w = np.vstack(w)
            stackdim = 1
    elif dims == 2:
        if w.shape[0] != 3:
            raise RuntimeError(
                'input is of incorrect shape; expecting shape[0] = 3'
            )
        else:
            stackdim = w.shape[1]
    else:
        raise RuntimeError(
            'input is incorrect shape; expecting ndim = 1 or 2'
        )

    zs = np.zeros((1, stackdim), dtype='float64')
    W = np.vstack(
        [zs,
         -w[2, :],
         w[1, :],
         w[2, :],
         zs,
         -w[0, :],
         -w[1, :],
         w[0, :],
         zs]
    )

    return np.squeeze(np.reshape(W.T, (stackdim, 3, 3)))


def vectorOfSkewMatrix(W):
    """
    vectorOfSkewMatrix(W)

    given an (n, 3, 3) or (3, 3) ndarray, W, of n stacked 3x3 skew
    matrices, computes the associated axial vector(s) and stores them
    in an (3, n) ndarray.  Result always has ndim = 2.

    See also: skewMatrixOfVector
    """
    stackdim = 0
    if W.ndim == 2:
        if W.shape[0] != 3 or W.shape[0] != 3:
            raise RuntimeError('input is not (3, 3)')
        stackdim = 1
        W.resize(1, 3, 3)
    elif W.ndim == 3:
        if W.shape[1] != 3 or W.shape[2] != 3:
            raise RuntimeError('input is not (3, 3)')
        stackdim = W.shape[0]
    else:
        raise RuntimeError('input is incorrect shape; expecting (n, 3, 3)')

    w = np.zeros((3, stackdim), dtype='float64')
    for i in range(stackdim):
        w[:, i] = np.r_[-W[i, 1, 2], W[i, 0, 2], -W[i, 0, 1]]

    return w


def multMatArray(ma1, ma2):
    """
    multiply two 3-d arrays of 2-d matrices
    """
    shp1 = ma1.shape
    shp2 = ma2.shape

    if len(shp1) != 3 or len(shp2) != 3:
        raise RuntimeError(
            'input is incorrect shape; '
            + 'expecting len(ma1).shape = len(ma2).shape = 3'
        )

    if shp1[0] != shp2[0]:
        raise RuntimeError('mismatch on number of matrices')

    if shp1[2] != shp2[1]:
        raise RuntimeError('mismatch on internal matrix dimensions')

    prod = np.zeros((shp1[0], shp1[1], shp2[2]))
    for j in range(shp1[0]):
        prod[j, :, :] = np.dot(ma1[j, :, :], ma2[j, :, :])

    return prod


def uniqueVectors(v, tol=1.0e-12):
    """
    Sort vectors and discard duplicates.

      USAGE:

          uvec = uniqueVectors(vec, tol=1.0e-12)

    v   --
    tol -- (optional) comparison tolerance

    D. E. Boyce 2010-03-18
    """

    vdims = v.shape

    iv = np.zeros(vdims)
    for row in range(vdims[0]):
        tmpord = np.argsort(v[row, :]).tolist()
        tmpsrt = v[np.ix_([row], tmpord)].squeeze()
        tmpcmp = abs(tmpsrt[1:] - tmpsrt[0:-1])
        indep = np.hstack([True, tmpcmp > tol])  # independent values
        rowint = indep.cumsum()
        iv[np.ix_([row], tmpord)] = rowint
    #
    #  Dictionary sort from bottom up
    #
    iNum = np.lexsort(iv)
    ivSrt = iv[:, iNum]
    vSrt = v[:, iNum]

    ivInd = np.zeros(vdims[1], dtype='int')
    nUniq = 1
    ivInd[0] = 0
    for col in range(1, vdims[1]):
        if any(ivSrt[:, col] != ivSrt[:, col - 1]):
            ivInd[nUniq] = col
            nUniq += 1

    return vSrt[:, ivInd[0:nUniq]]


def findDuplicateVectors_old(vec, tol=vTol, equivPM=False):
    """
    Find vectors in an array that are equivalent to within
    a specified tolerance

      USAGE:

          eqv = DuplicateVectors(vec, *tol)

      INPUT:

          1) vec is n x m, a double array of m horizontally concatenated
                           n-dimensional vectors.
         *2) tol is 1 x 1, a scalar tolerance.  If not specified, the default
                           tolerance is 1e-14.
         *3) set equivPM to True if vec and -vec
             are to be treated as equivalent

      OUTPUT:

          1) eqv is 1 x p, a list of p equivalence relationships.

      NOTES:

          Each equivalence relationship is a 1 x q vector of indices that
          represent the locations of duplicate columns/entries in the array
          vec.  For example:

                | 1     2     2     2     1     2     7 |
          vec = |                                       |
                | 2     3     5     3     2     3     3 |

          eqv = [[1x2 double]    [1x3 double]], where

          eqv[0] = [0  4]
          eqv[1] = [1  3  5]
    """

    vlen = vec.shape[1]
    vlen0 = vlen
    orid = np.asarray(list(range(vlen)), dtype="int")

    torid = orid.copy()
    tvec = vec.copy()

    eqv = []
    eqvTot = 0
    uid = 0

    ii = 1
    while vlen > 1 and ii < vlen0:
        dupl = np.tile(tvec[:, 0], (vlen, 1))

        if not equivPM:
            diff = abs(tvec - dupl.T).sum(0)
            match = abs(diff[1:]) <= tol    # logical to find duplicates
        else:
            diffn = abs(tvec - dupl.T).sum(0)
            matchn = abs(diffn[1:]) <= tol
            diffp = abs(tvec + dupl.T).sum(0)
            matchp = abs(diffp[1:]) <= tol
            match = matchn + matchp

        kick = np.hstack([True, match])    # pick self too

        if kick.sum() > 1:
            eqv += [torid[kick].tolist()]
            eqvTot = np.hstack([eqvTot, torid[kick]])
            uid = np.hstack([uid, torid[kick][0]])

        cmask = np.ones((vlen,))
        cmask[kick] = 0
        cmask = cmask != 0

        tvec = tvec[:, cmask]

        torid = torid[cmask]

        vlen = tvec.shape[1]

        ii += 1

    if len(eqv) == 0:
        eqvTot = []
        uid = []
    else:
        eqvTot = eqvTot[1:].tolist()
        uid = uid[1:].tolist()

    # find all single-instance vectors
    singles = np.sort(np.setxor1d(eqvTot, list(range(vlen0))))

    # now construct list of unique vector column indices
    uid = np.int_(np.sort(np.union1d(uid, singles))).tolist()
    # make sure is a 1D list
    if not hasattr(uid, '__len__'):
        uid = [uid]

    return eqv, uid

def findDuplicateVectors(vec, tol=vTol, equivPM=False):
    eqv = _findduplicatevectors(vec, tol, equivPM)
    uid = np.arange(0, vec.shape[1], dtype=np.int64)
    mask = ~np.isnan(eqv)
    idx = eqv[mask].astype(np.int64)
    uid2 = list(np.delete(uid, idx))
    eqv2 = []
    for ii in range(eqv.shape[0]):
        v = eqv[ii, mask[ii, :]]
        if v.shape[0] > 0:
            eqv2.append([ii] + list(v.astype(np.int64)))
    return eqv2, uid2


@numba_njit_if_available(cache=True, nogil=True)
def _findduplicatevectors(vec, tol, equivPM):
    """
    Find vectors in an array that are equivalent to within
    a specified tolerance. code is accelerated by numba

      USAGE:

          eqv = DuplicateVectors(vec, *tol)

      INPUT:

          1) vec is n x m, a double array of m horizontally concatenated
                           n-dimensional vectors.
         *2) tol is 1 x 1, a scalar tolerance.  If not specified, the default
                           tolerance is 1e-14.
         *3) set equivPM to True if vec and -vec
             are to be treated as equivalent

      OUTPUT:

          1) eqv is 1 x p, a list of p equivalence relationships.

      NOTES:

          Each equivalence relationship is a 1 x q vector of indices that
          represent the locations of duplicate columns/entries in the array
          vec.  For example:

                | 1     2     2     2     1     2     7 |
          vec = |                                       |
                | 2     3     5     3     2     3     3 |

          eqv = [[1x2 double]    [1x3 double]], where

          eqv[0] = [0  4]
          eqv[1] = [1  3  5]
    """

    if equivPM:
        vec2 = -vec.copy()

    n = vec.shape[0]
    m = vec.shape[1]

    eqv = np.zeros((m, m), dtype=np.float64)
    eqv[:] = np.nan
    eqv_elem_master = []

    for ii in range(m):
        ctr = 0
        eqv_elem = np.zeros((m, ), dtype=np.int64)
        for jj in range(ii+1, m):
            if not jj in eqv_elem_master:
                if equivPM:
                    diff  = np.sum(np.abs(vec[:, ii]-vec2[:, jj]))
                    diff2 = np.sum(np.abs(vec[:, ii]-vec[:, jj]))
                    if diff < tol or diff2 < tol:
                        eqv_elem[ctr] = jj
                        eqv_elem_master.append(jj)
                        ctr += 1
                else:
                    diff = np.sum(np.abs(vec[:, ii]-vec[:, jj]))
                    if diff < tol:
                        eqv_elem[ctr] = jj
                        eqv_elem_master.append(jj)
                        ctr += 1

        for kk in range(ctr):
            eqv[ii, kk] = eqv_elem[kk]

    return eqv

def normvec(v):
    mag = np.linalg.norm(v)
    return mag


def normvec3(v):
    """
    ??? deprecated
    """
    mag = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return mag


def normalized(v):
    mag = normvec(v)  # normvec3(v)
    n = v / mag
    return n


def cross(v1, v2):
    # return the cross product of v1 with another vector
    # return a vector
    newv3 = np.zeros(3, dtype='float64')
    newv3[0] = v1[1]*v2[2] - v1[2]*v2[1]
    newv3[1] = v1[2]*v2[0] - v1[0]*v2[2]
    newv3[2] = v1[0]*v2[1] - v1[1]*v2[0]
    return newv3


def determinant3(mat):
    v = np.cross(mat[0, :], mat[1, :])
    det = np.sum(mat[2, :] * v[:])
    return det


def strainTenToVec(strainTen):
    strainVec = np.zeros(6, dtype='float64')
    strainVec[0] = strainTen[0, 0]
    strainVec[1] = strainTen[1, 1]
    strainVec[2] = strainTen[2, 2]
    strainVec[3] = 2*strainTen[1, 2]
    strainVec[4] = 2*strainTen[0, 2]
    strainVec[5] = 2*strainTen[0, 1]
    strainVec = np.atleast_2d(strainVec).T
    return strainVec


def strainVecToTen(strainVec):
    strainTen = np.zeros((3, 3), dtype='float64')
    strainTen[0, 0] = strainVec[0]
    strainTen[1, 1] = strainVec[1]
    strainTen[2, 2] = strainVec[2]
    strainTen[1, 2] = strainVec[3] / 2.
    strainTen[0, 2] = strainVec[4] / 2.
    strainTen[0, 1] = strainVec[5] / 2.
    strainTen[2, 1] = strainVec[3] / 2.
    strainTen[2, 0] = strainVec[4] / 2.
    strainTen[1, 0] = strainVec[5] / 2.
    return strainTen


def stressTenToVec(stressTen):
    stressVec = np.zeros(6, dtype='float64')
    stressVec[0] = stressTen[0, 0]
    stressVec[1] = stressTen[1, 1]
    stressVec[2] = stressTen[2, 2]
    stressVec[3] = stressTen[1, 2]
    stressVec[4] = stressTen[0, 2]
    stressVec[5] = stressTen[0, 1]
    stressVec = np.atleast_2d(stressVec).T
    return stressVec


def stressVecToTen(stressVec):

    stressTen = np.zeros((3, 3), dtype='float64')
    stressTen[0, 0] = stressVec[0]
    stressTen[1, 1] = stressVec[1]
    stressTen[2, 2] = stressVec[2]
    stressTen[1, 2] = stressVec[3]
    stressTen[0, 2] = stressVec[4]
    stressTen[0, 1] = stressVec[5]
    stressTen[2, 1] = stressVec[3]
    stressTen[2, 0] = stressVec[4]
    stressTen[1, 0] = stressVec[5]

    return stressTen


def ale3dStrainOutToV(vecds):
    """
    convert from vecds representation to symmetry matrix
    takes 5 components of evecd and the 6th component is lndetv
    """
    eps = np.zeros([3, 3], dtype='float64')
    # Akk_by_3 = sqr3i * vecds[5]  # -p
    a = np.exp(vecds[5])**(1./3.)  # -p
    t1 = sqr2i*vecds[0]
    t2 = sqr6i*vecds[1]

    eps[0, 0] = t1 - t2
    eps[1, 1] = -t1 - t2
    eps[2, 2] = sqr2b3*vecds[1]
    eps[1, 0] = vecds[2] * sqr2i
    eps[2, 0] = vecds[3] * sqr2i
    eps[2, 1] = vecds[4] * sqr2i

    eps[0, 1] = eps[1, 0]
    eps[0, 2] = eps[2, 0]
    eps[1, 2] = eps[2, 1]

    epstar = eps/a

    V = (constants.identity_3x3 + epstar)*a
    Vinv = (constants.identity_3x3 - epstar)/a

    return V, Vinv


def vecdsToSymm(vecds):
    """convert from vecds representation to symmetry matrix"""
    A = np.zeros([3, 3], dtype='float64')
    Akk_by_3 = sqr3i * vecds[5]  # -p
    t1 = sqr2i*vecds[0]
    t2 = sqr6i*vecds[1]

    A[0, 0] = t1 - t2 + Akk_by_3
    A[1, 1] = -t1 - t2 + Akk_by_3
    A[2, 2] = sqr2b3*vecds[1] + Akk_by_3
    A[1, 0] = vecds[2] * sqr2i
    A[2, 0] = vecds[3] * sqr2i
    A[2, 1] = vecds[4] * sqr2i

    A[0, 1] = A[1, 0]
    A[0, 2] = A[2, 0]
    A[1, 2] = A[2, 1]
    return A


def traceToVecdsS(Akk):
    return sqr3i * Akk


def vecdsSToTrace(vecdsS):
    return vecdsS * sqr3


def trace3(A):
    return A[0, 0] + A[1, 1] + A[2, 2]


def symmToVecds(A):
    """convert from symmetry matrix to vecds representation"""
    vecds = np.zeros(6, dtype='float64')
    vecds[0] = sqr2i * (A[0, 0] - A[1, 1])
    vecds[1] = sqr6i * (2. * A[2, 2] - A[0, 0] - A[1, 1])
    vecds[2] = sqr2 * A[1, 0]
    vecds[3] = sqr2 * A[2, 0]
    vecds[4] = sqr2 * A[2, 1]
    vecds[5] = traceToVecdsS(trace3(A))
    return vecds


def solve_wahba(v, w, weights=None):
    """
    take unique vectors 3-vectors v = [[v0], [v1], ..., [vn]] in frame 1 that
    are aligned with vectors w = [[w0], [w1], ..., [wn]] in frame 2 and solve
    for the rotation that takes components in frame 1 to frame 2

    minimizes the cost function:

      J(R) = 0.5 * sum_{k=1}^{N} a_k * || w_k - R*v_k ||^2

    INPUTS:
      v is list-like, where each entry is a length 3 vector
      w is list-like, where each entry is a length 3 vector

      len(v) == len(w)

      weights are optional, and must have the same length as v, w

    OUTPUT:
      (3, 3) orthognal matrix that takes components in frame 1 to frame 2
    """
    n_vecs = len(v)

    assert len(w) == n_vecs

    if weights is not None:
        assert len(weights) == n_vecs
    else:
        weights = np.ones(n_vecs)

    # cast v, w, as arrays if not
    v = np.atleast_2d(v)
    w = np.atleast_2d(w)

    # compute weighted outer product sum
    B = np.zeros((3, 3))
    for i in range(n_vecs):
        B += weights[i]*np.dot(w[i].reshape(3, 1), v[i].reshape(1, 3))

    # compute svd
    Us, Ss, VsT = svd(B)

    # form diagonal matrix for solution
    M = np.diag([1., 1., np.linalg.det(Us)*np.linalg.det(VsT)])
    return np.dot(Us, np.dot(M, VsT))

# =============================================================================
# Numba-fied frame cache writer
# =============================================================================


if USE_NUMBA:
    @numba.njit(cache=True, nogil=True)
    def extract_ijv(in_array, threshold, out_i, out_j, out_v):
        n = 0
        w, h = in_array.shape
        for i in range(w):
            for j in range(h):
                v = in_array[i, j]
                if v > threshold:
                    out_i[n] = i
                    out_j[n] = j
                    out_v[n] = v
                    n += 1
        return n
else:    # not USE_NUMBA
    def extract_ijv(in_array, threshold, out_i, out_j, out_v):
        mask = in_array > threshold
        n = np.sum(mask)
        tmp_i, tmp_j = mask.nonzero()
        out_i[:n] = tmp_i
        out_j[:n] = tmp_j
        out_v[:n] = in_array[mask]
        return n
