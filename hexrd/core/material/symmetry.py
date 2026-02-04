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
# -*-python-*-
#
# Module containing functions relevant to symmetries
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray
from numpy import array, sqrt, pi, vstack, c_, dot, argmax

# from hexrd.core.rotations import quatOfAngleAxis, quatProductMatrix, fixQuat
from hexrd.core import rotations as rot
from hexrd.core import constants
from hexrd.core.utils.decorators import memoize

# Imports in case others are importing from here
from hexrd.core.rotations import (
    toFundamentalRegion,
    ltypeOfLaueGroup,
    quatOfLaueGroup,
)


# =============================================================================
# Module vars
# =============================================================================

eps = constants.sqrt_epsf
sq3by2 = sqrt(3.0) / 2.0
piby2 = pi / 2.0
piby3 = pi / 3.0
piby4 = pi / 4.0
piby6 = pi / 6.0


# =============================================================================
# Functions
# =============================================================================


def GeneratorString(sgnum: int) -> str:
    '''
    these rhombohedral space groups have a hexagonal setting
    with different symmetry matrices and generator strings
    146: 231
    148: 232
    ...
    and so on
    '''
    sg = sgnum - 1
    # sgdict = {146:231, 148:232, 155:233, 160:234, 161:235, 166:236, 167:237}
    # if(sgnum in sgdict):
    #     sg = sgdict[sgnum]-1

    return constants.SYM_GL[sg]


def MakeGenerators(genstr: str, setting: int) -> tuple[NDArray[np.float64], bool]:
    mat = SYM_fillgen('aOOO')
    genmat = mat

    # genmat[0,:,:] = constants.SYM_GENERATORS['a']
    centrosymmetric = False

    # check if space group has inversion symmetry
    if genstr[0] == '1':
        mat = SYM_fillgen('hOOO')
        genmat = np.concatenate((genmat, mat))
        centrosymmetric = True

    istop = 2
    n = int(genstr[1])
    if n > 0:
        for i in range(n):
            istart = 2 + i * 4
            istop = 2 + (i + 1) * 4

            mat = SYM_fillgen(genstr[istart:istop])
            genmat = np.concatenate((genmat, mat))

    # if there is an alternate setting for this space group check if the alternate
    # setting needs to be used
    if genstr[istop] != '0':
        if setting != 0:
            sym = np.squeeze(SYM_fillgen('a' + genstr[istop + 1 : istop + 4], sgn=-1))
            sym2 = np.squeeze(SYM_fillgen('a' + genstr[istop + 1 : istop + 4]))
            for i in range(1, genmat.shape[0]):
                generator = np.dot(sym2, np.dot(np.squeeze(genmat[i, :, :]), sym))
                frac = np.modf(generator[0:3, 3])[0]
                frac[frac < 0.0] += 1.0
                frac[np.abs(frac) < 1e-5] = 0.0
                frac[np.abs(frac - 1.0) < 1e-5] = 0.0
                generator[0:3, 3] = frac
                genmat[i, :, :] = generator

    return genmat, centrosymmetric


def SYM_fillgen(t: str, sgn: Literal[1, -1] = 1) -> NDArray[np.float64]:
    mat = np.zeros([4, 4])
    mat[3, 3] = 1.0

    mat[0:3, 0:3] = constants.SYM_GENERATORS[t[0]]
    mat[0:3, 3] = sgn * np.array(
        [
            constants.SYM_GENERATORS[t[1]],
            constants.SYM_GENERATORS[t[2]],
            constants.SYM_GENERATORS[t[3]],
        ]
    )

    mat = np.broadcast_to(mat, [1, 4, 4])
    return mat


@memoize(maxsize=20)
def GenerateSGSym(
    sgnum: int, setting: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool, bool]:
    '''Get the generators for a space group using the generator string'''
    genstr = GeneratorString(sgnum)
    genmat, centrosymmetric = MakeGenerators(genstr, setting)
    symmorphic = False
    if sgnum in constants.sgnum_symmorphic:
        symmorphic = True
    '''
    use the generator string to get the rest of the factor group

    genmat has shape ngenerators x 4 x 4
    '''
    nsym = genmat.shape[0]

    SYM_SG = genmat

    '''
    generate the factor group
    '''

    k1 = 0
    while k1 < nsym:

        g1 = np.squeeze(SYM_SG[k1, :, :])
        k2 = k1

        while k2 < nsym:
            g2 = np.squeeze(SYM_SG[k2, :, :])
            gnew = np.dot(g1, g2)

            # only fractional parts
            frac = np.modf(gnew[0:3, 3])[0]
            frac[frac < 0.0] += 1.0
            frac[np.abs(frac) < 1e-5] = 0.0
            frac[np.abs(frac - 1.0) < 1e-5] = 0.0
            gnew[0:3, 3] = frac

            if isnew(gnew, SYM_SG):
                gnew = np.broadcast_to(gnew, [1, 4, 4])
                SYM_SG = np.concatenate((SYM_SG, gnew))
                nsym += 1

                if nsym >= 192:
                    k2 = nsym
                    k1 = nsym

            k2 += 1
        k1 += 1

    SYM_PG_d = GeneratePGSym(SYM_SG)
    SYM_PG_d_laue = GeneratePGSym_Laue(SYM_PG_d)

    for s in SYM_PG_d:
        if np.allclose(-np.eye(3), s):
            centrosymmetric = True

    return SYM_SG, SYM_PG_d, SYM_PG_d_laue, centrosymmetric, symmorphic


def GeneratePGSym(SYM_SG):
    '''
    calculate the direct space point group symmetries
    from the space group symmetry. the direct point
    group symmetries are merely the space group
    symmetries with zero translation part. The reciprocal
    ones are calculated from the direct symmetries by
    using the metric tensors, but that is done in the unitcell
    class
    '''
    nsgsym = SYM_SG.shape[0]

    # first fill the identity rotation
    SYM_PG_d = SYM_SG[0, 0:3, 0:3]
    SYM_PG_d = np.broadcast_to(SYM_PG_d, [1, 3, 3])

    for i in range(1, nsgsym):
        g = SYM_SG[i, :, :]
        t = g[0:3, 3]
        g = g[0:3, 0:3]
        if isnew(g, SYM_PG_d):
            g = np.broadcast_to(g, [1, 3, 3])
            SYM_PG_d = np.concatenate((SYM_PG_d, g))

    return SYM_PG_d.astype(np.int32)


def GeneratePGSym_Laue(SYM_PG_d):
    '''
    generate the laue group symmetry for the given set of
    point group symmetry matrices. this function just adds
    the inversion symmetry and goes through the group action
    to generate the entire laue group for the direct point
    point group matrices
    '''

    '''
    first check if the group already has the inversion symmetry
    '''
    for s in SYM_PG_d:
        if np.allclose(s, -np.eye(3)):
            return SYM_PG_d

    '''
    if we get here, then the inversion symmetry is not present
    add the inversion symmetry
    '''
    SYM_PG_d_laue = SYM_PG_d
    g = np.broadcast_to(-np.eye(3).astype(np.int32), [1, 3, 3])
    SYM_PG_d_laue = np.concatenate((SYM_PG_d_laue, g))

    '''
    now go through the group actions and see if its a new matrix
    if it is then add it to the group
    '''
    nsym = SYM_PG_d_laue.shape[0]
    k1 = 0
    while k1 < nsym:
        g1 = np.squeeze(SYM_PG_d_laue[k1, :, :])
        k2 = k1
        while k2 < nsym:
            g2 = np.squeeze(SYM_PG_d_laue[k2, :, :])
            gnew = np.dot(g1, g2)

            if isnew(gnew, SYM_PG_d_laue):
                gnew = np.broadcast_to(gnew, [1, 3, 3])
                SYM_PG_d_laue = np.concatenate((SYM_PG_d_laue, gnew))
                nsym += 1

                if nsym >= 48:
                    k2 = nsym
                    k1 = nsym

            k2 += 1
        k1 += 1

    return SYM_PG_d_laue


@njit(cache=True, nogil=True)
def isnew(mat, sym_mats):
    for g in sym_mats:
        diff = np.sum(np.abs(mat - g))
        if diff < 1e-5:
            return False
    return True


def latticeType(sgnum):

    if sgnum <= 2:
        return 'triclinic'
    elif sgnum > 2 and sgnum <= 15:
        return 'monoclinic'
    elif sgnum > 15 and sgnum <= 74:
        return 'orthorhombic'
    elif sgnum > 74 and sgnum <= 142:
        return 'tetragonal'
    elif sgnum > 142 and sgnum <= 167:
        return 'trigonal'
    elif sgnum > 167 and sgnum <= 194:
        return 'hexagonal'
    elif sgnum > 194 and sgnum <= 230:
        return 'cubic'
    else:
        raise RuntimeError('symmetry.latticeType: unknown space group number')


def MakeGenerators_PGSYM(pggenstr):
    '''
    @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    @DATE    11/23/2020 SS 1.0 original
    @DETAIL. these are the supporting routine to generate the ppint group symmetry
             for any point group. this is needed for the coloring routines
    '''
    ngen = int(pggenstr[0])
    SYM_GEN_PG = np.zeros([ngen, 3, 3])

    for i in range(ngen):
        s = pggenstr[i + 1]
        SYM_GEN_PG[i, :, :] = constants.SYM_GENERATORS[s]

    return SYM_GEN_PG


def GeneratePGSYM(pgsym):
    '''
    @AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    @DATE    11/23/2020 SS 1.0 original
    @DETAIL. generate the point group symmetry given the point group symbol
    '''
    pggenstr = constants.SYM_GL_PG[pgsym]

    SYM_GEN_PG = MakeGenerators_PGSYM(pggenstr)

    '''
    generate the powers of the group
    '''

    '''
    now go through the group actions and see if its a new matrix
    if it is then add it to the group
    '''
    nsym = SYM_GEN_PG.shape[0]
    k1 = 0
    while k1 < nsym:
        g1 = np.squeeze(SYM_GEN_PG[k1, :, :])
        k2 = k1
        while k2 < nsym:
            g2 = np.squeeze(SYM_GEN_PG[k2, :, :])
            gnew = np.dot(g1, g2)

            if isnew(gnew, SYM_GEN_PG):
                gnew = np.broadcast_to(gnew, [1, 3, 3])
                SYM_GEN_PG = np.concatenate((SYM_GEN_PG, gnew))
                nsym += 1

                if nsym >= 48:
                    k2 = nsym
                    k1 = nsym

            k2 += 1
        k1 += 1

    SYM_GEN_PG[np.abs(SYM_GEN_PG) < eps] = 0.0

    return SYM_GEN_PG
