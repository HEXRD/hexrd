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

from numpy import array, sqrt, pi, \
     vstack, c_, dot, \
     argmax

# from hexrd.rotations import quatOfAngleAxis, quatProductMatrix, fixQuat
from hexrd import rotations as rot
from hexrd import constants
import numpy as np

# =============================================================================
# Module vars
# =============================================================================

eps = constants.sqrt_epsf
sq3by2 = sqrt(3.)/2.
piby2 = pi/2.
piby3 = pi/3.
piby4 = pi/4.
piby6 = pi/6.


# =============================================================================
# Functions
# =============================================================================


def toFundamentalRegion(q, crysSym='Oh', sampSym=None):
    """
    """
    qdims = q.ndim
    if qdims == 3:
        l3, m3, n3 = q.shape
        assert m3 == 4, 'your 3-d quaternion array isn\'t the right shape'
        q = q.transpose(0, 2, 1).reshape(l3*n3, 4).T
    if isinstance(crysSym, str):
        qsym_c = rot.quatProductMatrix(
            quatOfLaueGroup(crysSym), 'right'
        )  # crystal symmetry operator
    else:
        qsym_c = rot.quatProductMatrix(crysSym, 'right')

    n = q.shape[1]              # total number of quats
    m = qsym_c.shape[0]         # number of symmetry operations

    #
    # MAKE EQUIVALENCE CLASS
    #
    # Do R * Gc, store as
    # [q[:, 0] * Gc[:, 0:m], ..., 2[:, n-1] * Gc[:, 0:m]]
    qeqv = dot(qsym_c, q).transpose(2, 0, 1).reshape(m*n, 4).T

    if sampSym is None:
        # need to fix quats to sort
        qeqv = rot.fixQuat(qeqv)

        # Reshape scalar comp columnwise by point in qeqv
        q0 = qeqv[0, :].reshape(n, m).T

        # Find q0 closest to origin for each n equivalence classes
        q0maxColInd = argmax(q0, 0) + [x*m for x in range(n)]

        # store representatives in qr
        qr = qeqv[:, q0maxColInd]
    else:
        if isinstance(sampSym, str):
            qsym_s = rot.quatProductMatrix(
                quatOfLaueGroup(sampSym), 'left'
            )  # sample symmetry operator
        else:
            qsym_s = rot.quatProductMatrix(sampSym, 'left')

        p = qsym_s.shape[0]         # number of sample symmetry operations

        # Do Gs * (R * Gc), store as
        # [Gs[:, 0:p] * q[:,   0] * Gc[:, 0], ... , Gs[:, 0:p] * q[:,   0] * Gc[:, m-1], ...
        #  Gs[:, 0:p] * q[:, n-1] * Gc[:, 0], ... , Gs[:, 0:p] * q[:, n-1] * Gc[:, m-1]]
        qeqv = rot.fixQuat(
            dot(qsym_s, qeqv).transpose(2, 0, 1).reshape(p*m*n, 4).T
        )

        raise NotImplementedError

    # debug
    assert qr.shape[1] == n, 'oops, something wrong here with your reshaping'

    if qdims == 3:
        qr = qr.T.reshape(l3, n3, 4).transpose(0, 2, 1)

    return qr


def ltypeOfLaueGroup(tag):
    """
    See quatOfLaueGroup
    """

    if not isinstance(tag, str):
        raise RuntimeError("entered flag is not a string!")

    if tag.lower() == 'ci' or tag.lower() == 's2':
        ltype = 'triclinic'
    elif tag.lower() == 'c2h':
        ltype = 'monoclinic'
    elif tag.lower() == 'd2h' or tag.lower() == 'vh':
        ltype = 'orthorhombic'
    elif tag.lower() == 'c4h' or tag.lower() == 'd4h':
        ltype = 'tetragonal'
    elif tag.lower() == 'c3i' or tag.lower() == 's6' or tag.lower() == 'd3d':
        ltype = 'trigonal'
    elif tag.lower() == 'c6h' or tag.lower() == 'd6h':
        ltype = 'hexagonal'
    elif tag.lower() == 'th' or tag.lower() == 'oh':
        ltype = 'cubic'
    else:
        raise RuntimeError(
            "unrecognized symmetry group.  "
            + "See ''help(quatOfLaueGroup)'' for a list of valid options.  "
            + "Oh, and have a great day ;-)"
        )

    return ltype


def quatOfLaueGroup(tag):
    """
    Generate quaternion representation for the specified Laue group.

    USAGE:

         qsym = quatOfLaueGroup(schoenfliesTag)

    INPUTS:

         1) schoenfliesTag 1 x 1, a case-insensitive string representing the
         Schoenflies symbol for the desired Laue group.  The 14 available
         choices are:

              Class           Symbol      n
             -------------------------------
              Triclinic       Ci (S2)     1
              Monoclinic      C2h         2
              Orthorhombic    D2h (Vh)    4
              Tetragonal      C4h         4
                              D4h         8
              Trigonal        C3i (S6)    3
                              D3d         6
              Hexagonal       C6h         6
                              D6h         12
              Cubic           Th          12
                              Oh          24

    OUTPUTS:

         1) qsym is (4, n) the quaterions associated with each element of the
         chosen symmetry group having n elements (dep. on group -- see INPUTS
         list above).

    NOTES:

         *) The conventions used for assigning a RHON basis, {x1, x2, x3}, to
         each point group are consistent with those published in Appendix B
         of [1].

    REFERENCES:

         [1] Nye, J. F., ``Physical Properties of Crystals: Their
         Representation by Tensors and Matrices'', Oxford University Press,
         1985. ISBN 0198511655
    """

    if not isinstance(tag, str):
        raise RuntimeError("entered flag is not a string!")

    if tag.lower() == 'ci' or tag.lower() == 's2':
        # TRICLINIC
        angleAxis = vstack([0., 1., 0., 0.])  # identity
    elif tag.lower() == 'c2h':
        # MONOCLINIC
        angleAxis = c_[
            [0.,   1,   0,   0],  # identity
            [pi,   0,   1,   0],  # twofold about 010 (x2)
            ]
    elif tag.lower() == 'd2h' or tag.lower() == 'vh':
        # ORTHORHOMBIC
        angleAxis = c_[
            [0.,   1,   0,   0],  # identity
            [pi,   1,   0,   0],  # twofold about 100
            [pi,   0,   1,   0],  # twofold about 010
            [pi,   0,   0,   1],  # twofold about 001
            ]
    elif tag.lower() == 'c4h':
        # TETRAGONAL (LOW)
        angleAxis = c_[
            [0.0,       1,    0,    0],  # identity
            [piby2,     0,    0,    1],  # fourfold about 001 (x3)
            [pi,        0,    0,    1],  #
            [piby2*3,   0,    0,    1],  #
            ]
    elif tag.lower() == 'd4h':
        # TETRAGONAL (HIGH)
        angleAxis = c_[
            [0.0,       1,    0,    0],  # identity
            [piby2,     0,    0,    1],  # fourfold about 0  0  1 (x3)
            [pi,        0,    0,    1],  #
            [piby2*3,   0,    0,    1],  #
            [pi,        1,    0,    0],  # twofold about  1  0  0 (x1)
            [pi,        0,    1,    0],  # twofold about  0  1  0 (x2)
            [pi,        1,    1,    0],  # twofold about  1  1  0
            [pi,       -1,    1,    0],  # twofold about -1  1  0
            ]
    elif tag.lower() == 'c3i' or tag.lower() == 's6':
        # TRIGONAL (LOW)
        angleAxis = c_[
            [0.0,       1,    0,    0],  # identity
            [piby3*2,   0,    0,    1],  # threefold about 0001 (x3,c)
            [piby3*4,   0,    0,    1],  #
            ]
    elif tag.lower() == 'd3d':
        # TRIGONAL (HIGH)
        angleAxis = c_[
            [0.0,       1,     0,    0],  # identity
            [piby3*2,   0,     0,    1],  # threefold about 0001 (x3,c)
            [piby3*4,   0,     0,    1],  #
            [pi,        1,     0,    0],  # twofold about  2 -1 -1  0 (x1,a1)
            [pi,       -0.5,   sq3by2,  0],  # twofold about -1  2 -1  0 (a2)
            [pi,       -0.5,  -sq3by2,  0],  # twofold about -1 -1  2  0 (a3)
            ]
    elif tag.lower() == 'c6h':
        # HEXAGONAL (LOW)
        angleAxis = c_[
            [0.0,       1,     0,    0],  # identity
            [piby3,     0,     0,    1],  # sixfold about 0001 (x3,c)
            [piby3*2,   0,     0,    1],  #
            [pi,        0,     0,    1],  #
            [piby3*4,   0,     0,    1],  #
            [piby3*5,   0,     0,    1],  #
            ]
    elif tag.lower() == 'd6h':
        # HEXAGONAL (HIGH)
        angleAxis = c_[
            [0.0,       1,       0,       0],  # identity
            [piby3,     0,       0,       1],  # sixfold about  0  0  1 (x3,c)
            [piby3*2,   0,       0,       1],  #
            [pi,        0,       0,       1],  #
            [piby3*4,   0,       0,       1],  #
            [piby3*5,   0,       0,       1],  #
            [pi,        1,       0,       0],  # twofold about  2 -1  0 (x1,a1)
            [pi,       -0.5,     sq3by2,  0],  # twofold about -1  2  0 (a2)
            [pi,       -0.5,    -sq3by2,  0],  # twofold about -1 -1  0 (a3)
            [pi,        sq3by2,  0.5,     0],  # twofold about  1  0  0
            [pi,        0,       1,       0],  # twofold about -1  1  0 (x2)
            [pi,       -sq3by2,  0.5,     0],  # twofold about  0 -1  0
            ]
    elif tag.lower() == 'th':
        # CUBIC (LOW)
        angleAxis = c_[
            [0.0,       1,    0,    0],  # identity
            [pi,        1,    0,    0],  # twofold about    1  0  0 (x1)
            [pi,        0,    1,    0],  # twofold about    0  1  0 (x2)
            [pi,        0,    0,    1],  # twofold about    0  0  1 (x3)
            [piby3*2,   1,    1,    1],  # threefold about  1  1  1
            [piby3*4,   1,    1,    1],  #
            [piby3*2,  -1,    1,    1],  # threefold about -1  1  1
            [piby3*4,  -1,    1,    1],  #
            [piby3*2,  -1,   -1,    1],  # threefold about -1 -1  1
            [piby3*4,  -1,   -1,    1],  #
            [piby3*2,   1,   -1,    1],  # threefold about  1 -1  1
            [piby3*4,   1,   -1,    1],  #
            ]
    elif tag.lower() == 'oh':
        # CUBIC (HIGH)
        angleAxis = c_[
            [0.0,       1,    0,    0],  # identity
            [piby2,     1,    0,    0],  # fourfold about   1  0  0 (x1)
            [pi,        1,    0,    0],  #
            [piby2*3,   1,    0,    0],  #
            [piby2,     0,    1,    0],  # fourfold about   0  1  0 (x2)
            [pi,        0,    1,    0],  #
            [piby2*3,   0,    1,    0],  #
            [piby2,     0,    0,    1],  # fourfold about   0  0  1 (x3)
            [pi,        0,    0,    1],  #
            [piby2*3,   0,    0,    1],  #
            [piby3*2,   1,    1,    1],  # threefold about  1  1  1
            [piby3*4,   1,    1,    1],  #
            [piby3*2,  -1,    1,    1],  # threefold about -1  1  1
            [piby3*4,  -1,    1,    1],  #
            [piby3*2,  -1,   -1,    1],  # threefold about -1 -1  1
            [piby3*4,  -1,   -1,    1],  #
            [piby3*2,   1,   -1,    1],  # threefold about  1 -1  1
            [piby3*4,   1,   -1,    1],  #
            [pi,        1,    1,    0],  # twofold about    1  1  0
            [pi,       -1,    1,    0],  # twofold about   -1  1  0
            [pi,        1,    0,    1],  # twofold about    1  0  1
            [pi,        0,    1,    1],  # twofold about    0  1  1
            [pi,       -1,    0,    1],  # twofold about   -1  0  1
            [pi,        0,   -1,    1],  # twofold about    0 -1  1
            ]
    else:
        raise RuntimeError(
            "unrecognized symmetry group.  "
            + "See ``help(quatOfLaueGroup)'' for a list of valid options.  "
            + "Oh, and have a great day ;-)"
        )

    angle = angleAxis[0, ]
    axis = angleAxis[1:, ]

    #  Note: Axis does not need to be normalized in call to quatOfAngleAxis
    #  05/01/2014 JVB -- made output a contiguous C-ordered array
    qsym = array(rot.quatOfAngleAxis(angle, axis).T, order='C').T

    return qsym

def GeneratorString(sgnum):
    '''
    these rhombohedral space groups have a hexagonal setting
    with different symmetry matrices and generator strings
    146: 231
    148: 232
    ...
    and so on
    '''
    sg = sgnum-1
    # sgdict = {146:231, 148:232, 155:233, 160:234, 161:235, 166:236, 167:237}
    # if(sgnum in sgdict):
    #     sg = sgdict[sgnum]-1

    return constants.SYM_GL[sg]

def MakeGenerators(genstr, setting):

    t = 'aOOO'
    mat = SYM_fillgen(t)
    genmat = mat

    # genmat[0,:,:] = constants.SYM_GENERATORS['a']
    centrosymmetric = False

    # check if space group has inversion symmetry
    if(genstr[0] == '1'):
        t = 'hOOO'
        mat = SYM_fillgen(t)
        genmat = np.concatenate((genmat, mat))
        centrosymmetric = True
        
    n = int(genstr[1])
    if(n > 0):
        for i in range(n):
            istart = 2 + i * 4
            istop  = 2 + (i+1) * 4

            t = genstr[istart:istop]

            mat = SYM_fillgen(t)
            genmat = np.concatenate((genmat, mat))
    else:
        istop = 2
    '''
    if there is an alternate setting for this space group
    check if the alternate setting needs to be used
    '''
    if(genstr[istop] != '0'):
        if(setting != 0):
            t = genstr[istop+1:istop+4]
            trans = np.array([constants.SYM_GENERATORS[t[0]],\
                              constants.SYM_GENERATORS[t[1]],\
                              constants.SYM_GENERATORS[t[2]]
                              ])
            for i in range(genmat.shape[0]):
                genmat[i,0:3,3] -= trans

    return genmat, centrosymmetric

def SYM_fillgen(t):
    mat = np.zeros([4,4])
    mat[3,3] = 1.

    mat[0:3,0:3] = constants.SYM_GENERATORS[t[0]]
    mat[0:3,3] = np.array([constants.SYM_GENERATORS[t[1]],\
                           constants.SYM_GENERATORS[t[2]],\
                           constants.SYM_GENERATORS[t[3]]
                           ])

    mat = np.broadcast_to(mat, [1,4,4])
    return mat

def GenerateSGSym(sgnum, setting=0):

    '''
    get the generators for a space group using the
    generator string
    '''
    genstr = GeneratorString(sgnum)
    genmat, centrosymmetric = MakeGenerators(genstr, setting)
    symmorphic = False
    if(sgnum in constants.sgnum_symmorphic):
        symmorphic = True
    '''
    use the generator string to get the rest of the
    factor group

    genmat has shape ngenerators x 4 x 4
    '''
    nsym = genmat.shape[0]

    SYM_SG = genmat

    '''
    generate the factor group
    '''

    k1 = 0
    while k1 < nsym:

        g1 = np.squeeze(SYM_SG[k1,:,:])
        k2 = k1

        while k2 < nsym:
            g2 = np.squeeze(SYM_SG[k2,:,:])
            gnew = np.dot(g1, g2)

            # only fractional parts
            frac = np.modf(gnew[0:3,3])[0]
            frac[frac < 0.] += 1.
            frac[np.abs(frac) < 1E-5] = 0.0

            gnew[0:3,3] = frac

            if(isnew(gnew, SYM_SG)):
                gnew = np.broadcast_to(gnew, [1,4,4])
                SYM_SG = np.concatenate((SYM_SG, gnew))
                nsym += 1

                if (nsym >= 192):
                    k2 = nsym
                    k1 = nsym

            k2 += 1
        k1 += 1

    SYM_PG_d = GeneratePGSym(SYM_SG)
    SYM_PG_d_laue = GeneratePGSym_Laue(SYM_PG_d)

    for s in SYM_PG_d:
        if(np.allclose(-np.eye(3),s)):
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
    SYM_PG_d = SYM_SG[0,0:3,0:3]
    SYM_PG_d = np.broadcast_to(SYM_PG_d,[1,3,3])

    for i in range(1,nsgsym):
        g = SYM_SG[i,:,:]
        t = g[0:3,3]
        g = g[0:3,0:3]
        if(isnew(g,SYM_PG_d)):
            g = np.broadcast_to(g,[1,3,3])
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
        if(np.allclose(s,-np.eye(3))):
            return SYM_PG_d

    '''
    if we get here, then the inversion symmetry is not present
    add the inversion symmetry
    '''
    SYM_PG_d_laue = SYM_PG_d
    g = np.broadcast_to(-np.eye(3).astype(np.int32),[1,3,3])
    SYM_PG_d_laue = np.concatenate((SYM_PG_d_laue, g))

    '''
    now go through the group actions and see if its a new matrix
    if it is then add it to the group
    '''
    nsym = SYM_PG_d_laue.shape[0]
    k1 = 0
    while k1 < nsym:
        g1 = np.squeeze(SYM_PG_d_laue[k1,:,:])
        k2 = k1
        while k2 < nsym:
            g2 = np.squeeze(SYM_PG_d_laue[k2,:,:])
            gnew = np.dot(g1, g2)

            if(isnew(gnew, SYM_PG_d_laue)):
                gnew = np.broadcast_to(gnew, [1,3,3])
                SYM_PG_d_laue = np.concatenate((SYM_PG_d_laue, gnew))
                nsym += 1

                if (nsym >= 48):
                    k2 = nsym
                    k1 = nsym

            k2 += 1
        k1 += 1

    return SYM_PG_d_laue

def isnew(mat, sym_mats):
    isnew = True
    for g in sym_mats:
        diff = np.sum(np.abs(mat-g))
        if(diff < 1E-5):
            isnew = False
            break
    return isnew

def latticeType(sgnum):

    if(sgnum <= 2):
        return 'triclinic'
    elif(sgnum > 2 and sgnum <= 15):
        return 'monoclinic'
    elif(sgnum > 15 and sgnum <= 74):
        return 'orthorhombic'
    elif(sgnum > 74 and sgnum <= 142):
        return 'tetragonal'
    elif(sgnum > 142 and sgnum <= 167):
        return 'trigonal'
    elif(sgnum > 167 and sgnum <= 194):
        return 'hexagonal'
    elif(sgnum > 194 and sgnum <=230):
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
    SYM_GEN_PG = np.zeros([ngen,3,3])

    for i in range(ngen):
        s = pggenstr[i+1]
        SYM_GEN_PG[i,:,:] = constants.SYM_GENERATORS[s]

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
        g1 = np.squeeze(SYM_GEN_PG[k1,:,:])
        k2 = k1
        while k2 < nsym:
            g2 = np.squeeze(SYM_GEN_PG[k2,:,:])
            gnew = np.dot(g1, g2)

            if(isnew(gnew, SYM_GEN_PG)):
                gnew = np.broadcast_to(gnew, [1,3,3])
                SYM_GEN_PG = np.concatenate((SYM_GEN_PG, gnew))
                nsym += 1

                if (nsym >= 48):
                    k2 = nsym
                    k1 = nsym

            k2 += 1
        k1 += 1

    SYM_GEN_PG[np.abs(SYM_GEN_PG) < eps] = 0.

    return SYM_GEN_PG