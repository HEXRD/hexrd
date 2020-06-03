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
import re
import copy
from math import pi

import numpy as np
import csv
import os

try:
    from scipy import constants as C
except(ImportError):
    raise RuntimeError("scipy must be recent enough to have constants module")

from hexrd.matrixutil import unitVector, sum
from hexrd.rotations import rotMatOfExpMap, mapAngle, applySym
from hexrd import symmetry
from hexrd.transforms import xfcapi
from hexrd import valunits
from hexrd.valunits import toFloat
from hexrd.constants import d2r, r2d, sqrt3by2, sqrt_epsf

"""module vars"""

# units
dUnit = 'angstrom'
outputDegrees = False
outputDegrees_bak = outputDegrees


def hklToStr(x):
    return re.sub('\[|\]|\(|\)', '', str(x))


def tempSetOutputDegrees(val):
    global outputDegrees, outputDegrees_bak
    outputDegrees_bak = outputDegrees
    outputDegrees = val
    return


def revertOutputDegrees():
    global outputDegrees, outputDegrees_bak
    outputDegrees = outputDegrees_bak
    return


def cosineXform(a, b, c):
    """
    Spherical trig transform to take alpha, beta, gamma to expressions
    for cos(alpha*).  See ref below.

    [1] R. J. Neustadt, F. W. Cagle, Jr., and J. Waser, ``Vector algebra and
        the relations between direct and reciprocal lattice quantities''. Acta
        Cryst. (1968), A24, 247--248

    """
    cosar = (np.cos(b)*np.cos(c) - np.cos(a)) / (np.sin(b)*np.sin(c))
    sinar = np.sqrt(1 - cosar**2)
    return cosar, sinar


def processWavelength(arg):
    """
    Convert an energy value to a wavelength.  If argument has units of length
    or energy, will convert to globally specified unit type for wavelength
    (dUnit).  If argument is a scalar, assumed input units are keV.
    """
    if hasattr(arg, 'getVal'):
        if arg.isLength():
            retval = arg.getVal(dUnit)
        elif arg.isEnergy():
            speed = C.c
            planck = C.h
            e = arg.getVal('J')
            retval = valunits.valWUnit(
                'wavelength', 'length', planck*speed/e, 'm').getVal(dUnit)
        else:
            raise RuntimeError('do not know what to do with '+str(arg))
    else:
        keV2J = 1.e3*C.e
        e = keV2J * arg
        retval = valunits.valWUnit(
            'wavelength', 'length', C.h*C.c/e, 'm').getVal(dUnit)

    return retval


def latticeParameters(lvec):
    """
    Generates direct and reciprocal lattice vector components in a
    crystal-relative RHON basis, X. The convention for fixing X to the
    lattice is such that a || x1 and c* || x3, where a and c* are
    direct and reciprocal lattice vectors, respectively.
    """
    lnorm = np.sqrt(np.sum(lvec**2, 0))

    a = lnorm[0]
    b = lnorm[1]
    c = lnorm[2]

    ahat = lvec[:, 0]/a
    bhat = lvec[:, 1]/b
    chat = lvec[:, 2]/c

    gama = np.arccos(np.dot(ahat, bhat))
    beta = np.arccos(np.dot(ahat, chat))
    alfa = np.arccos(np.dot(bhat, chat))
    if outputDegrees:
        gama = r2d*gama
        beta = r2d*beta
        alfa = r2d*alfa

    return [a, b, c, alfa, beta, gama]


def latticePlanes(hkls, lparms,
                  ltype='cubic', wavelength=1.54059292, strainMag=None):
    """
    Generates lattice plane data in the direct lattice for a given set
    of Miller indices.  Vector components are written in the
    crystal-relative RHON basis, X. The convention for fixing X to the
    lattice is such that a || x1 and c* || x3, where a and c* are
    direct and reciprocal lattice vectors, respectively.

    USAGE:

    planeInfo = latticePlanes(hkls, lparms, **kwargs)

    INPUTS:

    1) hkls (3 x n float ndarray) is the array of Miller indices for
       the planes of interest.  The vectors are assumed to be
       concatenated along the 1-axis (horizontal).

    2) lparms (1 x m float list) is the array of lattice parameters,
       where m depends on the symmetry group (see below).

    3) The following optional keyword arguments are recognized:

       *) ltype=(string) is a string representing the symmetry type of
          the implied Laue group.  The 11 available choices are shown
          below.  The default value is 'cubic'. Note that each group
          expects a lattice parameter array of the indicated length
          and order.

          latticeType      lparms
          -----------      ------------
          'cubic'          a
          'hexagonal'      a, c
          'trigonal'       a, c
          'rhombohedral'   a, alpha (in degrees)
          'tetragonal'     a, c
          'orthorhombic'   a, b, c
          'monoclinic'     a, b, c, beta (in degrees)
          'triclinic'      a, b, c, alpha, beta, gamma (in degrees)

       *) wavelength=<float> is a value represented the wavelength in
          Angstroms to calculate bragg angles for.  The default value
          is for Cu K-alpha radiation (1.54059292 Angstrom)

       *) strainMag=None

    OUTPUTS:

    1) planeInfo is a dictionary containing the following keys/items:

       normals   (3, n) double array    array of the components to the
                                        unit normals for each {hkl} in
                                        X (horizontally concatenated)

       dspacings (n,  ) double array    array of the d-spacings for
                                        each {hkl}

       2thetas   (n,  ) double array    array of the Bragg angles for
                                        each {hkl} relative to the
                                        specified wavelength

    NOTES:

    *) This function is effectively a wrapper to 'latticeVectors'.
       See 'help(latticeVectors)' for additional info.

    *) Lattice plane d-spacings are calculated from the reciprocal
       lattice vectors specified by {hkl} as shown in Appendix 1 of
       [1].

    REFERENCES:

    [1] B. D. Cullity, ``Elements of X-Ray Diffraction, 2
        ed.''. Addison-Wesley Publishing Company, Inc., 1978. ISBN
        0-201-01174-3

    """
    location = 'latticePlanes'

    assert hkls.shape[0] == 3, \
        "hkls aren't column vectors in call to '%s'!" % location

    tag = ltype
    wlen = wavelength

    # get B
    L = latticeVectors(lparms, tag)

    # get G-vectors -- reciprocal vectors in crystal frame
    G = np.dot(L['B'], hkls)

    # magnitudes
    d = 1 / np.sqrt(np.sum(G**2, 0))

    aconv = 1.
    if outputDegrees:
        aconv = r2d

    # two thetas
    sth  = wlen / 2. / d
    mask = (np.abs(sth) < 1.)
    tth = np.zeros(sth.shape)

    tth[~mask] = np.nan
    tth[mask]  = aconv * 2. * np.arcsin(sth[mask])


    p = dict(normals=unitVector(G),
             dspacings=d,
             tThetas=tth)

    if strainMag is not None:

      p['tThetasLo'] = np.zeros(sth.shape)
      p['tThetasHi'] = np.zeros(sth.shape)

      mask = ( ( np.abs(wlen / 2. / (d * (1. + strainMag))) < 1.) & 
               ( np.abs(wlen / 2. / (d * (1. - strainMag))) < 1.0 ) )

      p['tThetasLo'][~mask] = np.nan
      p['tThetasHi'][~mask] = np.nan

      p['tThetasLo'][mask] = aconv * 2 * np.arcsin(wlen/2./(d[mask]*(1. + strainMag)))
      p['tThetasHi'][mask] = aconv * 2 * np.arcsin(wlen/2./(d[mask]*(1. - strainMag)))

    return p


def latticeVectors(lparms, tag='cubic', radians=False, debug=False):
    """
    Generates direct and reciprocal lattice vector components in a
    crystal-relative RHON basis, X. The convention for fixing X to the
    lattice is such that a || x1 and c* || x3, where a and c* are
    direct and reciprocal lattice vectors, respectively.

    USAGE:

    lattice = LatticeVectors(lparms, <symmTag>)

    INPUTS:

    1) lparms (1 x n float list) is the array of lattice parameters,
       where n depends on the symmetry group (see below).

    2) symTag (string) is a case-insensitive string representing the
       symmetry type of the implied Laue group.  The 11 available choices
       are shown below.  The default value is 'cubic'. Note that each
       group expects a lattice parameter array of the indicated length
       and order.

       latticeType      lparms
       -----------      ------------
       'cubic'          a
       'hexagonal'      a, c
       'trigonal'       a, c
       'rhombohedral'   a, alpha (in degrees)
       'tetragonal'     a, c
       'orthorhombic'   a, b, c
       'monoclinic'     a, b, c, beta (in degrees)
       'triclinic'      a, b, c, alpha, beta, gamma (in degrees)

    OUTPUTS:

    1) lattice is a dictionary containing the following keys/items:

       F         (3, 3) double array    transformation matrix taking
                                        componenents in the direct
                                        lattice (i.e. {uvw}) to the
                                        reference, X

       B         (3, 3) double array    transformation matrix taking
                                        componenents in the reciprocal
                                        lattice (i.e. {hkl}) to X

       BR        (3, 3) double array    transformation matrix taking
                                        componenents in the reciprocal
                                        lattice to the Fable reference
                                        frame (see notes)

       U0        (3, 3) double array    transformation matrix
                                        (orthogonal) taking
                                        componenents in the
                                        Fable reference frame to X

       vol       double                 the unit cell volume


       dparms    (6, ) double list      the direct lattice parameters:
                                        [a b c alpha beta gamma]

       rparms    (6, ) double list      the reciprocal lattice
                                        parameters:
                                        [a* b* c* alpha* beta* gamma*]

    NOTES:

    *) The conventions used for assigning a RHON basis,
       X -> {x1, x2, x3}, to each point group are consistent with
       those published in Appendix B of [1]. Namely: a || x1 and
       c* || x3.  This differs from the convention chosen by the Fable
       group, where a* || x1 and c || x3 [2].

    *) The unit cell angles are defined as follows:
       alpha=acos(b'*c/|b||c|), beta=acos(c'*a/|c||a|), and
       gamma=acos(a'*b/|a||b|).

    *) The reciprocal lattice vectors are calculated using the
       crystallographic convention, where the prefactor of 2*pi is
       omitted. In this convention, the reciprocal lattice volume is
       1/V.

    *) Several relations from [3] were employed in the component
       calculations.

    REFERENCES:

    [1] J. F. Nye, ``Physical Properties of Crystals: Their
        Representation by Tensors and Matrices''. Oxford University
        Press, 1985. ISBN 0198511655

    [2] E. M. Lauridsen, S. Schmidt, R. M. Suter, and H. F. Poulsen,
        ``Tracking: a method for structural characterization of grains
        in powders or polycrystals''. J. Appl. Cryst. (2001). 34,
        744--750

    [3] R. J. Neustadt, F. W. Cagle, Jr., and J. Waser, ``Vector
        algebra and the relations between direct and reciprocal
        lattice quantities''. Acta Cryst. (1968), A24, 247--248


    """

    # build index for sorting out lattice parameters
    lattStrings = [
        'cubic',
        'hexagonal',
        'trigonal',
        'rhombohedral',
        'tetragonal',
        'orthorhombic',
        'monoclinic',
        'triclinic'
        ]

    if radians:
        aconv = 1.
    else:
        aconv = pi/180.  # degToRad
    deg90 = pi/2.
    deg120 = 2.*pi/3.
    #
    if tag == lattStrings[0]:
        # cubic
        cellparms = np.r_[np.tile(lparms[0], (3,)), deg90*np.ones((3,))]
    elif tag == lattStrings[1] or tag == lattStrings[2]:
        # hexagonal | trigonal (hex indices)
        cellparms = np.r_[lparms[0], lparms[0], lparms[1],
                          deg90, deg90, deg120]
    elif tag == lattStrings[3]:
        # rhombohedral
        cellparms = np.r_[np.tile(lparms[0], (3,)),
                          np.tile(aconv*lparms[1], (3,))]
    elif tag == lattStrings[4]:
        # tetragonal
        cellparms = np.r_[lparms[0], lparms[0], lparms[1],
                          deg90, deg90, deg90]
    elif tag == lattStrings[5]:
        # orthorhombic
        cellparms = np.r_[lparms[0], lparms[1], lparms[2],
                          deg90, deg90, deg90]
    elif tag == lattStrings[6]:
        # monoclinic
        cellparms = np.r_[lparms[0], lparms[1], lparms[2],
                          deg90, aconv*lparms[3], deg90]
    elif tag == lattStrings[7]:
        # triclinic
        # FIXME: fixed DP 2/24/16
        cellparms = np.r_[lparms[0], lparms[1], lparms[2],
                          aconv*lparms[3], aconv*lparms[4], aconv*lparms[5]]
    else:
        raise RuntimeError('lattice tag \'%s\' is not recognized' % (tag))

    if debug:
        print((str(cellparms[0:3]) + ' ' + str(r2d*cellparms[3:6])))
    alfa = cellparms[3]
    beta = cellparms[4]
    gama = cellparms[5]

    cosalfar, sinalfar = cosineXform(alfa, beta, gama)

    a = cellparms[0]*np.r_[1, 0, 0]
    b = cellparms[1]*np.r_[np.cos(gama), np.sin(gama), 0]
    c = cellparms[2]*np.r_[np.cos(beta),
                           -cosalfar*np.sin(beta),
                           sinalfar*np.sin(beta)]

    ad = np.sqrt(sum(a**2))
    bd = np.sqrt(sum(b**2))
    cd = np.sqrt(sum(c**2))

    # Cell volume
    V = np.dot(a, np.cross(b, c))

    # F takes components in the direct lattice to X
    F = np.c_[a, b, c]

    # Reciprocal lattice vectors
    astar = np.cross(b, c)/V
    bstar = np.cross(c, a)/V
    cstar = np.cross(a, b)/V

    # and parameters
    ar = np.sqrt(sum(astar**2))
    br = np.sqrt(sum(bstar**2))
    cr = np.sqrt(sum(cstar**2))

    alfar = np.arccos(np.dot(bstar, cstar)/br/cr)
    betar = np.arccos(np.dot(cstar, astar)/cr/ar)
    gamar = np.arccos(np.dot(astar, bstar)/ar/br)

    # B takes components in the reciprocal lattice to X
    B = np.c_[astar, bstar, cstar]

    cosalfar2, sinalfar2 = cosineXform(alfar, betar, gamar)

    afable = ar*np.r_[1, 0, 0]
    bfable = br*np.r_[np.cos(gamar), np.sin(gamar), 0]
    cfable = cr*np.r_[np.cos(betar),
                      -cosalfar2*np.sin(betar),
                      sinalfar2*np.sin(betar)]

    BR = np.c_[afable, bfable, cfable]
    U0 = np.dot(B, np.linalg.inv(BR))
    if outputDegrees:
        dparms = np.r_[ad, bd, cd, r2d*np.r_[alfa, beta, gama]]
        rparms = np.r_[ar, br, cr, r2d*np.r_[alfar, betar, gamar]]
    else:
        dparms = np.r_[ad, bd, cd, np.r_[alfa, beta, gama]]
        rparms = np.r_[ar, br, cr, np.r_[alfar, betar, gamar]]

    L = {'F': F,
         'B': B,
         'BR': BR,
         'U0': U0,
         'vol': V,
         'dparms': dparms,
         'rparms': rparms}

    return L


def hexagonalIndicesFromRhombohedral(hkl):
    """
    converts rhombohedral hkl to hexagonal indices
    """
    HKL = np.zeros((3, hkl.shape[1]), dtype='int')

    HKL[0, :] = hkl[0, :] - hkl[1, :]
    HKL[1, :] = hkl[1, :] - hkl[2, :]
    HKL[2, :] = hkl[0, :] + hkl[1, :] + hkl[2, :]

    return HKL


def rhombohedralIndicesFromHexagonal(HKL):
    """
    converts hexagonal hkl to rhombohedral indices
    """
    hkl = np.zeros((3, HKL.shape[1]), dtype='int')

    hkl[0, :] = 2 * HKL[0, :] + HKL[1, :] + HKL[2, :]
    hkl[1, :] = -HKL[0, :] + HKL[1, :] + HKL[2, :]
    hkl[2, :] = -HKL[0, :] - 2 * HKL[1, :] + HKL[2, :]

    hkl = hkl / 3.
    return hkl


def rhombohedralParametersFromHexagonal(a_h, c_h):
    """
    converts hexagonal lattice parameters (a, c) to rhombohedral
    lattice parameters (a, alpha)
    """
    a_r = np.sqrt(3 * a_h**2 + c_h**2) / 3.
    alfa_r = 2 * np.arcsin(3. / (2 * np.sqrt(3 + (c_h / a_h)**2)))
    if outputDegrees:
        alfa_r = r2d * alfa_r
    return a_r, alfa_r


def millerBravaisDirectionToVector(dir_ind, a=1., c=1.):
    """
    converts direction indices into a unit vector in the crystal frame

    INPUT [uv.w] the Miller-Bravais convention in the hexagonal basis
    {a1, a2, a3, c}.  The basis for the output, {o1, o2, o3}, is
    chosen such that

    o1 || a1
    o3 || c
    o2 = o3 ^ o1
    """
    dir_ind = np.atleast_2d(dir_ind)
    num_in = len(dir_ind)
    u = dir_ind[:, 0]
    v = dir_ind[:, 1]
    w = dir_ind[:, 2]
    return unitVector(
        np.vstack([1.5*u*a, sqrt3by2*(2.*v + u)*a, w*c]).reshape(3, num_in)
    )


class PlaneData(object):
    """
    Careful with ordering: Outputs are ordered by the 2-theta for the
    hkl unless you get self.__hkls directly, and this order can change
    with changes in lattice parameters (lparms); setting and getting
    exclusions works on the current hkl ordering, not the original
    ordering (in self.__hkls), but exclusions are stored in the
    original ordering in case the hkl ordering does change with
    lattice parameters

    if not None, tThWidth takes priority over strainMag in setting
    two-theta ranges; changing strainMag automatically turns off
    tThWidth
    """

    def __init__(self,
                 hkls,
                 *args,
                 **kwargs):

        self.phaseID = None
        self.__doTThSort = True
        self.__exclusions = None
        self.__tThMax = None
        #
        if len(args) == 4:
            lparms, laueGroup, wavelength, strainMag = args
            tThWidth = None
            self.__wavelength = processWavelength(wavelength)
            self.__lparms = self.__parseLParms(lparms)
        elif len(args) == 1 and hasattr(args[0], 'getParams'):
            other = args[0]
            lparms, laueGroup, wavelength, strainMag, tThWidth = \
                other.getParams()
            self.__wavelength = wavelength
            self.__lparms = lparms
            self.phaseID = other.phaseID
            self.__doTThSort = other.__doTThSort
            self.__exclusions = other.__exclusions
            self.__tThMax = other.__tThMax
            if hkls is None:
                hkls = other.__hkls
        else:
            raise NotImplementedError('args : '+str(args))

        self.__laueGroup = laueGroup
        self.__qsym = symmetry.quatOfLaueGroup(self.__laueGroup)
        self.__hkls = copy.deepcopy(hkls)
        self.__strainMag = strainMag
        self.__structFact = np.ones(self.__hkls.shape[1])
        self.tThWidth = tThWidth

        # ... need to implement tThMin too
        if 'phaseID' in kwargs:
            self.phaseID = kwargs.pop('phaseID')
        if 'doTThSort' in kwargs:
            self.__doTThSort = kwargs.pop('doTThSort')
        if 'exclusions' in kwargs:
            self.__exclusions = kwargs.pop('exclusions')
        if 'tThMax' in kwargs:
            self.__tThMax = toFloat(kwargs.pop('tThMax'), 'radians')
        if 'tThWidth' in kwargs:
            self.tThWidth = kwargs.pop('tThWidth')
        if len(kwargs) > 0:
            raise RuntimeError('have unparsed keyword arguments with keys: '
                               + str(list(kwargs.keys())))

        self.__calc()

        return

    def __calc(self):
        symmGroup = symmetry.ltypeOfLaueGroup(self.__laueGroup)
        latPlaneData, latVecOps, hklDataList = PlaneData.makePlaneData(
            self.__hkls, self.__lparms, self.__qsym,
            symmGroup, self.__strainMag, self.wavelength)
        'sort by tTheta'
        tThs = np.array(
            [hklDataList[iHKL]['tTheta'] for iHKL in range(len(hklDataList))]
        )
        if self.__doTThSort:
            # sorted hkl -> __hkl
            # __hkl -> sorted hkl
            self.tThSort = np.argsort(tThs)
            self.tThSortInv = np.empty(len(hklDataList), dtype=int)
            self.tThSortInv[self.tThSort] = np.arange(len(hklDataList))
            self.hklDataList = [hklDataList[iHKL] for iHKL in self.tThSort]
        else:
            self.tThSort = np.arange(len(hklDataList))
            self.tThSortInv = np.arange(len(hklDataList))
            self.hklDataList = hklDataList
        self.__latVecOps = latVecOps
        self.nHKLs = len(self.getHKLs())
        return

    def __str__(self):
        s = '========== plane data ==========\n'
        s += 'lattice parameters:\n   ' + str(self.lparms) + '\n'
        s += 'two theta width: (%s)\n' % str(self.tThWidth)
        s += 'strain magnitude: (%s)\n' % str(self.strainMag)
        s += 'beam energy (%s)\n' % str(self.wavelength)
        s += 'hkls: (%d)\n' % self.nHKLs
        s += str(self.getHKLs())
        return s

    def getNHKLs(self):
        return self.nHKLs

    def getPhaseID(self):
        'may return None if not set'
        return self.phaseID

    def getParams(self):
        return (self.__lparms, self.__laueGroup, self.__wavelength,
                self.__strainMag, self.tThWidth)

    def getNhklRef(self):
        'does not use exclusions or the like'
        retval = len(self.hklDataList)
        return retval

    def get_hkls(self):
        """
        do not do return self.__hkls, as everywhere else hkls are returned
        in 2-theta order; transpose is to comply with lparm convention
        """
        return self.getHKLs().T

    def set_hkls(self, hkls):
        raise RuntimeError('for now, not allowing hkls to be reset')
        # self.__exclusions = None
        # self.__hkls = hkls
        # self.__calc()
        return
    hkls = property(get_hkls, set_hkls, None)

    def get_tThMax(self):
        return self.__tThMax

    def set_tThMax(self, tThMax):
        self.__tThMax = toFloat(tThMax, 'radians')
        # self.__calc() # no need to redo calc for tThMax
        return
    tThMax = property(get_tThMax, set_tThMax, None)

    def get_exclusions(self):
        retval = np.zeros(self.getNhklRef(), dtype=bool)
        if self.__exclusions is not None:
            # report in current hkl ordering
            retval[:] = self.__exclusions[self.tThSortInv]
        if self.__tThMax is not None:
            for iHKLr, hklData in enumerate(self.hklDataList):
                if hklData['tTheta'] > self.__tThMax:
                    retval[iHKLr] = True
        return retval

    def set_exclusions(self, exclusions):
        excl = np.zeros(len(self.hklDataList), dtype=bool)
        if exclusions is not None:
            exclusions = np.atleast_1d(exclusions)
            if len(exclusions) == len(self.hklDataList):
                assert exclusions.dtype == 'bool', \
                    'exclusions should be bool if full length'
                # convert from current hkl ordering to __hkl ordering
                excl[:] = exclusions[self.tThSort]
            else:
                if len(exclusions.shape) == 1:
                    # treat exclusions as indices
                    excl[self.tThSort[exclusions]] = True
                elif len(exclusions.shape) == 2:
                    raise NotImplementedError(
                        'have not yet coded treating exclusions as ranges'
                    )
                else:
                    raise RuntimeError(
                        'do not now what to do with exclusions with shape '
                        + str(exclusions.shape)
                    )
        self.__exclusions = excl
        self.nHKLs = np.sum(np.logical_not(self.__exclusions))
        return
    exclusions = property(get_exclusions, set_exclusions, None)

    def get_lparms(self):
        return self.__lparms

    def __parseLParms(self, lparms):
        lparmsDUnit = []
        for lparmThis in lparms:
            if hasattr(lparmThis, 'getVal'):
                if lparmThis.isLength():
                    lparmsDUnit.append(lparmThis.getVal(dUnit))
                elif lparmThis.isAngle():
                    # plumbing set up to default to degrees
                    # for lattice parameters
                    lparmsDUnit.append(lparmThis.getVal('degrees'))
                else:
                    raise RuntimeError(
                        'do not know what to do with '
                        + str(lparmThis)
                    )
            else:
                lparmsDUnit.append(lparmThis)
        return lparmsDUnit

    def set_lparms(self, lparms):
        self.__lparms = self.__parseLParms(lparms)
        self.__calc()
        return
    lparms = property(get_lparms, set_lparms, None)

    def get_strainMag(self):
        return self.__strainMag

    def set_strainMag(self, strainMag):
        self.__strainMag = strainMag
        self.tThWidth = None
        self.__calc()
        return
    strainMag = property(get_strainMag, set_strainMag, None)

    def get_wavelength(self):
        return self.__wavelength

    def set_wavelength(self, wavelength):
        self.__wavelength = processWavelength(wavelength)
        self.__calc()
        return
    wavelength = property(get_wavelength, set_wavelength, None)

    def get_structFact(self):
        return self.__structFact

    def set_structFact(self, structFact):
        self.__structFact = structFact

    structFact = property(get_wavelength, set_wavelength, None)

    def get_SGsym(self):
        return self.__sgsym

    def set_SGsym(self, sgsym):
        self.__sgsym = sgsym 

    SGsym = property(get_SGsym, set_SGsym, None)

    @staticmethod
    def makePlaneData(hkls, lparms, qsym, symmGroup, strainMag, wavelength):
        """
        hkls       : need to work with crystallography.latticePlanes
        lparms     : need to work with crystallography.latticePlanes
        laueGroup  : see symmetry module
        wavelength : wavelength
        strainMag  : swag of strian magnitudes
        """

        tempSetOutputDegrees(False)
        latPlaneData = latticePlanes(hkls, lparms,
                                     ltype=symmGroup,
                                     strainMag=strainMag,
                                     wavelength=wavelength)

        latVecOps = latticeVectors(lparms, symmGroup)

        hklDataList = []
        for iHKL in range(len(hkls.T)):
            # need transpose because of convention for hkls ordering

            """
            latVec = latPlaneData['normals'][:,iHKL]
            # ... if not spots, may be able to work with a subset of these
            latPlnNrmlList = applySym(
                np.c_[latVec], qsym, csFlag=True, cullPM=False
            )
            """
            # returns UN-NORMALIZED lattice plane normals
            latPlnNrmls = applySym(
                np.dot(latVecOps['B'], hkls[:, iHKL].reshape(3, 1)),
                qsym,
                csFlag=True,
                cullPM=False)

            # check for +/- in symmetry group
            latPlnNrmlsM = applySym(
                np.dot(latVecOps['B'], hkls[:, iHKL].reshape(3, 1)),
                qsym,
                csFlag=False,
                cullPM=False)

            csRefl = latPlnNrmls.shape[1] == latPlnNrmlsM.shape[1]

            # added this so that I retain the actual symmetric
            # integer hkls as well
            symHKLs = np.array(
                np.round(
                    np.dot(latVecOps['F'].T, latPlnNrmls)
                ), dtype='int'
            )

            hklDataList.append(
                dict(hklID=iHKL,
                     hkl=hkls[:, iHKL],
                     tTheta=latPlaneData['tThetas'][iHKL],
                     dSpacings=latPlaneData['dspacings'][iHKL],
                     tThetaLo=latPlaneData['tThetasLo'][iHKL],
                     tThetaHi=latPlaneData['tThetasHi'][iHKL],
                     latPlnNrmls=unitVector(latPlnNrmls),
                     symHKLs=symHKLs,
                     centrosym=csRefl
                     )
                )

        revertOutputDegrees()
        return latPlaneData, latVecOps, hklDataList

    def getLatticeType(self):
        """This is the lattice type"""
        return symmetry.ltypeOfLaueGroup(self.__laueGroup)

    def getLaueGroup(self):
        """This is the Schoenflies tag"""
        return self.__laueGroup

    def getQSym(self):
        return self.__qsym  # symmetry.quatOfLaueGroup(self.__laueGroup)

    def getPlaneSpacings(self):
        """
        gets plane spacings
        """
        dspacings = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self.__thisHKL(iHKLr):
                continue
            dspacings.append(hklData['dSpacings'])
        return dspacings

    def getPlaneNormals(self):
        """
        gets both +(hkl) and -(hkl) normals
        """
        plnNrmls = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self.__thisHKL(iHKLr):
                continue
            plnNrmls.append(hklData['latPlnNrmls'])
        return plnNrmls

    def getLatticeOperators(self):
        """
        gets lattice vector operators as a new (deepcopy)
        """
        return copy.deepcopy(self.__latVecOps)

    def setLatticeOperators(self, val):
        raise RuntimeError(
            'do not set latVecOps directly, change other things instead'
        )
    latVecOps = property(getLatticeOperators, setLatticeOperators, None)

    def __thisHKL(self, iHKLr):
        retval = True
        hklData = self.hklDataList[iHKLr]
        if self.__exclusions is not None:
            if self.__exclusions[self.tThSortInv[iHKLr]]:
                retval = False
        if self.__tThMax is not None:
            # FIXME: check for nans here???
            if hklData['tTheta'] > self.__tThMax \
              or np.isnan(hklData['tTheta']):
                retval = False
        return retval

    def __getTThRange(self, iHKLr):
        hklData = self.hklDataList[iHKLr]
        if self.tThWidth is not None:  # tThHi-tThLo < self.tThWidth
            tTh = hklData['tTheta']
            tThHi = tTh + self.tThWidth * 0.5
            tThLo = tTh - self.tThWidth * 0.5
        else:
            tThHi = hklData['tThetaHi']
            tThLo = hklData['tThetaLo']
        return (tThLo, tThHi)

    def getTThRanges(self, strainMag=None, lparms=None):
        """
        Return 2-theta ranges for included hkls

        return array is n x 2
        """
        if lparms is None:
            tThRanges = []
            for iHKLr, hklData in enumerate(self.hklDataList):
                if not self.__thisHKL(iHKLr):
                    continue
                # tThRanges.append([hklData['tThetaLo'], hklData['tThetaHi']])
                if strainMag is None:
                    tThRanges.append(self.__getTThRange(iHKLr))
                else:
                    hklData = self.hklDataList[iHKLr]
                    d = hklData['dSpacings']
                    tThLo = 2.0 * np.arcsin(
                        self.__wavelength / 2. / (d*(1.+strainMag))
                    )
                    tThHi = 2.0 * np.arcsin(
                        self.__wavelength / 2. / (d*(1.-strainMag))
                    )
                    tThRanges.append((tThLo, tThHi))
        else:
            new = self.__class__(self.__hkls, self)
            new.lparms = lparms
            tThRanges = new.getTThRanges(strainMag=strainMag)
        return np.array(tThRanges)

    def getMergedRanges(self, cullDupl=False):
        """
        return indices and ranges for specified planeData, merging where
        there is overlap based on the tThWidth and line positions
        """
        tThs = self.getTTh()
        tThRanges = self.getTThRanges()

        # if you end exlcusions in a doublet (or multiple close rings)
        # then this will 'fail'.  May need to revisit...
        nonoverlapNexts = np.hstack(
            (tThRanges[:-1, 1] < tThRanges[1:, 0], True)
        )
        iHKLLists = []
        mergedRanges = []
        hklsCur = []
        tThLoIdx = 0
        tThHiCur = 0.
        for iHKL, nonoverlapNext in enumerate(nonoverlapNexts):
            tThHi = tThRanges[iHKL, -1]
            if not nonoverlapNext:
                if cullDupl and abs(tThs[iHKL] - tThs[iHKL+1]) < sqrt_epsf:
                    continue
                else:
                    hklsCur.append(iHKL)
                    tThHiCur = tThHi
            else:
                hklsCur.append(iHKL)
                tThHiCur = tThHi
                iHKLLists.append(hklsCur)
                mergedRanges.append([tThRanges[tThLoIdx, 0], tThHiCur])
                tThLoIdx = iHKL + 1
                hklsCur = []
        return iHKLLists, mergedRanges

    def makeNew(self):
        new = self.__class__(None, self)
        return new

    def getTTh(self, lparms=None):
        if lparms is None:
            tTh = []
            for iHKLr, hklData in enumerate(self.hklDataList):
                if not self.__thisHKL(iHKLr):
                    continue
                tTh.append(hklData['tTheta'])
        else:
            new = self.makeNew()
            new.lparms = lparms
            tTh = new.getTTh()
        return np.array(tTh)

    def getDD_tThs_lparms(self):
        """
        derivatives of tThs with respect to lattice parameters;
        have not yet done coding for analytic derivatives, just wimp out
        and finite difference
        """
        pert = 1.0e-5  # assume they are all around unity
        pertInv = 1.0/pert

        lparmsRef = copy.deepcopy(self.__lparms)
        tThRef = self.getTTh()
        ddtTh = np.empty((len(tThRef), len(lparmsRef)))

        for iLparm in range(len(lparmsRef)):
            self.__lparms = copy.deepcopy(lparmsRef)
            self.__lparms[iLparm] += pert
            self.__calc()

            iTTh = 0
            for iHKLr, hklData in enumerate(self.hklDataList):
                if not self.__thisHKL(iHKLr):
                    continue
                ddtTh[iTTh, iLparm] = \
                    np.r_[hklData['tTheta'] - tThRef[iTTh]]*pertInv
                iTTh += 1

        'restore'
        self.__lparms = lparmsRef
        self.__calc()

        return ddtTh

    def getMultiplicity(self):          # ... JVB: is this incorrect?
        multip = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self.__thisHKL(iHKLr):
                continue
            multip.append(hklData['symHKLs'].shape[1])
        return np.array(multip)

    def getHKLID(self, hkl):
        'can call on a single hkl or list of hkls'
        if hasattr(hkl, '__setitem__'):  # tuple does not have __setitem__
            if hasattr(hkl, 'shape'):
                'if is ndarray, assume is 3xN'
                retval = list(map(self.__getHKLID, hkl.T))
            else:
                retval = list(map(self.__getHKLID, hkl))
        else:
            retval = self.__getHKLID(hkl)
        return retval

    def __getHKLID(self, hkl):
        """
        for hkl that is a tuple, return externally visible hkl index
        """
        if isinstance(hkl, int):
            retval = hkl
        else:
            hklList = self.getHKLs().tolist()
            dHKLInv = dict(
                [[tuple(hklThis), iHKL]
                 for iHKL, hklThis in enumerate(hklList)]
            )
            retval = dHKLInv[tuple(hkl)]
        return retval

    def getHKLs(self, asStr=False, thisTTh=None, allHKLs=False):
        """
        if pass thisTTh, then only return hkls overlapping the specified
        2-theta; if set allHKLs to true, the ignore exlcusions, tThMax, etc
        """
        hkls = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not allHKLs:
                if not self.__thisHKL(iHKLr):
                    continue
            if thisTTh is not None:
                tThLo, tThHi = self.__getTThRange(iHKLr)
                if thisTTh < tThHi and thisTTh > tThLo:
                    hkls.append(hklData['hkl'])
            else:
                hkls.append(hklData['hkl'])
        if asStr:
            retval = list(map(hklToStr, np.array(hkls)))
        else:
            retval = np.array(hkls)
        return retval

    def getSymHKLs(self, asStr=False, withID=False, indices=None):
        """
        new function that returns all symmetric hkls
        """
        retval = []
        iRetval = 0
        if indices is not None:
            indB = np.zeros(self.nHKLs, dtype=bool)
            indB[np.array(indices)] = True
        else:
            indB = np.ones(self.nHKLs, dtype=bool)
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self.__thisHKL(iHKLr):
                continue
            if indB[iRetval]:
                hkls = hklData['symHKLs']
                if asStr:
                    retval.append(list(map(hklToStr, np.array(hkls).T)))
                elif withID:
                    retval.append(
                        np.vstack(
                            [np.tile(hklData['hklID'], (1, hkls.shape[1])),
                             hkls]
                        )
                    )
                else:
                    retval.append(np.array(hkls))
            iRetval += 1
        return retval

    def getCentroSymHKLs(self):
        retval = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self.__thisHKL(iHKLr):
                continue
            retval.append(hklData['centrosym'])
        return retval

    def makeTheseScatteringVectors(self, hklList, rMat,
                                   bMat=None, wavelength=None, chiTilt=None):
        iHKLList = np.atleast_1d(self.getHKLID(hklList))
        fHKLs = np.hstack(self.getSymHKLs(indices=iHKLList))
        if bMat is None:
            bMat = self.__latVecOps['B']
        if wavelength is None:
            wavelength = self.__wavelength
        retval = PlaneData.makeScatteringVectors(
            fHKLs, rMat, bMat, wavelength, chiTilt=chiTilt
        )
        return retval

    def makeAllScatteringVectors(self, rMat,
                                 bMat=None, wavelength=None, chiTilt=None):
        fHKLs = np.hstack(self.getSymHKLs())
        if bMat is None:
            bMat = self.__latVecOps['B']
        if wavelength is None:
            wavelength = self.__wavelength
        retval = PlaneData.makeScatteringVectors(
            fHKLs, rMat, bMat, wavelength, chiTilt=chiTilt
        )
        return retval

    @staticmethod
    def makeScatteringVectors(hkls, rMat_c, bMat, wavelength, chiTilt=None):
        """
        Static method for calculating g-vectors and scattering vector angles
        for specified hkls, subject to the bragg conditions specified by
        lattice vectors, orientation matrix, and wavelength

        FIXME: must do testing on strained bMat
        """
        # arg munging
        if chiTilt is None:
            chi = 0.
        else:
            chi = float(chiTilt)
        rMat_c = rMat_c.squeeze()

        # these are the reciprocal lattice vectors in the SAMPLE FRAME
        # ** NOTE **
        #   if strained, assumes that you handed it a bMat calculated from
        #   strained [a, b, c] in the CRYSTAL FRAME
        gVec_s = np.dot(rMat_c, np.dot(bMat, hkls))

        dim0, nRefl = gVec_s.shape
        assert dim0 == 3, \
            "Looks like something is wrong with your lattice plane normals"

        # call model from transforms now
        oangs0, oangs1 = xfcapi.oscillAnglesOfHKLs(
            hkls.T, chi, rMat_c, bMat, wavelength
        )

        return gVec_s, oangs0.T, oangs1.T

    def __makeScatteringVectors(self, rMat, bMat=None, chiTilt=None):
        """
        modeled after QFromU.m
        """

        if bMat is None:
            bMat = self.__latVecOps['B']

        Qs_vec = []
        Qs_ang0 = []
        Qs_ang1 = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self.__thisHKL(iHKLr):
                continue
            thisQs, thisAng0, thisAng1 = PlaneData.makeScatteringVectors(
                hklData['symHKLs'], rMat, bMat,
                self.__wavelength, chiTilt=chiTilt
            )
            Qs_vec.append(thisQs)
            Qs_ang0.append(thisAng0)
            Qs_ang1.append(thisAng1)

        return Qs_vec, Qs_ang0, Qs_ang1

    def calcStructFactor(self, atominfo):
        """
        Calculates unit cell structure factors as a function of hkl

        USAGE:

        FSquared = calcStructFactor(atominfo,hkls,B)

        INPUTS:

        1) atominfo (m x 1 float ndarray) the first threee columns of the
        matrix contain fractional atom positions [uvw] of atoms in the unit
        cell. The last column contains the number of electrons for a given atom

        2) hkls (3 x n float ndarray) is the array of Miller indices for
        the planes of interest.  The vectors are assumed to be
        concatenated along the 1-axis (horizontal)

        3) B (3 x 3 float ndarray) is a matrix of reciprocal lattice basis
        vectors,where each column contains a reciprocal lattice basis vector
        ({g}=[B]*{hkl})


        OUTPUTS:

        1) FSquared (n x 1 float ndarray) array of structure factors,
        one for each hkl passed into the function
        """
        r = atominfo[:, 0:3]
        elecNum = atominfo[:, 3]
        hkls = self.hkls
        B = self.latVecOps['B']
        sinThOverLamdaList, ffDataList = LoadFormFactorData()
        FSquared = np.zeros(hkls.shape[1])

        for jj in np.arange(0, hkls.shape[1]):
            # ???: probably have other functions for this
            # Calculate G for each hkl
            # Calculate magnitude of G for each hkl
            G = hkls[0, jj]*B[:, 0] + hkls[1, jj]*B[:, 1] + hkls[2, jj]*B[:, 2]
            magG = np.sqrt(G[0]**2 + G[1]**2 + G[2]**2)

            # Begin calculating form factor
            F = 0
            for ii in np.arange(0, r.shape[0]):
                ff = RetrieveAtomicFormFactor(
                    elecNum[ii], magG, sinThOverLamdaList, ffDataList
                )
                exparg = complex(
                    0.,
                    2.*np.pi*(hkls[0, jj]*r[ii, 0]
                              + hkls[1, jj]*r[ii, 1]
                              + hkls[2, jj]*r[ii, 2])
                )
                F = F + ff*np.exp(exparg)

            """
            F = sum_atoms(ff(Q)*e^(2*pi*i(hu+kv+lw)))
            """
            FSquared[jj] = np.real(F*np.conj(F))

        return FSquared


def getFriedelPair(tth0, eta0, *ome0, **kwargs):
    """
    Get the diffractometer angular coordinates in degrees for
    the Friedel pair of a given reflection (min angular distance).

    AUTHORS:

    J. V. Bernier -- 10 Nov 2009

    USAGE:

    ome1, eta1 = getFriedelPair(tth0, eta0, *ome0,
                                display=False,
                                units='degrees',
                                convention='hexrd')

    INPUTS:

    1) tth0 is a list (or ndarray) of 1 or n the bragg angles (2theta) for
       the n reflections (tiled to match eta0 if only 1 is given).

    2) eta0 is a list (or ndarray) of 1 or n azimuthal coordinates for the n
       reflections  (tiled to match tth0 if only 1 is given).

    3) ome0 is a list (or ndarray) of 1 or n reference oscillation
       angles for the n reflections (denoted omega in [1]).  This argument
       is optional.

    4) Keyword arguments may be one of the following:

    Keyword             Values|{default}        Action
    --------------      --------------          --------------
    'display'           True|{False}            toggles display to cmd line
    'units'             'radians'|{'degrees'}   sets units for input angles
    'convention'        'fable'|{'hexrd'}       sets conventions defining
                                                the angles (see below)
    'chiTilt'           None                    the inclination (about Xlab) of
                                                the oscillation axis

    OUTPUTS:

    1) ome1 contains the oscialltion angle coordinates of the
       Friedel pairs associated with the n input reflections, relative to ome0
       (i.e. ome1 = <result> + ome0).  Output is in DEGREES!

    2) eta1 contains the azimuthal coordinates of the Friedel
       pairs associated with the n input reflections.  Output units are
       controlled via the module variable 'outputDegrees'

    NOTES:

    !!!: The ouputs ome1, eta1 are written using the selected convention, but
         the units are alway degrees.  May change this to work with Nathan's
         global...

    !!!: In the 'fable' convention [1], {XYZ} form a RHON basis where X is
         downstream, Z is vertical, and eta is CCW with +Z defining eta = 0.

    !!!: In the 'hexrd' convention [2], {XYZ} form a RHON basis where Z is
         upstream, Y is vertical, and eta is CCW with +X defining eta = 0.

    REFERENCES:

    [1] E. M. Lauridsen, S. Schmidt, R. M. Suter, and H. F. Poulsen,
        ``Tracking: a method for structural characterization of grains in
        powders or polycrystals''. J. Appl. Cryst. (2001). 34, 744--750

    [2] J. V. Bernier, M. P. Miller, J. -S. Park, and U. Lienert,
        ``Quantitative Stress Analysis of Recrystallized OFHC Cu Subject
        to Deformed In Situ'', J. Eng. Mater. Technol. (2008). 130.
        DOI:10.1115/1.2870234
    """

    dispFlag = False
    fableFlag = False
    chi = None
    c1 = 1.
    c2 = pi/180.

    # cast to arrays (in case they aren't)
    if np.isscalar(eta0):
        eta0 = [eta0]

    if np.isscalar(tth0):
        tth0 = [tth0]

    if np.isscalar(ome0):
        ome0 = [ome0]

    eta0 = np.asarray(eta0)
    tth0 = np.asarray(tth0)
    ome0 = np.asarray(ome0)

    if eta0.ndim != 1:
        raise RuntimeError('azimuthal input must be 1-D')

    npts = len(eta0)

    if tth0.ndim != 1:
        raise RuntimeError('Bragg angle input must be not 1-D')
    else:
        if len(tth0) != npts:
            if len(tth0) == 1:
                tth0 = tth0*np.ones(npts)
            elif npts == 1:
                npts = len(tth0)
                eta0 = eta0*np.ones(npts)
            else:
                raise RuntimeError(
                    'the azimuthal and Bragg angle inputs are inconsistent'
                )

    if len(ome0) == 0:
        ome0 = np.zeros(npts)                  # dummy ome0
    elif len(ome0) == 1 and npts > 1:
        ome0 = ome0*np.ones(npts)
    else:
        if len(ome0) != npts:
            raise RuntimeError(
                'your oscialltion angle input is inconsistent; '
                + 'it has length %d while it should be %d'
                % (len(ome0), npts)
            )

    # keyword args processing
    kwarglen = len(kwargs)
    if kwarglen > 0:
        argkeys = list(kwargs.keys())
        for i in range(kwarglen):
            if argkeys[i] == 'display':
                dispFlag = kwargs[argkeys[i]]
            elif argkeys[i] == 'convention':
                if kwargs[argkeys[i]].lower() == 'fable':
                    fableFlag = True
            elif argkeys[i] == 'units':
                if kwargs[argkeys[i]] == 'radians':
                    c1 = 180./pi
                    c2 = 1.
            elif argkeys[i] == 'chiTilt':
                if kwargs[argkeys[i]] is not None:
                    chi = kwargs[argkeys[i]]

    # a little talkback...
    if dispFlag:
        if fableFlag:
            print('\nUsing Fable angle convention\n')
        else:
            print('\nUsing image-based angle convention\n')

    # mapped eta input
    #   - in DEGREES, thanks to c1
    eta0 = mapAngle(c1*eta0, [-180, 180], units='degrees')
    if fableFlag:
        eta0 = 90 - eta0

    # must put args into RADIANS
    #   - eta0 is in DEGREES,
    #   - the others are in whatever was entered, hence c2
    eta0 = d2r*eta0
    tht0 = c2*tth0/2
    if chi is not None:
        chi = c2*chi
    else:
        chi = 0

    """
    SYSTEM SOLVE


    cos(chi)cos(eta)cos(theta)sin(x) - cos(chi)sin(theta)cos(x) \
        = sin(theta) - sin(chi)sin(eta)cos(theta)


    Identity: a sin x + b cos x = sqrt(a**2 + b**2) sin (x + alfa)

           /
           |      atan(b/a) for a > 0
     alfa <
           | pi + atan(b/a) for a < 0
           \

     => sin (x + alfa) = c / sqrt(a**2 + b**2)

     must use both branches for sin(x) = n:
     x = u (+ 2k*pi) | x = pi - u (+ 2k*pi)
    """
    cchi = np.cos(chi)
    schi = np.sin(chi)
    ceta = np.cos(eta0)
    seta = np.sin(eta0)
    ctht = np.cos(tht0)
    stht = np.sin(tht0)

    nchi = np.c_[0., cchi, schi].T

    gHat0_l = -np.vstack([ceta * ctht,
                          seta * ctht,
                          stht])

    a = cchi*ceta*ctht
    b = -cchi*stht
    c = stht + schi*seta*ctht

    # form solution
    abMag = np.sqrt(a*a + b*b)
    assert np.all(abMag > 0), \
        "Beam vector specification is infeasible!"
    phaseAng = np.arctan2(b, a)
    rhs = c / abMag
    rhs[abs(rhs) > 1.] = np.nan
    rhsAng = np.arcsin(rhs)

    # write ome angle output arrays (NaNs persist here)
    ome1 = rhsAng - phaseAng
    ome2 = np.pi - rhsAng - phaseAng

    ome1 = mapAngle(ome1, [-np.pi, np.pi], units='radians')
    ome2 = mapAngle(ome2, [-np.pi, np.pi], units='radians')

    ome_stack = np.vstack([ome1, ome2])

    min_idx = np.argmin(abs(ome_stack), axis=0)

    ome_min = ome_stack[min_idx, list(range(len(ome1)))]
    eta_min = np.nan * np.ones_like(ome_min)

    # mark feasible reflections
    goodOnes = -np.isnan(ome_min)

    numGood = sum(goodOnes)
    tmp_eta = np.empty(numGood)
    tmp_gvec = gHat0_l[:, goodOnes]
    for i in range(numGood):
        rchi = rotMatOfExpMap(
            np.tile(ome_min[goodOnes][i], (3, 1)) * nchi
        )
        gHat_l = np.dot(rchi, tmp_gvec[:, i].reshape(3, 1))
        tmp_eta[i] = np.arctan2(gHat_l[1], gHat_l[0])
        pass
    eta_min[goodOnes] = tmp_eta

    # everybody back to DEGREES!
    #     - ome1 is in RADIANS here
    #     - convert and put into [-180, 180]
    ome1 = mapAngle(
        mapAngle(
            r2d*ome_min, [-180, 180], units='degrees'
        ) + c1*ome0, [-180, 180], units='degrees'
    )

    # put eta1 in [-180, 180]
    eta1 = mapAngle(r2d*eta_min, [-180, 180], units='degrees')

    if not outputDegrees:
        ome1 = d2r * ome1
        eta1 = d2r * eta1

    return ome1, eta1


def getDparms(lp, lpTag, radians=True):
    """
    Utility routine for getting dparms, that is the lattice parameters
    without symmetry -- 'triclinic'
    """
    latVecOps = latticeVectors(lp, tag=lpTag, radians=radians)
    return latVecOps['dparms']


def LoadFormFactorData():
    """
    Script to read in a csv file containing information relating the
    magnitude of Q (sin(th)/lambda) to atomic form factor


    Notes:
    Atomic form factor data gathered from the International Tables of
    Crystallography:

     P. J. Brown, A. G. Fox,  E. N. Maslen, M. A. O'Keefec and B. T. M. Willis,
    "Chapter 6.1. Intensity of diffracted intensities", International Tables
     for Crystallography (2006). Vol. C, ch. 6.1, pp. 554-595
    """

    dir1 = os.path.split(valunits.__file__)
    dataloc = os.path.join(dir1[0], 'data', 'FormFactorVsQ.csv')

    data = np.zeros((62, 99), float)

    # FIXME: marked broken by DP
    jj = 0
    with open(dataloc, 'rU') as csvfile:
        datareader = csv.reader(csvfile, dialect=csv.excel)
        for row in datareader:
            ii = 0
            for val in row:
                data[jj, ii] = float(val)
                ii += 1
            jj += 1

    sinThOverLamdaList = data[:, 0]
    ffDataList = data[:, 1:]

    return sinThOverLamdaList, ffDataList


def RetrieveAtomicFormFactor(elecNum, magG, sinThOverLamdaList, ffDataList):
    """ Interpolates between tabulated data to find the atomic form factor
    for an atom with elecNum electrons for a given magnitude of Q

    USAGE:

    ff = RetrieveAtomicFormFactor(elecNum,magG,sinThOverLamdaList,ffDataList)

    INPUTS:

    1) elecNum, (1 x 1 float) number of electrons for atom of interest

    2) magG (1 x 1 float) magnitude of G

    3) sinThOverLamdaList (n x 1 float ndarray) form factor data is tabulated
       in terms of sin(theta)/lambda (A^-1).

    3) ffDataList (n x m float ndarray) form factor data is tabulated in terms
       of sin(theta)/lambda (A^-1). Each column corresponds to a different
       number of electrons

    OUTPUTS:

    1) ff (n x 1 float) atomic form factor for atom and hkl of interest

    NOTES:
    Data should be calculated in terms of G at some point

    """
    sinThOverLambda = 0.5*magG
    # lambda=2*d*sin(th)
    # lambda=2*sin(th)/G
    # 1/2*G=sin(th)/lambda

    ff = np.interp(
        sinThOverLambda, sinThOverLamdaList, ffDataList[:, (elecNum - 1)]
    )

    return ff
