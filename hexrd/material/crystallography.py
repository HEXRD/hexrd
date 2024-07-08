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
from typing import Optional, Union, Dict, List, Tuple

import numpy as np

from hexrd.material.unitcell import unitcell
from hexrd.deprecation import deprecated
from hexrd import constants
from hexrd.matrixutil import unitVector
from hexrd.rotations import (
    rotMatOfExpMap,
    mapAngle,
    applySym,
    ltypeOfLaueGroup,
    quatOfLaueGroup,
)
from hexrd.transforms import xfcapi
from hexrd import valunits
from hexrd.valunits import toFloat
from hexrd.constants import d2r, r2d, sqrt_epsf

"""module vars"""

# units
dUnit = 'angstrom'
outputDegrees = False
outputDegrees_bak = outputDegrees


def hklToStr(hkl: np.ndarray) -> str:
    """
    Converts hkl representation to a string.

    Parameters
    ----------
    hkl : np.ndarray
        3 element list of h, k, and l values (Miller indices).

    Returns
    -------
    str
        Space-separated string representation of h, k, and l values.

    """
    return re.sub(r'[\[\]\(\)\{\},]', '', str(hkl))


def tempSetOutputDegrees(val: bool) -> None:
    """
    Set the global outputDegrees flag temporarily. Can be reverted with
    revertOutputDegrees().

    Parameters
    ----------
    val : bool
        True to output angles in degrees, False to output angles in radians.

    Returns
    -------
    None

    """
    global outputDegrees, outputDegrees_bak
    outputDegrees_bak = outputDegrees
    outputDegrees = val


def revertOutputDegrees() -> None:
    """
    Revert the effect of tempSetOutputDegrees(), resetting the outputDegrees
    flag to its previous value (True to output in degrees, False for radians).

    Returns
    -------
    None
    """
    global outputDegrees, outputDegrees_bak
    outputDegrees = outputDegrees_bak


def cosineXform(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spherical trig transform to take alpha, beta, gamma to expressions
    for cos(alpha*).  See ref below.

    [1] R. J. Neustadt, F. W. Cagle, Jr., and J. Waser, ``Vector algebra and
        the relations between direct and reciprocal lattice quantities''. Acta
        Cryst. (1968), A24, 247--248

    Parameters
    ----------
    a : np.ndarray
        List of alpha angle values (radians).
    b : np.ndarray
        List of beta angle values (radians).
    c : np.ndarray
        List of gamma angle values (radians).

    Returns
    -------
    np.ndarray
        List of cos(alpha*) values.
    np.ndarray
        List of sin(alpha*) values.

    """
    cosar = (np.cos(b) * np.cos(c) - np.cos(a)) / (np.sin(b) * np.sin(c))
    sinar = np.sqrt(1 - cosar**2)
    return cosar, sinar


def processWavelength(arg: Union[valunits.valWUnit, float]) -> float:
    """
    Convert an energy value to a wavelength.  If argument has units of length
    or energy, will convert to globally specified unit type for wavelength
    (dUnit).  If argument is a scalar, assumed input units are keV.
    """
    if isinstance(arg, valunits.valWUnit):
        # arg is a valunits.valWUnit object
        if arg.isLength():
            return arg.getVal(dUnit)
        elif arg.isEnergy():
            e = arg.getVal('keV')
            return valunits.valWUnit(
                'wavelength', 'length', constants.keVToAngstrom(e), 'angstrom'
            ).getVal(dUnit)
        else:
            raise RuntimeError('do not know what to do with ' + str(arg))
    else:
        # !!! assuming arg is in keV
        return valunits.valWUnit(
            'wavelength', 'length', constants.keVToAngstrom(arg), 'angstrom'
        ).getVal(dUnit)


def latticePlanes(
    hkls: np.ndarray,
    lparms: np.ndarray,
    ltype: Optional[str] = 'cubic',
    wavelength: Optional[float] = 1.54059292,
    strainMag: Optional[float] = None,
) -> Dict[str, np.ndarray]:
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

    The following optional arguments are recognized:

       3) ltype=(string) is a string representing the symmetry type of
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

       4) wavelength=<float> is a value represented the wavelength in
          Angstroms to calculate bragg angles for.  The default value
          is for Cu K-alpha radiation (1.54059292 Angstrom)

       5) strainMag=None

    OUTPUTS:

    1) planeInfo is a dictionary containing the following keys/items:

       normals   (3, n) double array    array of the components to the
                                        unit normals for each {hkl} in
                                        X (horizontally concatenated)

       dspacings (n,  ) double array    array of the d-spacings for
                                        each {hkl}

       tThetas   (n,  ) double array    array of the Bragg angles for
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

    assert (
        hkls.shape[0] == 3
    ), f"hkls aren't column vectors in call to '{location}'!"

    tag = ltype
    wlen = wavelength

    # get B
    L = latticeVectors(lparms, tag)

    # get G-vectors -- reciprocal vectors in crystal frame
    G = np.dot(L['B'], hkls)

    # magnitudes
    d = 1 / np.sqrt(np.sum(G**2, 0))

    aconv = 1.0
    if outputDegrees:
        aconv = r2d

    # two thetas
    sth = wlen / 2.0 / d
    mask = np.abs(sth) < 1.0
    tth = np.zeros(sth.shape)

    tth[~mask] = np.nan
    tth[mask] = aconv * 2.0 * np.arcsin(sth[mask])

    p = dict(normals=unitVector(G), dspacings=d, tThetas=tth)

    if strainMag is not None:
        p['tThetasLo'] = np.zeros(sth.shape)
        p['tThetasHi'] = np.zeros(sth.shape)

        mask = (np.abs(wlen / 2.0 / (d * (1.0 + strainMag))) < 1.0) & (
            np.abs(wlen / 2.0 / (d * (1.0 - strainMag))) < 1.0
        )

        p['tThetasLo'][~mask] = np.nan
        p['tThetasHi'][~mask] = np.nan

        p['tThetasLo'][mask] = (
            aconv * 2 * np.arcsin(wlen / 2.0 / (d[mask] * (1.0 + strainMag)))
        )
        p['tThetasHi'][mask] = (
            aconv * 2 * np.arcsin(wlen / 2.0 / (d[mask] * (1.0 - strainMag)))
        )

    return p


def latticeVectors(
    lparms: np.ndarray,
    tag: Optional[str] = 'cubic',
    radians: Optional[bool] = False,
) -> Dict[str, Union[np.ndarray, float]]:
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

    2) tag (string) is a case-insensitive string representing the
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

    The following optional arguments are recognized:

       3) radians=<bool> is a boolean flag indicating usage of radians rather
          than degrees, defaults to false.

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
        'triclinic',
    ]

    if radians:
        aconv = 1.0
    else:
        aconv = pi / 180.0  # degToRad
    deg90 = pi / 2.0
    deg120 = 2.0 * pi / 3.0
    #
    if tag == lattStrings[0]:
        # cubic
        cellparms = np.r_[np.tile(lparms[0], (3,)), deg90 * np.ones((3,))]
    elif tag == lattStrings[1] or tag == lattStrings[2]:
        # hexagonal | trigonal (hex indices)
        cellparms = np.r_[
            lparms[0], lparms[0], lparms[1], deg90, deg90, deg120
        ]
    elif tag == lattStrings[3]:
        # rhombohedral
        cellparms = np.r_[
            np.tile(lparms[0], (3,)), np.tile(aconv * lparms[1], (3,))
        ]
    elif tag == lattStrings[4]:
        # tetragonal
        cellparms = np.r_[lparms[0], lparms[0], lparms[1], deg90, deg90, deg90]
    elif tag == lattStrings[5]:
        # orthorhombic
        cellparms = np.r_[lparms[0], lparms[1], lparms[2], deg90, deg90, deg90]
    elif tag == lattStrings[6]:
        # monoclinic
        cellparms = np.r_[
            lparms[0], lparms[1], lparms[2], deg90, aconv * lparms[3], deg90
        ]
    elif tag == lattStrings[7]:
        # triclinic
        cellparms = np.r_[
            lparms[0],
            lparms[1],
            lparms[2],
            aconv * lparms[3],
            aconv * lparms[4],
            aconv * lparms[5],
        ]
    else:
        raise RuntimeError(f'lattice tag "{tag}" is not recognized')

    alpha, beta, gamma = cellparms[3:6]
    cosalfar, sinalfar = cosineXform(alpha, beta, gamma)

    a = cellparms[0] * np.r_[1, 0, 0]
    b = cellparms[1] * np.r_[np.cos(gamma), np.sin(gamma), 0]
    c = (
        cellparms[2]
        * np.r_[
            np.cos(beta), -cosalfar * np.sin(beta), sinalfar * np.sin(beta)
        ]
    )

    ad = np.sqrt(np.sum(a**2))
    bd = np.sqrt(np.sum(b**2))
    cd = np.sqrt(np.sum(c**2))

    # Cell volume
    V = np.dot(a, np.cross(b, c))

    # F takes components in the direct lattice to X
    F = np.c_[a, b, c]

    # Reciprocal lattice vectors
    astar = np.cross(b, c) / V
    bstar = np.cross(c, a) / V
    cstar = np.cross(a, b) / V

    # and parameters
    ar = np.sqrt(np.sum(astar**2))
    br = np.sqrt(np.sum(bstar**2))
    cr = np.sqrt(np.sum(cstar**2))

    alfar = np.arccos(np.dot(bstar, cstar) / br / cr)
    betar = np.arccos(np.dot(cstar, astar) / cr / ar)
    gamar = np.arccos(np.dot(astar, bstar) / ar / br)

    # B takes components in the reciprocal lattice to X
    B = np.c_[astar, bstar, cstar]

    cosalfar2, sinalfar2 = cosineXform(alfar, betar, gamar)

    afable = ar * np.r_[1, 0, 0]
    bfable = br * np.r_[np.cos(gamar), np.sin(gamar), 0]
    cfable = (
        cr
        * np.r_[
            np.cos(betar),
            -cosalfar2 * np.sin(betar),
            sinalfar2 * np.sin(betar),
        ]
    )

    BR = np.c_[afable, bfable, cfable]
    U0 = np.dot(B, np.linalg.inv(BR))
    if outputDegrees:
        dparms = np.r_[ad, bd, cd, r2d * np.r_[alpha, beta, gamma]]
        rparms = np.r_[ar, br, cr, r2d * np.r_[alfar, betar, gamar]]
    else:
        dparms = np.r_[ad, bd, cd, np.r_[alpha, beta, gamma]]
        rparms = np.r_[ar, br, cr, np.r_[alfar, betar, gamar]]

    return {
        'F': F,
        'B': B,
        'BR': BR,
        'U0': U0,
        'vol': V,
        'dparms': dparms,
        'rparms': rparms,
    }


class PlaneData(object):
    """
    Careful with ordering: Outputs are ordered by the 2-theta for the
    hkl unless you get self._hkls directly, and this order can change
    with changes in lattice parameters (lparms); setting and getting
    exclusions works on the current hkl ordering, not the original
    ordering (in self._hkls), but exclusions are stored in the
    original ordering in case the hkl ordering does change with
    lattice parameters

    if not None, tThWidth takes priority over strainMag in setting
    two-theta ranges; changing strainMag automatically turns off
    tThWidth
    """

    def __init__(self, hkls: Optional[np.ndarray], *args, **kwargs) -> None:
        """
        Constructor for PlaneData

        Parameters
        ----------
        hkls : np.ndarray, optional
            Miller indices to be used in the plane data.  Can be set later

        *args
            Unnamed arguments. Could be in the format of `lparms, laueGroup,
            wavelength, strainMag`, or just a `PlaneData` object.

        **kwargs
            Valid keyword arguments include:
            - doTThSort
            - exclusions
            - tThMax
            - tThWidth
        """
        self._doTThSort = True
        self._exclusions = None
        self._tThMax = None

        if len(args) == 4:
            lparms, laueGroup, wavelength, strainMag = args
            tThWidth = None
            self._wavelength = processWavelength(wavelength)
            self._lparms = self._parseLParms(lparms)
        elif len(args) == 1 and isinstance(args[0], PlaneData):
            other = args[0]
            lparms, laueGroup, wavelength, strainMag, tThWidth = (
                other.getParams()
            )
            self._wavelength = wavelength
            self._lparms = lparms
            self._doTThSort = other._doTThSort
            self._exclusions = other._exclusions
            self._tThMax = other._tThMax
            if hkls is None:
                hkls = other._hkls
        else:
            raise NotImplementedError(f'args : {args}')

        self._laueGroup = laueGroup
        self._hkls = copy.deepcopy(hkls)
        self._strainMag = strainMag
        self._structFact = np.ones(self._hkls.shape[1])
        self.tThWidth = tThWidth

        # ... need to implement tThMin too
        if 'doTThSort' in kwargs:
            self._doTThSort = kwargs.pop('doTThSort')
        if 'exclusions' in kwargs:
            self._exclusions = kwargs.pop('exclusions')
        if 'tThMax' in kwargs:
            self._tThMax = toFloat(kwargs.pop('tThMax'), 'radians')
        if 'tThWidth' in kwargs:
            self.tThWidth = kwargs.pop('tThWidth')
        if len(kwargs) > 0:
            raise RuntimeError(
                f'have unparsed keyword arguments with keys: {kwargs.keys()}'
            )

        # This is only used to calculate the structure factor if invalidated
        self._unitcell: unitcell = None

        self._calc()

    def _calc(self):
        symmGroup = ltypeOfLaueGroup(self._laueGroup)
        self._q_sym = quatOfLaueGroup(self._laueGroup)
        _, latVecOps, hklDataList = PlaneData.makePlaneData(
            self._hkls,
            self._lparms,
            self._q_sym,
            symmGroup,
            self._strainMag,
            self.wavelength,
        )
        'sort by tTheta'
        tThs = np.array(
            [hklDataList[iHKL]['tTheta'] for iHKL in range(len(hklDataList))]
        )
        if self._doTThSort:
            # sorted hkl -> _hkl
            # _hkl -> sorted hkl
            self.tThSort = np.argsort(tThs)
            self.tThSortInv = np.empty(len(hklDataList), dtype=int)
            self.tThSortInv[self.tThSort] = np.arange(len(hklDataList))
            self.hklDataList = [hklDataList[iHKL] for iHKL in self.tThSort]
        else:
            self.tThSort = np.arange(len(hklDataList))
            self.tThSortInv = np.arange(len(hklDataList))
            self.hklDataList = hklDataList
        self._latVecOps = latVecOps
        self.nHKLs = len(self.getHKLs())

    def __str__(self):
        s = '========== plane data ==========\n'
        s += 'lattice parameters:\n   ' + str(self.lparms) + '\n'
        s += f'two theta width: ({str(self.tThWidth)})\n'
        s += f'strain magnitude: ({str(self.strainMag)})\n'
        s += f'beam energy ({str(self.wavelength)})\n'
        s += 'hkls: (%d)\n' % self.nHKLs
        s += str(self.getHKLs())
        return s

    def getParams(self):
        """
        Getter for the parameters of the plane data.

        Returns
        -------
        tuple
            The parameters of the plane data. In the order of
            _lparams, _laueGroup, _wavelength, _strainMag, tThWidth

        """
        return (
            self._lparms,
            self._laueGroup,
            self._wavelength,
            self._strainMag,
            self.tThWidth,
        )

    def getNhklRef(self) -> int:
        """
        Get the total number of hkl's in the plane data, not ignoring
        ones that are excluded in exclusions.

        Returns
        -------
        int
            The total number of hkl's in the plane data.
        """
        return len(self.hklDataList)

    @property
    def hkls(self) -> np.ndarray:
        """
        hStacked Hkls of the plane data (Miller indices).
        """
        return self.getHKLs().T

    @hkls.setter
    def hkls(self, hkls):
        raise NotImplementedError('for now, not allowing hkls to be reset')

    @property
    def tThMax(self) -> Optional[float]:
        """
        Maximum 2-theta value of the plane data.

        float or None
        """
        return self._tThMax

    @tThMax.setter
    def tThMax(self, t_th_max: Union[float, valunits.valWUnit]) -> None:
        self._tThMax = toFloat(t_th_max, 'radians')

    @property
    def exclusions(self) -> np.ndarray:
        """
        Excluded HKL's the plane data.

        Set as type np.ndarray, as a mask of length getNhklRef(), a list of
            indices to be excluded, or a list of ranges of indices.

        Read as a mask of length getNhklRef().
        """
        retval = np.zeros(self.getNhklRef(), dtype=bool)
        if self._exclusions is not None:
            # report in current hkl ordering
            retval[:] = self._exclusions[self.tThSortInv]
        if self._tThMax is not None:
            for iHKLr, hklData in enumerate(self.hklDataList):
                if hklData['tTheta'] > self._tThMax:
                    retval[iHKLr] = True
        return retval

    @exclusions.setter
    def exclusions(self, new_exclusions: Optional[np.ndarray]) -> None:
        excl = np.zeros(len(self.hklDataList), dtype=bool)
        if new_exclusions is not None:
            exclusions = np.atleast_1d(new_exclusions)
            if len(exclusions) == len(self.hklDataList):
                assert (
                    exclusions.dtype == 'bool'
                ), 'Exclusions should be bool if full length'
                # convert from current hkl ordering to _hkl ordering
                excl[:] = exclusions[self.tThSort]
            else:
                if len(exclusions.shape) == 1:
                    # treat exclusions as indices
                    excl[self.tThSort[exclusions]] = True
                elif len(exclusions.shape) == 2:
                    # treat exclusions as ranges of indices
                    for r in exclusions:
                        excl[self.tThSort[r[0] : r[1]]] = True
                else:
                    raise RuntimeError(
                        f'Unclear behavior for shape {exclusions.shape}'
                    )
        self._exclusions = excl
        self.nHKLs = np.sum(np.logical_not(self._exclusions))

    def exclude(
        self,
        dmin: Optional[float] = None,
        dmax: Optional[float] = None,
        tthmin: Optional[float] = None,
        tthmax: Optional[float] = None,
        sfacmin: Optional[float] = None,
        sfacmax: Optional[float] = None,
        pintmin: Optional[float] = None,
        pintmax: Optional[float] = None,
    ) -> None:
        """
        Set exclusions according to various parameters

        Any hkl with a value below any min or above any max will be excluded. So
        to be included, an hkl needs to have values between the min and max
        for all of the conditions given.

        Note that method resets the tThMax attribute to None.

        PARAMETERS
        ----------
        dmin: float > 0
            minimum lattice spacing (angstroms)
        dmax: float > 0
            maximum lattice spacing (angstroms)
        tthmin: float > 0
            minimum two theta (radians)
        tthmax: float > 0
            maximum two theta (radians)
        sfacmin: float > 0
            minimum structure factor as a proportion of maximum
        sfacmax: float > 0
            maximum structure factor as a proportion of maximum
        pintmin: float > 0
            minimum powder intensity as a proportion of maximum
        pintmax: float > 0
            maximum powder intensity as a proportion of maximum
        """
        excl = np.zeros(self.getNhklRef(), dtype=bool)
        self.exclusions = None
        self.tThMax = None

        if (dmin is not None) or (dmax is not None):
            d = np.array(self.getPlaneSpacings())
            if dmin is not None:
                excl[d < dmin] = True
            if dmax is not None:
                excl[d > dmax] = True

        if (tthmin is not None) or (tthmax is not None):
            tth = self.getTTh()
            if tthmin is not None:
                excl[tth < tthmin] = True
            if tthmax is not None:
                excl[tth > tthmax] = True

        if (sfacmin is not None) or (sfacmax is not None):
            sfac = self.structFact
            sfac = sfac / sfac.max()
            if sfacmin is not None:
                excl[sfac < sfacmin] = True
            if sfacmax is not None:
                excl[sfac > sfacmax] = True

        if (pintmin is not None) or (pintmax is not None):
            pint = self.powder_intensity
            pint = pint / pint.max()
            if pintmin is not None:
                excl[pint < pintmin] = True
            if pintmax is not None:
                excl[pint > pintmax] = True

        self.exclusions = excl

    def _parseLParms(
        self, lparms: List[Union[valunits.valWUnit, float]]
    ) -> List[float]:
        lparmsDUnit = []
        for lparmThis in lparms:
            if isinstance(lparmThis, valunits.valWUnit):
                if lparmThis.isLength():
                    lparmsDUnit.append(lparmThis.getVal(dUnit))
                elif lparmThis.isAngle():
                    # plumbing set up to default to degrees
                    # for lattice parameters
                    lparmsDUnit.append(lparmThis.getVal('degrees'))
                else:
                    raise RuntimeError(
                        f'Do not know what to do with {lparmThis}'
                    )
            else:
                lparmsDUnit.append(lparmThis)
        return lparmsDUnit

    @property
    def lparms(self) -> List[float]:
        """
        Lattice parameters of the plane data.

        Can be set as a List[float | valWUnit], but will be converted to
        List[float].
        """
        return self._lparms

    @lparms.setter
    def lparms(self, lparms: List[Union[valunits.valWUnit, float]]) -> None:
        self._lparms = self._parseLParms(lparms)
        self._calc()

    @property
    def strainMag(self) -> Optional[float]:
        """
        Strain magnitude of the plane data.

        float or None
        """
        return self._strainMag

    @strainMag.setter
    def strainMag(self, strain_mag: float) -> None:
        self._strainMag = strain_mag
        self.tThWidth = None
        self._calc()

    @property
    def wavelength(self) -> float:
        """
        Wavelength of the plane data.

        Set as float or valWUnit.

        Read as float
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength: Union[float, valunits.valWUnit]) -> None:
        wavelength = processWavelength(wavelength)
        # Do not re-compute if it is almost the same
        if np.isclose(self._wavelength, wavelength):
            return

        self._wavelength = wavelength
        self._calc()

    def invalidate_structure_factor(self, ucell: unitcell) -> None:
        """
        It can be expensive to compute the structure factor
        This method just invalidates it, providing a unit cell,
        so that it can be lazily computed from the unit cell.

        Parameters:
        -----------
        unitcell : unitcell
            The unit cell to be used to compute the structure factor
        """
        self._structFact = None
        self._hedm_intensity = None
        self._powder_intensity = None
        self._unitcell = ucell

    def _compute_sf_if_needed(self):
        any_invalid = (
            self._structFact is None
            or self._hedm_intensity is None
            or self._powder_intensity is None
        )
        if any_invalid and self._unitcell is not None:
            # Compute the structure factor first.
            # This can be expensive to do, so we lazily compute it when needed.
            hkls = self.getHKLs(allHKLs=True)
            self.structFact = self._unitcell.CalcXRSF(hkls)

    @property
    def structFact(self) -> np.ndarray:
        """
        Structure factors for each hkl.

        np.ndarray
        """
        self._compute_sf_if_needed()
        return self._structFact[~self.exclusions]

    @structFact.setter
    def structFact(self, structFact: np.ndarray) -> None:
        self._structFact = structFact
        multiplicity = self.getMultiplicity(allHKLs=True)
        tth = self.getTTh(allHKLs=True)

        hedm_intensity = (
            structFact * lorentz_factor(tth) * polarization_factor(tth)
        )

        powderI = hedm_intensity * multiplicity

        # Now scale them
        hedm_intensity = 100.0 * hedm_intensity / np.nanmax(hedm_intensity)
        powderI = 100.0 * powderI / np.nanmax(powderI)

        self._hedm_intensity = hedm_intensity
        self._powder_intensity = powderI

    @property
    def powder_intensity(self) -> np.ndarray:
        """
        Powder intensity for each hkl.
        """
        self._compute_sf_if_needed()
        return self._powder_intensity[~self.exclusions]

    @property
    def hedm_intensity(self) -> np.ndarray:
        """
        HEDM (high energy x-ray diffraction microscopy) intensity for each hkl.
        """
        self._compute_sf_if_needed()
        return self._hedm_intensity[~self.exclusions]

    @staticmethod
    def makePlaneData(
        hkls: np.ndarray,
        lparms: np.ndarray,
        qsym: np.ndarray,
        symmGroup,
        strainMag,
        wavelength,
    ) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray | float], List[Dict]
    ]:
        """
        Generate lattice plane data from inputs.

        Parameters:
        -----------
        hkls: np.ndarray
            Miller indices, as in crystallography.latticePlanes
        lparms: np.ndarray
            Lattice parameters, as in crystallography.latticePlanes
        qsym: np.ndarray
            (4, n) containing quaternions of symmetry
        symmGroup: str
            Tag for the symmetry (Laue) group of the lattice. Can generate from
            ltypeOfLaueGroup
        strainMag: float
            Swag of strain magnitudes
        wavelength: float
            Wavelength

        Returns:
        -------
        dict:
            Dictionary containing lattice plane data
        dict:
            Dictionary containing lattice vector operators
        list:
            List of dictionaries, each containing the data for one hkl
        """

        tempSetOutputDegrees(False)
        latPlaneData = latticePlanes(
            hkls,
            lparms,
            ltype=symmGroup,
            strainMag=strainMag,
            wavelength=wavelength,
        )

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
                cullPM=False,
            )

            # check for +/- in symmetry group
            latPlnNrmlsM = applySym(
                np.dot(latVecOps['B'], hkls[:, iHKL].reshape(3, 1)),
                qsym,
                csFlag=False,
                cullPM=False,
            )

            csRefl = latPlnNrmls.shape[1] == latPlnNrmlsM.shape[1]

            # added this so that I retain the actual symmetric
            # integer hkls as well
            symHKLs = np.array(
                np.round(np.dot(latVecOps['F'].T, latPlnNrmls)), dtype='int'
            )

            hklDataList.append(
                dict(
                    hklID=iHKL,
                    hkl=hkls[:, iHKL],
                    tTheta=latPlaneData['tThetas'][iHKL],
                    dSpacings=latPlaneData['dspacings'][iHKL],
                    tThetaLo=latPlaneData['tThetasLo'][iHKL],
                    tThetaHi=latPlaneData['tThetasHi'][iHKL],
                    latPlnNrmls=unitVector(latPlnNrmls),
                    symHKLs=symHKLs,
                    centrosym=csRefl,
                )
            )

        revertOutputDegrees()
        return latPlaneData, latVecOps, hklDataList

    @property
    def laueGroup(self) -> str:
        """
        This is the Schoenflies tag, describing symmetry group of the lattice.
        Note that setting this with incompatible lattice parameters will
        cause an error. If changing both, use set_laue_and_lparms.

        str
        """
        return self._laueGroup

    @laueGroup.setter
    def laueGroup(self, laueGroup: str) -> None:
        self._laueGroup = laueGroup
        self._calc()

    def set_laue_and_lparms(
        self, laueGroup: str, lparms: List[valunits.valWUnit | float]
    ) -> None:
        """
        Set the Laue group and lattice parameters simultaneously

        When the Laue group changes, the lattice parameters may be
        incompatible, and cause an error in self._calc(). This function
        allows us to update both the Laue group and lattice parameters
        simultaneously to avoid this issue.

        Parameters:
        -----------
        laueGroup : str
            The symmetry (Laue) group to be set
        lparms : List[valunits.valWUnit | float]
            Lattice parameters to be set
        """
        self._laueGroup = laueGroup
        self._lparms = self._parseLParms(lparms)
        self._calc()

    @property
    def q_sym(self) -> np.ndarray:
        """
        Quaternions of symmetry for each hkl, generated from the Laue group

        np.ndarray((4, n))
        """
        return self._q_sym  # rotations.quatOfLaueGroup(self._laueGroup)

    def getPlaneSpacings(self) -> List[float]:
        """
        Plane spacings for each hkl.

        Returns:
        -------
        List[float]
            List of plane spacings for each hkl
        """
        dspacings = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self._thisHKL(iHKLr):
                continue
            dspacings.append(hklData['dSpacings'])
        return dspacings

    @property
    def latVecOps(self) -> Dict[str, np.ndarray | float]:
        """
        gets lattice vector operators as a new (deepcopy)

        Returns:
        -------
        Dict[str, np.ndarray | float]
            Dictionary containing lattice vector operators
        """
        return copy.deepcopy(self._latVecOps)

    def _thisHKL(self, iHKLr: int) -> bool:
        hklData = self.hklDataList[iHKLr]
        if self._exclusions is not None:
            if self._exclusions[self.tThSortInv[iHKLr]]:
                return False
        if self._tThMax is not None:
            if hklData['tTheta'] > self._tThMax or np.isnan(hklData['tTheta']):
                return False
        return True

    def _getTThRange(self, iHKLr: int) -> Tuple[float, float]:
        hklData = self.hklDataList[iHKLr]
        if self.tThWidth is not None:  # tThHi-tThLo < self.tThWidth
            tTh = hklData['tTheta']
            tThHi = tTh + self.tThWidth * 0.5
            tThLo = tTh - self.tThWidth * 0.5
        else:
            tThHi = hklData['tThetaHi']
            tThLo = hklData['tThetaLo']
        return (tThLo, tThHi)

    def getTThRanges(self, strainMag: Optional[float] = None) -> np.ndarray:
        """
        Get the 2-theta ranges for included hkls

        Parameters:
        -----------
        strainMag : Optional[float]
            Optional swag of strain magnitude

        Returns:
        -------
        np.ndarray:
            hstacked array of hstacked tThLo and tThHi for each hkl (n x 2)
        """
        tThRanges = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self._thisHKL(iHKLr):
                continue
            if strainMag is None:
                tThRanges.append(self._getTThRange(iHKLr))
            else:
                hklData = self.hklDataList[iHKLr]
                d = hklData['dSpacings']
                tThLo = 2.0 * np.arcsin(
                    self._wavelength / 2.0 / (d * (1.0 + strainMag))
                )
                tThHi = 2.0 * np.arcsin(
                    self._wavelength / 2.0 / (d * (1.0 - strainMag))
                )
                tThRanges.append((tThLo, tThHi))
        return np.array(tThRanges)

    def getMergedRanges(
        self, cullDupl: Optional[bool] = False
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Return indices and ranges for specified planeData, merging where
        there is overlap based on the tThWidth and line positions

        Parameters:
        -----------
        cullDupl : (optional) bool
            If True, cull duplicate 2-theta values (within sqrt_epsf). Defaults
            to False.

        Returns:
        --------
        List[List[int]]
            List of indices for each merged range

        List[List[float]]
            List of merged ranges, (n x 2)
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
        tThHiCur = 0.0
        for iHKL, nonoverlapNext in enumerate(nonoverlapNexts):
            tThHi = tThRanges[iHKL, -1]
            if not nonoverlapNext:
                if cullDupl and abs(tThs[iHKL] - tThs[iHKL + 1]) < sqrt_epsf:
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

    def getTTh(self, allHKLs: Optional[bool] = False) -> np.ndarray:
        """
        Get the 2-theta values for each hkl.

        Parameters:
        -----------
        allHKLs : (optional) bool
            If True, return all 2-theta values, even if they are excluded in
            the current planeData. Default is False.

        Returns:
        -------
        np.ndarray
            Array of 2-theta values for each hkl
        """
        tTh = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not allHKLs and not self._thisHKL(iHKLr):
                continue
            tTh.append(hklData['tTheta'])
        return np.array(tTh)

    def getMultiplicity(self, allHKLs: Optional[bool] = False) -> np.ndarray:
        """
        Get the multiplicity for each hkl (number of symHKLs).

        Paramters:
        ----------
        allHKLs : (optional) bool
            If True, return all multiplicities, even if they are excluded in
            the current planeData.  Defaults to false.

        Returns
        -------
        np.ndarray
            Array of multiplicities for each hkl
        """
        # ... JVB: is this incorrect?
        multip = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if allHKLs or self._thisHKL(iHKLr):
                multip.append(hklData['symHKLs'].shape[1])
        return np.array(multip)

    def getHKLID(
        self,
        hkl: Union[int, Tuple[int, int, int], np.ndarray],
        master: Optional[bool] = False,
    ) -> Union[List[int], int]:
        """
        Return the unique ID of a list of hkls.

        Parameters
        ----------
        hkl : int | tuple | list | numpy.ndarray
            The input hkl.  If an int, or a list of ints, it just passes
            through (FIXME).
            If a tuple, treated as a single (h, k, l).
            If a list of lists/tuples, each is treated as an (h, k, l).
            If an numpy.ndarray, it is assumed to have shape (3, N) with the
            N (h, k, l) vectors stacked column-wise

        master : bool, optional
            If True, return the master hklID, else return the index from the
            external (sorted and reduced) list.

        Returns
        -------
        hkl_ids : list
            The list of requested hklID values associate with the input.

        Notes
        -----
        TODO: revisit this weird API???

        Changes:
        -------
        2020-05-21 (JVB) -- modified to handle all symmetric equavlent reprs.
        """
        if hasattr(hkl, '__setitem__'):  # tuple does not have __setitem__
            if isinstance(hkl, np.ndarray):
                # if is ndarray, assume is 3xN
                return [self._getHKLID(x, master=master) for x in hkl.T]
            else:
                return [self._getHKLID(x, master=master) for x in hkl]
        else:
            return self._getHKLID(hkl, master=master)

    def _getHKLID(
        self,
        hkl: Union[int, Tuple[int, int, int], np.ndarray],
        master: Optional[bool] = False,
    ) -> int:
        """
        for hkl that is a tuple, return externally visible hkl index
        """
        if isinstance(hkl, int):
            return hkl
        else:
            hklList = self.getSymHKLs()  # !!! list, reduced by exclusions
            intl_hklIDs = np.asarray([i['hklID'] for i in self.hklDataList])
            intl_hklIDs_sorted = intl_hklIDs[~self.exclusions[self.tThSortInv]]
            dHKLInv = {}
            for iHKL, symHKLs in enumerate(hklList):
                idx = intl_hklIDs_sorted[iHKL] if master else iHKL
                for thisHKL in symHKLs.T:
                    dHKLInv[tuple(thisHKL)] = idx
            try:
                return dHKLInv[tuple(hkl)]
            except KeyError:
                raise RuntimeError(
                    f"hkl '{tuple(hkl)}' is not present in this material!"
                )

    def getHKLs(self, *hkl_ids: int, **kwargs) -> Union[List[str], np.ndarray]:
        """
        Returns the powder HKLs subject to specified options.

        Parameters
        ----------
        *hkl_ids : int
            Optional list of specific master hklIDs.
        **kwargs : dict
            One or more of the following keyword arguments:
                asStr : bool
                    If True, return a list of strings.  The default is False.
                thisTTh : scalar | None
                    If not None, only return hkls overlapping the specified
                    2-theta (in radians).  The default is None.
                allHKLs : bool
                    If True, then ignore exlcusions.  The default is False.

        Raises
        ------
        TypeError
            If an unknown kwarg is passed.
        RuntimeError
            If an invalid hklID is passed.

        Returns
        -------
        hkls : list | numpy.ndarray
            Either a list of hkls as strings (if asStr=True) or a vstacked
            array of hkls.

        Notes
        -----
        !!! the shape of the return value when asStr=False is the _transpose_
            of the typical return value for self.get_hkls() and self.hkls!
            This _may_ change to avoid confusion, but going to leave it for
            now so as not to break anything.

        2022/08/05 JVB:
            - Added functionality to handle optional hklID args
            - Updated docstring
        """
        # kwarg parsing
        opts = dict(asStr=False, thisTTh=None, allHKLs=False)
        if len(kwargs) > 0:
            # check keys
            for k, v in kwargs.items():
                if k not in opts:
                    raise TypeError(
                        f"getHKLs() got an unexpected keyword argument '{k}'"
                    )
            opts.update(kwargs)

        hkls = []
        if len(hkl_ids) == 0:
            for iHKLr, hklData in enumerate(self.hklDataList):
                if not opts['allHKLs']:
                    if not self._thisHKL(iHKLr):
                        continue
                if opts['thisTTh'] is not None:
                    tThLo, tThHi = self._getTThRange(iHKLr)
                    if opts['thisTTh'] < tThHi and opts['thisTTh'] > tThLo:
                        hkls.append(hklData['hkl'])
                else:
                    hkls.append(hklData['hkl'])
        else:
            # !!! changing behavior here; if the hkl_id is invalid, raises
            #     RuntimeError, and if allHKLs=True and the hkl_id is
            #     excluded, it also raises a RuntimeError
            all_hkl_ids = np.asarray([i['hklID'] for i in self.hklDataList])
            sorted_excl = self.exclusions[self.tThSortInv]
            idx = np.zeros(len(self.hklDataList), dtype=int)
            for i, hkl_id in enumerate(hkl_ids):
                # find ordinal index of current hklID
                try:
                    idx[i] = int(np.where(all_hkl_ids == hkl_id)[0])
                except TypeError:
                    raise RuntimeError(
                        f"Requested hklID '{hkl_id}'is invalid!"
                    )
                if sorted_excl[idx[i]] and not opts['allHKLs']:
                    raise RuntimeError(
                        f"Requested hklID '{hkl_id}' is excluded!"
                    )
                hkls.append(self.hklDataList[idx[i]]['hkl'])

        # handle output kwarg
        if opts['asStr']:
            return list(map(hklToStr, np.array(hkls)))
        else:
            return np.array(hkls)

    def getSymHKLs(
        self,
        asStr: Optional[bool] = False,
        withID: Optional[bool] = False,
        indices: Optional[List[int]] = None,
    ) -> Union[List[List[str]], List[np.ndarray]]:
        """
        Return all symmetry HKLs.

        Parameters
        ----------
        asStr : bool, optional
            If True, return the symmetry HKLs as strings. The default is False.
        withID : bool, optional
            If True, return the symmetry HKLs with the hklID. The default is
            False. Does nothing if asStr is True.
        indices : list[inr], optional
            Optional list of indices of hkls to include.

        Returns
        -------
        sym_hkls : list list of strings, or list of numpy.ndarray
            List of symmetry HKLs for each HKL, either as strings or as a
            vstacked array.
        """
        sym_hkls = []
        hkl_index = 0
        if indices is not None:
            indB = np.zeros(self.nHKLs, dtype=bool)
            indB[np.array(indices)] = True
        else:
            indB = np.ones(self.nHKLs, dtype=bool)
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self._thisHKL(iHKLr):
                continue
            if indB[hkl_index]:
                hkls = hklData['symHKLs']
                if asStr:
                    sym_hkls.append(list(map(hklToStr, np.array(hkls).T)))
                elif withID:
                    sym_hkls.append(
                        np.vstack(
                            [
                                np.tile(hklData['hklID'], (1, hkls.shape[1])),
                                hkls,
                            ]
                        )
                    )
                else:
                    sym_hkls.append(np.array(hkls))
            hkl_index += 1
        return sym_hkls

    @staticmethod
    def makeScatteringVectors(
        hkls: np.ndarray,
        rMat_c: np.ndarray,
        bMat: np.ndarray,
        wavelength: float,
        chiTilt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Static method for calculating g-vectors and scattering vector angles
        for specified hkls, subject to the bragg conditions specified by
        lattice vectors, orientation matrix, and wavelength

        Parameters
        ----------
        hkls : np.ndarray
            (3, n) array of hkls.
        rMat_c : np.ndarray
            (3, 3) rotation matrix from the crystal to the sample frame.
        bMat : np.ndarray, optional
            (3, 3) COB from reciprocal lattice frame to the crystal frame.
        wavelength : float
            xray wavelength in Angstroms.
        chiTilt : float, optional
            0 <= chiTilt <= 90 degrees, defaults to 0

        Returns
        -------
        gVec_s : np.ndarray
            (3, n) array of g-vectors (reciprocal lattice) in the sample frame.
        oangs0 : np.ndarray
            (3, n) array containing the feasible (2-theta, eta, ome) triplets
            for each input hkl (first solution)
        oangs1 : np.ndarray
            (3, n) array containing the feasible (2-theta, eta, ome) triplets
            for each input hkl (second solution)

        FIXME: must do testing on strained bMat
        """
        # arg munging
        chi = float(chiTilt) if chiTilt is not None else 0.0
        rMat_c = rMat_c.squeeze()

        # these are the reciprocal lattice vectors in the SAMPLE FRAME
        # ** NOTE **
        #   if strained, assumes that you handed it a bMat calculated from
        #   strained [a, b, c] in the CRYSTAL FRAME
        gVec_s = np.dot(rMat_c, np.dot(bMat, hkls))

        dim0 = gVec_s.shape[0]
        if dim0 != 3:
            raise ValueError(f'Number of lattice plane normal dims is {dim0}')

        # call model from transforms now
        oangs0, oangs1 = xfcapi.oscillAnglesOfHKLs(
            hkls.T, chi, rMat_c, bMat, wavelength
        )

        return gVec_s, oangs0.T, oangs1.T

    def _makeScatteringVectors(
        self,
        rMat: np.ndarray,
        bMat: Optional[np.ndarray] = None,
        chiTilt: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        modeled after QFromU.m
        """

        if bMat is None:
            bMat = self._latVecOps['B']

        Qs_vec = []
        Qs_ang0 = []
        Qs_ang1 = []
        for iHKLr, hklData in enumerate(self.hklDataList):
            if not self._thisHKL(iHKLr):
                continue
            thisQs, thisAng0, thisAng1 = PlaneData.makeScatteringVectors(
                hklData['symHKLs'],
                rMat,
                bMat,
                self._wavelength,
                chiTilt=chiTilt,
            )
            Qs_vec.append(thisQs)
            Qs_ang0.append(thisAng0)
            Qs_ang1.append(thisAng1)

        return Qs_vec, Qs_ang0, Qs_ang1


@deprecated(removal_date='2025-01-01')
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
    c1 = 1.0
    c2 = pi / 180.0

    eta0 = np.atleast_1d(eta0)
    tth0 = np.atleast_1d(tth0)
    ome0 = np.atleast_1d(ome0)

    if eta0.ndim != 1:
        raise RuntimeError('azimuthal input must be 1-D')

    npts = len(eta0)

    if tth0.ndim != 1:
        raise RuntimeError('Bragg angle input must be not 1-D')
    else:
        if len(tth0) != npts:
            if len(tth0) == 1:
                tth0 *= np.ones(npts)
            elif npts == 1:
                npts = len(tth0)
                eta0 *= np.ones(npts)
            else:
                raise RuntimeError(
                    'the azimuthal and Bragg angle inputs are inconsistent'
                )

    if len(ome0) == 0:
        ome0 = np.zeros(npts)  # dummy ome0
    elif len(ome0) == 1 and npts > 1:
        ome0 *= np.ones(npts)
    else:
        if len(ome0) != npts:
            raise RuntimeError(
                'your oscialltion angle input is inconsistent; '
                + f'it has length {len(ome0)} while it should be {npts}'
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
                    c1 = 180.0 / pi
                    c2 = 1.0
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
    eta0 = mapAngle(c1 * eta0, [-180, 180], units='degrees')
    if fableFlag:
        eta0 = 90 - eta0

    # must put args into RADIANS
    #   - eta0 is in DEGREES,
    #   - the others are in whatever was entered, hence c2
    eta0 = d2r * eta0
    tht0 = c2 * tth0 / 2
    if chi is not None:
        chi = c2 * chi
    else:
        chi = 0

    """
    SYSTEM SOLVE


    cos(chi)cos(eta)cos(theta)sin(x) - cos(chi)sin(theta)cos(x) \
        = sin(theta) - sin(chi)sin(eta)cos(theta)


    Identity: a sin x + b cos x = sqrt(a**2 + b**2) sin (x + alpha)

           /
           |      atan(b/a) for a > 0
     alpha <
           | pi + atan(b/a) for a < 0
           \

     => sin (x + alpha) = c / sqrt(a**2 + b**2)

     must use both branches for sin(x) = n:
     x = u (+ 2k*pi) | x = pi - u (+ 2k*pi)
    """
    cchi = np.cos(chi)
    schi = np.sin(chi)
    ceta = np.cos(eta0)
    seta = np.sin(eta0)
    ctht = np.cos(tht0)
    stht = np.sin(tht0)

    nchi = np.c_[0.0, cchi, schi].T

    gHat0_l = -np.vstack([ceta * ctht, seta * ctht, stht])

    a = cchi * ceta * ctht
    b = -cchi * stht
    c = stht + schi * seta * ctht

    # form solution
    abMag = np.sqrt(a * a + b * b)
    assert np.all(abMag > 0), "Beam vector specification is infeasible!"
    phaseAng = np.arctan2(b, a)
    rhs = c / abMag
    rhs[abs(rhs) > 1.0] = np.nan
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
    goodOnes = ~np.isnan(ome_min)

    numGood = np.sum(goodOnes)
    tmp_eta = np.empty(numGood)
    tmp_gvec = gHat0_l[:, goodOnes]
    for i in range(numGood):
        rchi = rotMatOfExpMap(np.tile(ome_min[goodOnes][i], (3, 1)) * nchi)
        gHat_l = np.dot(rchi, tmp_gvec[:, i].reshape(3, 1))
        tmp_eta[i] = np.arctan2(gHat_l[1], gHat_l[0])

    eta_min[goodOnes] = tmp_eta

    # everybody back to DEGREES!
    #     - ome1 is in RADIANS here
    #     - convert and put into [-180, 180]
    ome1 = mapAngle(
        mapAngle(r2d * ome_min, [-180, 180], units='degrees') + c1 * ome0,
        [-180, 180],
        units='degrees',
    )

    # put eta1 in [-180, 180]
    eta1 = mapAngle(r2d * eta_min, [-180, 180], units='degrees')

    if not outputDegrees:
        ome1 *= d2r
        eta1 *= d2r

    return ome1, eta1


def getDparms(
    lp: np.ndarray, lpTag: str, radians: Optional[bool] = True
) -> np.ndarray:
    """
    Utility routine for getting dparms, that is the lattice parameters
    without symmetry -- 'triclinic'

    Parameters
    ----------
    lp : np.ndarray
        Parsed lattice parameters
    lpTag : str
        Tag for the symmetry group of the lattice (from Laue group)
    radians : bool, optional
        Whether or not to use radians for angles, default is True

    Returns
    -------
    np.ndarray
        The lattice parameters without symmetry.
    """
    latVecOps = latticeVectors(lp, tag=lpTag, radians=radians)
    return latVecOps['dparms']


def lorentz_factor(tth: np.ndarray) -> np.ndarray:
    """
    05/26/2022 SS adding lorentz factor computation
    to the detector so that it can be compenstated for in the
    intensity correction

    Parameters
    ----------
    tth: np.ndarray
        2-theta of every pixel in radians

    Returns
    -------
    np.ndarray
        Lorentz factor for each pixel
    """

    theta = 0.5 * tth

    cth = np.cos(theta)
    sth2 = np.sin(theta) ** 2

    return 1.0 / (4.0 * cth * sth2)


def polarization_factor(
    tth: np.ndarray,
    unpolarized: Optional[bool] = True,
    eta: Optional[np.ndarray] = None,
    f_hor: Optional[float] = None,
    f_vert: Optional[float] = None,
) -> np.ndarray:
    """
    06/14/2021 SS adding lorentz polarization factor computation
    to the detector so that it can be compenstated for in the
    intensity correction

    05/26/2022 decoupling lorentz factor from polarization factor

    parameters: tth two theta of every pixel in radians
                if unpolarized is True, all subsequent arguments are optional
                eta azimuthal angle of every pixel
                f_hor fraction of horizontal polarization
                (~1 for XFELs)
                f_vert fraction of vertical polarization
                (~0 for XFELs)
    notice f_hor + f_vert = 1

    FIXME, called without parameters like eta, f_hor, f_vert, but they default
    to none in the current implementation, which will throw an error.
    """

    ctth2 = np.cos(tth) ** 2

    if unpolarized:
        return (1 + ctth2) / 2

    seta2 = np.sin(eta) ** 2
    ceta2 = np.cos(eta) ** 2
    return f_hor * (seta2 + ceta2 * ctth2) + f_vert * (ceta2 + seta2 * ctth2)
