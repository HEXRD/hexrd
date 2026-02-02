#! /usr/bin/env python
# ============================================================
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
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License (as published
# by the Free Software
# Foundation) version 2.1 dated February 1999.
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
# ============================================================
#
"""This module contains mappings from space group number to either
Hall or Hermann-Mauguin notation, as well as the inverse notations.

Space groups can be mapped to crystal class (one of 32 point groups)
and then to crystal system .

NOTES:

* Laue group is the cyrstal class if you add a center of symmetry.
  There are 11 Laue groups, determined directly from the point group.
* This module avoids the use of numpy and uses math module instead.
  That means the hkl lists are not numpy arrays, but simple lists of
  tuples.
* Rhombohedral lattices:

REFERENCES

1.  Mappings among space group number, Hall symbols and Hermann-Mauguin symbols.
http://cci.lbl.gov/sginfo/hall_symbols.html

2.  For mapping space group number to point group (crystal class in Schonflies
    notation)
http://en.wikipedia.org/wiki/Space_group

3.  Crystallography and crystal defects By Anthony Kelly, G. W. Groves, P. Kidd

4.  Contains point group from sgnum.
http://en.wikipedia.org/wiki/Space_group#Classification_systems_for_space_groups

5.  Point group to laue group
http://www.ruppweb.org/Xray/tutorial/32point.htm

6.  For discussion of rhombohedral lattice and "obverse"
and "reverse" settings for lattice parameters.
Crystal structure determination (book) By Werner Massa

TESTING

Run this module as main to generate all space groups and test
the HKL evaluation.
"""

from collections import OrderedDict
from math import sqrt, floor
from typing import TYPE_CHECKING

from hexrd.core import constants
from hexrd.core.material import symbols, symmetry

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from hexrd.core.material import Material

#
__all__ = ['SpaceGroup']


#
# ---------------------------------------------------CLASS:  SpaceGroup
#


class SpaceGroup:
    def __init__(self, sgnum):
        """Constructor for SpaceGroup


        INPUTS
        sgnum -- (int) space group number (between 1 and 230)
        """
        self.sgnum = sgnum

    def __str__(self):
        """Print information about the space group"""
        s = 'space group number:  %d\n' % self.sgnum
        s += '       Hall Symbol:  %s\n' % self.HallSymbol
        s += '   Hermann-Mauguin:  %s\n' % self.hermannMauguin
        s += '       Point Group:  %s\n' % self.pointGroup
        s += '        Laue Group:  %s\n' % self.laueGroup
        s += '      Lattice Type:  %s\n' % self.latticeType

        return s

    #
    # ============================== API
    #
    #                     ========== Properties
    #
    # property:  sgnum

    def _get_sgnum(self):
        """Get method for sgnum"""
        return self._sgnum

    def _set_sgnum(self, v):
        """Set method for sgnum"""
        self._sgnum = v
        self._Hall = lookupHall[v]
        #
        # Set point group and laue group
        #
        # * Point group dictionary maps a range of space group numbers
        #   to a Schoenflies string for the point group
        #
        for k in list(_pgDict.keys()):
            if v in k:
                pglg = _pgDict[k]
                self._pointGroup = pglg[0]
                self._laueGroup = pglg[1]

    sgnum = property(_get_sgnum, _set_sgnum, None, "Space group number")

    @property
    def laueGroup(self):
        """Schonflies symbol for Laue group (read only)"""
        return self._laueGroup

    @property
    def laueGroup_international(self):
        """Internationl symbol for Laue group (read only)"""
        return _laue_international[self._laueGroup]

    @property
    def pointGroup(self):
        """Schonflies symbol for point group (read only)"""
        return self._pointGroup

    @property
    def latticeType(self):
        """Lattice type

        Possible values are 'cubic', 'hexagonal', 'trigonal',
        'tetragonal', 'orthorhombic', 'monoclinic' and 'triclinic'

        Rhombohedral lattices are treated as trigonal using the
        "obverse" setting.
        """
        return _ltDict[self.laueGroup]

    @property
    def reqParams(self):
        """(read only) Zero-based indices of required lattice parameters"""
        return _rqpDict[self.latticeType][0]

    @property
    def hermannMauguin(self):
        """(read only) Hermann-Mauguin symbol"""
        return lookupHM[self.sgnum]

    @property
    def HallSymbol(self):
        """(read only) Hall symbol"""
        return self._Hall

    #                     ========== Public Methods
    #
    def sixLatticeParams(self, lparams):
        """
        Return the complete set of six lattice parameters from
        the abbreviated set


        INPUTS
        lparams -- (tuple) the abbreviated set of lattice parameters

        OUTPUTS
        sparams -- (tuple) the complete set of lattice parameters;
                           (a, b, c, alpha, beta, gamma)

        DESCRIPTION
        * Output angles are in degrees
        """
        return _rqpDict[self.latticeType][1](lparams)


#
# -----------------------------------------------END CLASS:  SpaceGroup
#


def _buildDict(hstr):
    """build the dictionaries from the notation string

    Returns two dictionaries:  one taking sg number to string
    and the inverse

    Takes the first Hall symbol it finds.  This is desired
    for the rhombohedral lattices so that they use the hexagonal
    convention.
    """
    d = dict()
    di = dict()
    hs = hstr.split('\n')
    for l in hs:
        li = l.strip()
        if li:
            nstr, hstr = li.split(None, 1)
            nstr = nstr.split(':', 1)[0]
            n = int(nstr)
            if n not in d:
                d[n] = hstr
            di[hstr] = n

    return d, di


lookupHall, Hall_to_sgnum = _buildDict(symbols.HALL_STR)
lookupHM, HM_to_sgnum = _buildDict(symbols.HM_STR)


def _map_sg_info(hstr):
    """build the dictionaries from the notation string
    Returns two dictionaries:  one taking sg id to string
    and the inverse.
    """
    d = OrderedDict()
    di = OrderedDict()
    hs = hstr.split('\n')
    for l in hs:
        li = l.strip()
        if li:
            sgid, altid = li.split(None, 1)
            d[sgid] = altid
            di[altid] = sgid

    return d, di


sgid_to_hall, hall_to_sgid = _map_sg_info(symbols.HALL_STR)
sgid_to_hm, hm_to_sgid = _map_sg_info(symbols.HM_STR)


# ==================== Point Groups/Laue Groups

# TODO: make sane mappings

laue_1 = 'ci'
laue_2 = 'c2h'
laue_3 = 'd2h'
laue_4 = 'c4h'
laue_5 = 'd4h'
laue_6 = 's6'
laue_7 = 'd3d'
laue_8 = 'c6h'
laue_9 = 'd6h'
laue_10 = 'th'
laue_11 = 'oh'

_laue_international = {
    laue_1: "-1",
    laue_2: "2/m",
    laue_3: "mmm",
    laue_4: "4/m",
    laue_5: "4/mmm",
    laue_6: "-3",
    laue_7: "-3m",
    laue_8: "6/m",
    laue_9: "6/mmm",
    laue_10: "m3",
    laue_11: "m3m",
}


def _sgrange(min, max):
    return tuple(range(min, max + 1))  # inclusive range


_pgDict = {
    _sgrange(1, 1): ('c1', laue_1),  # Triclinic
    _sgrange(2, 2): ('ci', laue_1),  # laue 1
    _sgrange(3, 5): ('c2', laue_2),  # Monoclinic
    _sgrange(6, 9): ('cs', laue_2),
    _sgrange(10, 15): ('c2h', laue_2),  # laue 2
    _sgrange(16, 24): ('d2', laue_3),  # Orthorhombic
    _sgrange(25, 46): ('c2v', laue_3),
    _sgrange(47, 74): ('d2h', laue_3),  # laue 3
    _sgrange(75, 80): ('c4', laue_4),  # Tetragonal
    _sgrange(81, 82): ('s4', laue_4),
    _sgrange(83, 88): ('c4h', laue_4),  # laue 4
    _sgrange(89, 98): ('d4', laue_5),
    _sgrange(99, 110): ('c4v', laue_5),
    _sgrange(111, 122): ('d2d', laue_5),
    _sgrange(123, 142): ('d4h', laue_5),  # laue 5
    _sgrange(143, 146): ('c3', laue_6),  # Trigonal
    _sgrange(147, 148): ('s6', laue_6),  # laue 6 [also c3i]
    _sgrange(149, 155): ('d3', laue_7),
    _sgrange(156, 161): ('c3v', laue_7),
    _sgrange(162, 167): ('d3d', laue_7),  # laue 7
    _sgrange(168, 173): ('c6', laue_8),  # Hexagonal
    _sgrange(174, 174): ('c3h', laue_8),
    _sgrange(175, 176): ('c6h', laue_8),  # laue 8
    _sgrange(177, 182): ('d6', laue_9),
    _sgrange(183, 186): ('c6v', laue_9),
    _sgrange(187, 190): ('d3h', laue_9),
    _sgrange(191, 194): ('d6h', laue_9),  # laue 9
    _sgrange(195, 199): ('t', laue_10),  # Cubic
    _sgrange(200, 206): ('th', laue_10),  # laue 10
    _sgrange(207, 214): ('o', laue_11),
    _sgrange(215, 220): ('td', laue_11),
    _sgrange(221, 230): ('oh', laue_11),  # laue 11
}

#
# Lattice type dictionary on Laue Group
# .  to replace Symmetry.ltypeOfLaueGroup()
# .  see also lparm.latticeVectors()
#
ltype_1 = 'triclinic'
ltype_2 = 'monoclinic'
ltype_3 = 'orthorhombic'
ltype_4 = 'tetragonal'
ltype_5 = 'trigonal'
ltype_6 = 'hexagonal'
ltype_7 = 'cubic'

_ltDict = {
    laue_1: ltype_1,
    laue_2: ltype_2,
    laue_3: ltype_3,
    laue_4: ltype_4,
    laue_5: ltype_4,
    laue_6: ltype_5,
    laue_7: ltype_5,
    laue_8: ltype_6,
    laue_9: ltype_6,
    laue_10: ltype_7,
    laue_11: ltype_7,
}


# Required parameters by lattice type
# * dictionary provides list of required indices with
#   a function that takes the reduced set of lattice parameters
#   to the full set
# * consistent with lparm.latticeVectors
#
_rqpDict = {
    ltype_1: (tuple(range(6)), lambda p: p),  # all 6
    # note beta
    ltype_2: ((0, 1, 2, 4), lambda p: (p[0], p[1], p[2], 90, p[3], 90)),
    ltype_3: ((0, 1, 2), lambda p: (p[0], p[1], p[2], 90, 90, 90)),
    ltype_4: ((0, 2), lambda p: (p[0], p[0], p[1], 90, 90, 90)),
    ltype_5: ((0, 2), lambda p: (p[0], p[0], p[1], 90, 90, 120)),
    ltype_6: ((0, 2), lambda p: (p[0], p[0], p[1], 90, 90, 120)),
    ltype_7: ((0,), lambda p: (p[0], p[0], p[0], 90, 90, 90)),
}


def get_symmetry_directions(mat: 'Material') -> NDArray[np.int32]:
    """
    helper function to get a list of primary,
    secondary and tertiary directions of the
    space group of mat. e.g. cubic systems have
        primary: [001]
        secondary: [111]
        tertiary: [110]

    For some symmetries like monoclinic and trigonal,
    some of the symmetry directions are not present. In
    that case, we have choden them to be the same as the
    crystal axis

    For trigonal systems, it is ALWAYS assumed that they
    are represented in the hexagonal basis. so the directions
    are the same for the two
    """
    ltype = mat.latticeType

    if ltype == "triclinic":
        primary = [0, 0, 1]
        secondary = [0, 1, 0]
        tertiary = [1, 0, 0]
    elif ltype == "monoclinic":
        primary = [0, 1, 0]
        secondary = [1, 0, 0]
        tertiary = [0, 0, 1]
    elif ltype == "orthorhombic":
        primary = [1, 0, 0]
        secondary = [0, 1, 0]
        tertiary = [1, 0, 1]
    elif ltype == "tetragonal":
        primary = [0, 0, 1]
        secondary = [1, 0, 0]
        tertiary = [1, 1, 0]
    elif ltype == "trigonal":
        # it is assumed that trigonal crystals are
        # represented in the hexagonal basis
        primary = [0, 0, 1]
        secondary = [1, 0, 0]
        tertiary = [1, 2, 0]
    elif ltype == "hexagonal":
        primary = [0, 0, 1]
        secondary = [1, 0, 0]
        tertiary = [1, 2, 0]
    elif ltype == "cubic":
        primary = [0, 0, 1]
        secondary = [1, 1, 1]
        tertiary = [1, 1, 0]

    return np.array([primary, secondary, tertiary], dtype=np.int32)


def Allowed_HKLs(sgnum: int, hkllist: NDArray[np.int32]) -> NDArray[np.int32]:
    """ Checks if a g vector is allowed by lattice centering, screw axis or glide plane
    """
    sg_hmsymbol = symbols.pstr_spacegroup[sgnum - 1].strip()
    symmorphic = False
    if sgnum in constants.sgnum_symmorphic:
        symmorphic = True

    hkllist = np.atleast_2d(hkllist)

    centering = sg_hmsymbol[0]
    if centering == 'P':
        # all reflections are allowed
        mask = np.ones([hkllist.shape[0]], dtype=bool)
    elif centering == 'F':
        # same parity
        seo = np.sum(np.mod(hkllist + 100, 2), axis=1)
        mask = np.logical_not(np.logical_or(seo == 1, seo == 2))
    elif centering == 'I':
        # sum is even
        seo = np.mod(np.sum(hkllist, axis=1) + 100, 2)
        mask = seo == 0
    elif centering == 'A':
        # k+l is even
        seo = np.mod(np.sum(hkllist[:, 1:3], axis=1) + 100, 2)
        mask = seo == 0
    elif centering == 'B':
        # h+l is even
        seo = np.mod(hkllist[:, 0] + hkllist[:, 2] + 100, 2)
        mask = seo == 0
    elif centering == 'C':
        # h+k is even
        seo = np.mod(hkllist[:, 0] + hkllist[:, 1] + 100, 2)
        mask = seo == 0
    elif centering == 'R':
        # -h+k+l is divisible by 3
        seo = np.mod(-hkllist[:, 0] + hkllist[:, 1] + hkllist[:, 2] + 90, 3)
        mask = seo == 0
    else:
        raise ValueError(f'Unknown lattice centering: "{centering}"')

    hkls = hkllist[mask, :]
    if not symmorphic:
        hkls = NonSymmorphicAbsences(sgnum, hkls)
    return hkls.astype(np.int32)


def omitscrewaxisabsences(sgnum, hkllist, ax, iax):
    """
    this function encodes the table on pg 48 of
    international table of crystallography vol A
    the systematic absences due to different screw
    axis is encoded here.
    iax encodes the primary, secondary or tertiary axis
    iax == 0 : primary
    iax == 1 : secondary
    iax == 2 : tertiary
    @NOTE: only unique b axis in monoclinic systems
    implemented as thats the standard setting
    """
    latticeType = symmetry.latticeType(sgnum)

    if latticeType == 'triclinic':
        """
        no systematic absences for the triclinic crystals
        """
        pass

    elif latticeType == 'monoclinic':
        if ax != '2_1':
            raise RuntimeError(
                'omitscrewaxisabsences: monoclinic systems\
                 can only have 2_1 screw axis'
            )
        """
            only unique b-axis will be encoded
            it is the users responsibility to input
            lattice parameters in the standard setting
            with b-axis having the 2-fold symmetry
        """
        if iax == 1:
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
            mask2 = np.mod(hkllist[:, 1] + 100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        else:
            raise RuntimeError(
                'omitscrewaxisabsences: only b-axis\
                 can have 2_1 screw axis'
            )

    elif latticeType == 'orthorhombic':
        if ax != '2_1':
            raise RuntimeError(
                'omitscrewaxisabsences: orthorhombic systems\
                 can only have 2_1 screw axis'
            )
        """
        2_1 screw on primary axis
        h00 ; h = 2n
        """
        if iax == 0:
            mask1 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
            mask2 = np.mod(hkllist[:, 0] + 100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif iax == 1:
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
            mask2 = np.mod(hkllist[:, 1] + 100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif iax == 2:
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            mask2 = np.mod(hkllist[:, 2] + 100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

    elif latticeType == 'tetragonal':
        if iax == 0:
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            if ax == '4_2':
                mask2 = np.mod(hkllist[:, 2] + 100, 2) != 0
            elif ax in ['4_1', '4_3']:
                mask2 = np.mod(hkllist[:, 2] + 100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif iax == 1:
            mask1 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
            mask2 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
            if ax == '2_1':
                mask3 = np.mod(hkllist[:, 0] + 100, 2) != 0
                mask4 = np.mod(hkllist[:, 1] + 100, 2) != 0
            mask1 = np.logical_not(np.logical_and(mask1, mask3))
            mask2 = np.logical_not(np.logical_and(mask2, mask4))
            mask = ~np.logical_or(~mask1, ~mask2)
            hkllist = hkllist[mask, :]

    elif latticeType == 'trigonal':
        mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
        if iax == 0:
            if ax in ['3_1', '3_2']:
                mask2 = np.mod(hkllist[:, 2] + 90, 3) != 0
        else:
            raise RuntimeError(
                'omitscrewaxisabsences: trigonal \
                systems can only have screw axis'
            )
        mask = np.logical_not(np.logical_and(mask1, mask2))
        hkllist = hkllist[mask, :]

    elif latticeType == 'hexagonal':
        mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
        if iax == 0:
            if ax == '6_3':
                mask2 = np.mod(hkllist[:, 2] + 100, 2) != 0
            elif ax in ['3_1', '3_2', '6_2', '6_4']:
                mask2 = np.mod(hkllist[:, 2] + 90, 3) != 0
            elif ax in ['6_1', '6_5']:
                mask2 = np.mod(hkllist[:, 2] + 120, 6) != 0
        else:
            raise RuntimeError(
                'omitscrewaxisabsences: hexagonal \
                systems can only have screw axis'
            )
        mask = np.logical_not(np.logical_and(mask1, mask2))
        hkllist = hkllist[mask, :]

    elif latticeType == 'cubic':
        mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
        mask2 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
        mask3 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
        if ax in ['2_1', '4_2']:
            mask4 = np.mod(hkllist[:, 2] + 100, 2) != 0
            mask5 = np.mod(hkllist[:, 1] + 100, 2) != 0
            mask6 = np.mod(hkllist[:, 0] + 100, 2) != 0
        elif ax in ['4_1', '4_3']:
            mask4 = np.mod(hkllist[:, 2] + 100, 4) != 0
            mask5 = np.mod(hkllist[:, 1] + 100, 4) != 0
            mask6 = np.mod(hkllist[:, 0] + 100, 4) != 0
        mask1 = np.logical_not(np.logical_and(mask1, mask4))
        mask2 = np.logical_not(np.logical_and(mask2, mask5))
        mask3 = np.logical_not(np.logical_and(mask3, mask6))
        mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
        hkllist = hkllist[mask, :]
    return hkllist.astype(np.int32)


def omitglideplaneabsences(sgnum, hkllist, plane, ip):
    """
    this function encodes the table on pg 47 of
    international table of crystallography vol A
    the systematic absences due to different glide
    planes is encoded here.
    ip encodes the primary, secondary or tertiary plane normal
    ip == 0 : primary
    ip == 1 : secondary
    ip == 2 : tertiary
    @NOTE: only unique b axis in monoclinic systems
    implemented as thats the standard setting
    """
    latticeType = symmetry.latticeType(sgnum)

    if latticeType == 'triclinic':
        pass

    elif latticeType == 'monoclinic':
        if ip == 1:
            mask1 = hkllist[:, 1] == 0
            if plane == 'c':
                mask2 = np.mod(hkllist[:, 2] + 100, 2) != 0
            elif plane == 'a':
                mask2 = np.mod(hkllist[:, 0] + 100, 2) != 0
            elif plane == 'n':
                mask2 = np.mod(hkllist[:, 0] + hkllist[:, 2] + 100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

    elif latticeType == 'orthorhombic':
        if ip == 0:
            mask1 = hkllist[:, 0] == 0
            if plane == 'b':
                mask2 = np.mod(hkllist[:, 1] + 100, 2) != 0
            elif plane == 'c':
                mask2 = np.mod(hkllist[:, 2] + 100, 2) != 0
            elif plane == 'n':
                mask2 = np.mod(hkllist[:, 1] + hkllist[:, 2] + 100, 2) != 0
            elif plane == 'd':
                mask2 = np.mod(hkllist[:, 1] + hkllist[:, 2] + 100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

        elif ip == 1:
            mask1 = hkllist[:, 1] == 0
            if plane == 'c':
                mask2 = np.mod(hkllist[:, 2] + 100, 2) != 0
            elif plane == 'a':
                mask2 = np.mod(hkllist[:, 0] + 100, 2) != 0
            elif plane == 'n':
                mask2 = np.mod(hkllist[:, 0] + hkllist[:, 2] + 100, 2) != 0
            elif plane == 'd':
                mask2 = np.mod(hkllist[:, 0] + hkllist[:, 2] + 100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

        elif ip == 2:
            mask1 = hkllist[:, 2] == 0
            if plane == 'a':
                mask2 = np.mod(hkllist[:, 0] + 100, 2) != 0
            elif plane == 'b':
                mask2 = np.mod(hkllist[:, 1] + 100, 2) != 0
            elif plane == 'n':
                mask2 = np.mod(hkllist[:, 0] + hkllist[:, 1] + 100, 2) != 0
            elif plane == 'd':
                mask2 = np.mod(hkllist[:, 0] + hkllist[:, 1] + 100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

    elif latticeType == 'tetragonal':
        if ip == 0:
            mask1 = hkllist[:, 2] == 0
            if plane == 'a':
                mask2 = np.mod(hkllist[:, 0] + 100, 2) != 0
            elif plane == 'b':
                mask2 = np.mod(hkllist[:, 1] + 100, 2) != 0
            elif plane == 'n':
                mask2 = np.mod(hkllist[:, 0] + hkllist[:, 1] + 100, 2) != 0
            elif plane == 'd':
                mask2 = np.mod(hkllist[:, 0] + hkllist[:, 1] + 100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

        elif ip == 1:
            mask1 = hkllist[:, 0] == 0
            mask2 = hkllist[:, 1] == 0
            if plane in ['a', 'b']:
                mask3 = np.mod(hkllist[:, 1] + 100, 2) != 0
                mask4 = np.mod(hkllist[:, 0] + 100, 2) != 0
            elif plane == 'c':
                mask3 = np.mod(hkllist[:, 2] + 100, 2) != 0
                mask4 = mask3
            elif plane == 'n':
                mask3 = np.mod(hkllist[:, 1] + hkllist[:, 2] + 100, 2) != 0
                mask4 = np.mod(hkllist[:, 0] + hkllist[:, 2] + 100, 2) != 0
            elif plane == 'd':
                mask3 = np.mod(hkllist[:, 1] + hkllist[:, 2] + 100, 4) != 0
                mask4 = np.mod(hkllist[:, 0] + hkllist[:, 2] + 100, 4) != 0
            mask1 = np.logical_not(np.logical_and(mask1, mask3))
            mask2 = np.logical_not(np.logical_and(mask2, mask4))
            mask = ~np.logical_or(~mask1, ~mask2)
            hkllist = hkllist[mask, :]

        elif ip == 2:
            mask1 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 1])
            if plane in ['c', 'n']:
                mask2 = np.mod(hkllist[:, 2] + 100, 2) != 0
            elif plane == 'd':
                mask2 = np.mod(2 * hkllist[:, 0] + hkllist[:, 2] + 100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

    elif latticeType == 'trigonal':
        if plane != 'c':
            raise RuntimeError(
                'omitglideplaneabsences: only c-glide \
                allowed for trigonal systems'
            )
        if ip == 1:
            mask1 = hkllist[:, 0] == 0
            mask2 = hkllist[:, 1] == 0
            mask3 = hkllist[:, 0] == -hkllist[:, 1]
            if plane == 'c':
                mask4 = np.mod(hkllist[:, 2] + 100, 2) != 0
            else:
                raise RuntimeError(
                    'omitglideplaneabsences: only c-glide \
                    allowed for trigonal systems'
                )

        elif ip == 2:
            mask1 = hkllist[:, 1] == hkllist[:, 0]
            mask2 = hkllist[:, 0] == -2 * hkllist[:, 1]
            mask3 = -2 * hkllist[:, 0] == hkllist[:, 1]
            if plane == 'c':
                mask4 = np.mod(hkllist[:, 2] + 100, 2) != 0
            else:
                raise RuntimeError(
                    'omitglideplaneabsences: only c-glide \
                    allowed for trigonal systems'
                )
        mask1 = np.logical_and(mask1, mask4)
        mask2 = np.logical_and(mask2, mask4)
        mask3 = np.logical_and(mask3, mask4)
        mask = np.logical_not(np.logical_or(mask1, np.logical_or(mask2, mask3)))
        hkllist = hkllist[mask, :]

    elif latticeType == 'hexagonal':
        if plane != 'c':
            raise RuntimeError(
                'omitglideplaneabsences: only c-glide \
                allowed for hexagonal systems'
            )
        if ip == 2:
            mask1 = hkllist[:, 0] == hkllist[:, 1]
            mask2 = hkllist[:, 0] == -2 * hkllist[:, 1]
            mask3 = -2 * hkllist[:, 0] == hkllist[:, 1]
            mask4 = np.mod(hkllist[:, 2] + 100, 2) != 0
            mask1 = np.logical_and(mask1, mask4)
            mask2 = np.logical_and(mask2, mask4)
            mask3 = np.logical_and(mask3, mask4)
            mask = np.logical_not(np.logical_or(mask1, np.logical_or(mask2, mask3)))

        elif ip == 1:
            mask1 = hkllist[:, 1] == 0
            mask2 = hkllist[:, 0] == 0
            mask3 = hkllist[:, 1] == -hkllist[:, 0]
            mask4 = np.mod(hkllist[:, 2] + 100, 2) != 0
        mask1 = np.logical_and(mask1, mask4)
        mask2 = np.logical_and(mask2, mask4)
        mask3 = np.logical_and(mask3, mask4)
        mask = np.logical_not(np.logical_or(mask1, np.logical_or(mask2, mask3)))
        hkllist = hkllist[mask, :]

    elif latticeType == 'cubic':
        if ip == 0:
            mask1 = hkllist[:, 0] == 0
            mask2 = hkllist[:, 1] == 0
            mask3 = hkllist[:, 2] == 0
            mask4 = np.mod(hkllist[:, 0] + 100, 2) != 0
            mask5 = np.mod(hkllist[:, 1] + 100, 2) != 0
            mask6 = np.mod(hkllist[:, 2] + 100, 2) != 0
            if plane == 'a':
                mask1 = np.logical_or(
                    np.logical_and(mask1, mask5), np.logical_and(mask1, mask6)
                )
                mask2 = np.logical_or(
                    np.logical_and(mask2, mask4), np.logical_and(mask2, mask6)
                )
                mask3 = np.logical_and(mask3, mask4)
                mask = np.logical_not(np.logical_or(mask1, np.logical_or(mask2, mask3)))
            elif plane == 'b':
                mask1 = np.logical_and(mask1, mask5)
                mask3 = np.logical_and(mask3, mask5)
                mask = np.logical_not(np.logical_or(mask1, mask3))
            elif plane == 'c':
                mask1 = np.logical_and(mask1, mask6)
                mask2 = np.logical_and(mask2, mask6)
                mask = np.logical_not(np.logical_or(mask1, mask2))
            elif plane == 'n':
                mask4 = np.mod(hkllist[:, 1] + hkllist[:, 2] + 100, 2) != 0
                mask5 = np.mod(hkllist[:, 0] + hkllist[:, 2] + 100, 2) != 0
                mask6 = np.mod(hkllist[:, 0] + hkllist[:, 1] + 100, 2) != 0
                mask1 = np.logical_not(np.logical_and(mask1, mask4))
                mask2 = np.logical_not(np.logical_and(mask2, mask5))
                mask3 = np.logical_not(np.logical_and(mask3, mask6))
                mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
            elif plane == 'd':
                mask4 = np.mod(hkllist[:, 1] + hkllist[:, 2] + 100, 4) != 0
                mask5 = np.mod(hkllist[:, 0] + hkllist[:, 2] + 100, 4) != 0
                mask6 = np.mod(hkllist[:, 0] + hkllist[:, 1] + 100, 4) != 0
                mask1 = np.logical_not(np.logical_and(mask1, mask4))
                mask2 = np.logical_not(np.logical_and(mask2, mask5))
                mask3 = np.logical_not(np.logical_and(mask3, mask6))
                mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
            else:
                raise RuntimeError(
                    'omitglideplaneabsences: unknown glide \
                    plane encountered.'
                )
            hkllist = hkllist[mask, :]

        if ip == 2:
            mask1 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 1])
            mask2 = np.abs(hkllist[:, 1]) == np.abs(hkllist[:, 2])
            mask3 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 2])
            if plane in ['a', 'b', 'c', 'n']:
                mask4 = np.mod(hkllist[:, 2] + 100, 2) != 0
                mask5 = np.mod(hkllist[:, 0] + 100, 2) != 0
                mask6 = np.mod(hkllist[:, 1] + 100, 2) != 0
            elif plane == 'd':
                mask4 = np.mod(2 * hkllist[:, 0] + hkllist[:, 2] + 100, 4) != 0
                mask5 = np.mod(hkllist[:, 0] + 2 * hkllist[:, 1] + 100, 4) != 0
                mask6 = np.mod(2 * hkllist[:, 0] + hkllist[:, 1] + 100, 4) != 0
            else:
                raise RuntimeError(
                    'omitglideplaneabsences: unknown glide \
                    plane encountered.'
                )
            mask1 = np.logical_not(np.logical_and(mask1, mask4))
            mask2 = np.logical_not(np.logical_and(mask2, mask5))
            mask3 = np.logical_not(np.logical_and(mask3, mask6))
            mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
            hkllist = hkllist[mask, :]

    return hkllist


def NonSymmorphicAbsences(sgnum, hkllist):
    """
    this function prunes hkl list for the screw axis and glide
    plane absences
    """
    planes = constants.SYS_AB[sgnum][0]
    for ip, p in enumerate(planes):
        if p != '':
            hkllist = omitglideplaneabsences(sgnum, hkllist, p, ip)
    axes = constants.SYS_AB[sgnum][1]
    for iax, ax in enumerate(axes):
        if ax != '':
            hkllist = omitscrewaxisabsences(sgnum, hkllist, ax, iax)
    return hkllist


#
# ================================================== HKL Enumeration
#


def _getHKLsBySS(ss):
    """Return a list of HKLs with a given sum of squares'

    ss - (int) sum of squares

    """

    #
    #  NOTE:  the loop below could be speeded up by requiring
    #         h >= k > = l, and then applying all permutations
    #         and sign changes.  Could possibly save up to
    #         a factor of 48.
    #
    def pmrange(n):
        return list(range(n, -(n + 1), -1))  # plus/minus range

    def iroot(n):
        return int(floor(sqrt(n)))  # integer square root

    hkls = []
    hmax = iroot(ss)
    for h in pmrange(hmax):
        ss2 = ss - h * h
        kmax = iroot(ss2)
        for k in pmrange(kmax):
            rem = ss2 - k * k
            if rem == 0:
                hkls.append((h, k, 0))
            else:
                l = iroot(rem)
                if l * l == rem:
                    hkls += [(h, k, l), (h, k, -l)]

    return hkls

