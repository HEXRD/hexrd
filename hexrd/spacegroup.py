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
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free Software
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
"""Interface with sglite for hkl generation and Laue group determination

This module contains mappings from space group number to either
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

from hexrd.extensions import sglite
from hexrd import symbols, constants, symmetry
import numpy as np
#
__all__ = ['SpaceGroup']


#
# ---------------------------------------------------CLASS:  SpaceGroup
#


class SpaceGroup(object):
    """Wrapper on sglite
    """
    def __init__(self, sgnum):
        """Constructor for SpaceGroup


        INPUTS
        sgnum -- (int) space group number (between 1 and 230)
        """
        self.sgnum  = sgnum  # call before sglite (sets Hall symbol)
        #
        # Do not include SgOps type as target, as that will make
        # this class unpicklable.
        #
        # self._verifySGNum() # test passed on all symmetry groups
        #
        return

    def __str__(self):
        """Print information about the space group"""
        ind = 5*' '
        s  = 'space group number:  %d\n' % self.sgnum
        s += '       Hall Symbol:  %s\n' % self.HallSymbol
        s += '   Hermann-Mauguin:  %s\n' % self.hermannMauguin
        s += '       Point Group:  %s\n' % self.pointGroup
        s += '        Laue Group:  %s\n' % self.laueGroup
        s += '      Lattice Type:  %s\n' % self.latticeType

        return s

    def _verifySGNum(self):
        """verify that sglite agrees with space group number"""
        sgops = self.SgOps
        d = sgops.getSpaceGroupType()
        if not d['SgNumber'] == self.sgnum:
            raise ValueError('space group number failed consistency check')

        return
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
                pass
            pass

        return

    sgnum = property(_get_sgnum, _set_sgnum, None,
                     "Space group number")

    @property
    def laueGroup(self):
        """Schonflies symbol for Laue group (read only)"""
        return self._laueGroup

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

    # property:  SgOps

    @property
    def SgOps(self):
        """(read only) An sglite.SgOps instance"""
        tmpSG = sglite.SgOps()
        tmpSG.__init__(HallSymbol=self._Hall)
        return tmpSG

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

    def getHKLs(self, ssmax):
        """Return a list of HKLs with a cutoff sum of square

        INPUTS
        ssmax -- cutoff sum of squares

        OUTPUTS
        hkls -- a list of all HKLs with sum of squares less than
                or equal to the cutoff, excluding systematic
                absences and symmetrically equivalent hkls

        DESCRIPTION
        """
        #
        # Not sure what the cut parameters are, but they seem
        # to involve octants to look for equivalent HKLs.
        # For sg(1), they are (-1, -1, -1) and for sg(230)
        # they are (0, 0, 0).  The argument to getCutParameters()
        # is 'FriedelSymmetry'---guessing this is an on/off switch.
        #
        sgops = self.SgOps
        cutp = sgops.getCutParameters(0)
        myHKLs = []
        for ssm in range(1, ssmax + 1):
            #
            #  Find all HKLs of a given magnitude (ssm)
            #
            for hkl in _getHKLsBySS(ssm):
                if sgops.isSysAbsMIx(hkl):
                    continue
                master, mate = sgops.get_MasterMIx_and_MateID(cutp, hkl)
                if master == hkl:
                    myHKLs.append(hkl)
                    pass
                pass
            pass  # ssm loop

        return myHKLs
    #
    pass  # end class


#
# -----------------------------------------------END CLASS:  SpaceGroup
#


#
# Hall Symbols copied from:
#
_hallStr = r"""
     1         P 1
     2        -P 1
     3:b       P 2y
     3:c       P 2
     3:a       P 2x
     4:b       P 2yb
     4:c       P 2c
     4:a       P 2xa
     5:b1      C 2y
     5:b2      A 2y
     5:b3      I 2y
     5:c1      A 2
     5:c2      B 2
     5:c3      I 2
     5:a1      B 2x
     5:a2      C 2x
     5:a3      I 2x
     6:b       P -2y
     6:c       P -2
     6:a       P -2x
     7:b1      P -2yc
     7:b2      P -2yac
     7:b3      P -2ya
     7:c1      P -2a
     7:c2      P -2ab
     7:c3      P -2b
     7:a1      P -2xb
     7:a2      P -2xbc
     7:a3      P -2xc
     8:b1      C -2y
     8:b2      A -2y
     8:b3      I -2y
     8:c1      A -2
     8:c2      B -2
     8:c3      I -2
     8:a1      B -2x
     8:a2      C -2x
     8:a3      I -2x
     9:b1      C -2yc
     9:b2      A -2yac
     9:b3      I -2ya
     9:-b1     A -2ya
     9:-b2     C -2ybc
     9:-b3     I -2yc
     9:c1      A -2a
     9:c2      B -2bc
     9:c3      I -2b
     9:-c1     B -2b
     9:-c2     A -2ac
     9:-c3     I -2a
     9:a1      B -2xb
     9:a2      C -2xbc
     9:a3      I -2xc
     9:-a1     C -2xc
     9:-a2     B -2xbc
     9:-a3     I -2xb
    10:b      -P 2y
    10:c      -P 2
    10:a      -P 2x
    11:b      -P 2yb
    11:c      -P 2c
    11:a      -P 2xa
    12:b1     -C 2y
    12:b2     -A 2y
    12:b3     -I 2y
    12:c1     -A 2
    12:c2     -B 2
    12:c3     -I 2
    12:a1     -B 2x
    12:a2     -C 2x
    12:a3     -I 2x
    13:b1     -P 2yc
    13:b2     -P 2yac
    13:b3     -P 2ya
    13:c1     -P 2a
    13:c2     -P 2ab
    13:c3     -P 2b
    13:a1     -P 2xb
    13:a2     -P 2xbc
    13:a3     -P 2xc
    14:b1     -P 2ybc
    14:b2     -P 2yn
    14:b3     -P 2yab
    14:c1     -P 2ac
    14:c2     -P 2n
    14:c3     -P 2bc
    14:a1     -P 2xab
    14:a2     -P 2xn
    14:a3     -P 2xac
    15:b1     -C 2yc
    15:b2     -A 2yac
    15:b3     -I 2ya
    15:-b1    -A 2ya
    15:-b2    -C 2ybc
    15:-b3    -I 2yc
    15:c1     -A 2a
    15:c2     -B 2bc
    15:c3     -I 2b
    15:-c1    -B 2b
    15:-c2    -A 2ac
    15:-c3    -I 2a
    15:a1     -B 2xb
    15:a2     -C 2xbc
    15:a3     -I 2xc
    15:-a1    -C 2xc
    15:-a2    -B 2xbc
    15:-a3    -I 2xb
    16         P 2 2
    17         P 2c 2
    17:cab     P 2a 2a
    17:bca     P 2 2b
    18         P 2 2ab
    18:cab     P 2bc 2
    18:bca     P 2ac 2ac
    19         P 2ac 2ab
    20         C 2c 2
    20:cab     A 2a 2a
    20:bca     B 2 2b
    21         C 2 2
    21:cab     A 2 2
    21:bca     B 2 2
    22         F 2 2
    23         I 2 2
    24         I 2b 2c
    25         P 2 -2
    25:cab     P -2 2
    25:bca     P -2 -2
    26         P 2c -2
    26:ba-c    P 2c -2c
    26:cab     P -2a 2a
    26:-cba    P -2 2a
    26:bca     P -2 -2b
    26:a-cb    P -2b -2
    27         P 2 -2c
    27:cab     P -2a 2
    27:bca     P -2b -2b
    28         P 2 -2a
    28:ba-c    P 2 -2b
    28:cab     P -2b 2
    28:-cba    P -2c 2
    28:bca     P -2c -2c
    28:a-cb    P -2a -2a
    29         P 2c -2ac
    29:ba-c    P 2c -2b
    29:cab     P -2b 2a
    29:-cba    P -2ac 2a
    29:bca     P -2bc -2c
    29:a-cb    P -2a -2ab
    30         P 2 -2bc
    30:ba-c    P 2 -2ac
    30:cab     P -2ac 2
    30:-cba    P -2ab 2
    30:bca     P -2ab -2ab
    30:a-cb    P -2bc -2bc
    31         P 2ac -2
    31:ba-c    P 2bc -2bc
    31:cab     P -2ab 2ab
    31:-cba    P -2 2ac
    31:bca     P -2 -2bc
    31:a-cb    P -2ab -2
    32         P 2 -2ab
    32:cab     P -2bc 2
    32:bca     P -2ac -2ac
    33         P 2c -2n
    33:ba-c    P 2c -2ab
    33:cab     P -2bc 2a
    33:-cba    P -2n 2a
    33:bca     P -2n -2ac
    33:a-cb    P -2ac -2n
    34         P 2 -2n
    34:cab     P -2n 2
    34:bca     P -2n -2n
    35         C 2 -2
    35:cab     A -2 2
    35:bca     B -2 -2
    36         C 2c -2
    36:ba-c    C 2c -2c
    36:cab     A -2a 2a
    36:-cba    A -2 2a
    36:bca     B -2 -2b
    36:a-cb    B -2b -2
    37         C 2 -2c
    37:cab     A -2a 2
    37:bca     B -2b -2b
    38         A 2 -2
    38:ba-c    B 2 -2
    38:cab     B -2 2
    38:-cba    C -2 2
    38:bca     C -2 -2
    38:a-cb    A -2 -2
    39         A 2 -2c
    39:ba-c    B 2 -2c
    39:cab     B -2c 2
    39:-cba    C -2b 2
    39:bca     C -2b -2b
    39:a-cb    A -2c -2c
    40         A 2 -2a
    40:ba-c    B 2 -2b
    40:cab     B -2b 2
    40:-cba    C -2c 2
    40:bca     C -2c -2c
    40:a-cb    A -2a -2a
    41         A 2 -2ac
    41:ba-c    B 2 -2bc
    41:cab     B -2bc 2
    41:-cba    C -2bc 2
    41:bca     C -2bc -2bc
    41:a-cb    A -2ac -2ac
    42         F 2 -2
    42:cab     F -2 2
    42:bca     F -2 -2
    43         F 2 -2d
    43:cab     F -2d 2
    43:bca     F -2d -2d
    44         I 2 -2
    44:cab     I -2 2
    44:bca     I -2 -2
    45         I 2 -2c
    45:cab     I -2a 2
    45:bca     I -2b -2b
    46         I 2 -2a
    46:ba-c    I 2 -2b
    46:cab     I -2b 2
    46:-cba    I -2c 2
    46:bca     I -2c -2c
    46:a-cb    I -2a -2a
    47        -P 2 2
    48:1       P 2 2 -1n
    48:2      -P 2ab 2bc
    49        -P 2 2c
    49:cab    -P 2a 2
    49:bca    -P 2b 2b
    50:1       P 2 2 -1ab
    50:2      -P 2ab 2b
    50:1cab    P 2 2 -1bc
    50:2cab   -P 2b 2bc
    50:1bca    P 2 2 -1ac
    50:2bca   -P 2a 2c
    51        -P 2a 2a
    51:ba-c   -P 2b 2
    51:cab    -P 2 2b
    51:-cba   -P 2c 2c
    51:bca    -P 2c 2
    51:a-cb   -P 2 2a
    52        -P 2a 2bc
    52:ba-c   -P 2b 2n
    52:cab    -P 2n 2b
    52:-cba   -P 2ab 2c
    52:bca    -P 2ab 2n
    52:a-cb   -P 2n 2bc
    53        -P 2ac 2
    53:ba-c   -P 2bc 2bc
    53:cab    -P 2ab 2ab
    53:-cba   -P 2 2ac
    53:bca    -P 2 2bc
    53:a-cb   -P 2ab 2
    54        -P 2a 2ac
    54:ba-c   -P 2b 2c
    54:cab    -P 2a 2b
    54:-cba   -P 2ac 2c
    54:bca    -P 2bc 2b
    54:a-cb   -P 2b 2ab
    55        -P 2 2ab
    55:cab    -P 2bc 2
    55:bca    -P 2ac 2ac
    56        -P 2ab 2ac
    56:cab    -P 2ac 2bc
    56:bca    -P 2bc 2ab
    57        -P 2c 2b
    57:ba-c   -P 2c 2ac
    57:cab    -P 2ac 2a
    57:-cba   -P 2b 2a
    57:bca    -P 2a 2ab
    57:a-cb   -P 2bc 2c
    58        -P 2 2n
    58:cab    -P 2n 2
    58:bca    -P 2n 2n
    59:1       P 2 2ab -1ab
    59:2      -P 2ab 2a
    59:1cab    P 2bc 2 -1bc
    59:2cab   -P 2c 2bc
    59:1bca    P 2ac 2ac -1ac
    59:2bca   -P 2c 2a
    60        -P 2n 2ab
    60:ba-c   -P 2n 2c
    60:cab    -P 2a 2n
    60:-cba   -P 2bc 2n
    60:bca    -P 2ac 2b
    60:a-cb   -P 2b 2ac
    61        -P 2ac 2ab
    61:ba-c   -P 2bc 2ac
    62        -P 2ac 2n
    62:ba-c   -P 2bc 2a
    62:cab    -P 2c 2ab
    62:-cba   -P 2n 2ac
    62:bca    -P 2n 2a
    62:a-cb   -P 2c 2n
    63        -C 2c 2
    63:ba-c   -C 2c 2c
    63:cab    -A 2a 2a
    63:-cba   -A 2 2a
    63:bca    -B 2 2b
    63:a-cb   -B 2b 2
    64        -C 2bc 2
    64:ba-c   -C 2bc 2bc
    64:cab    -A 2ac 2ac
    64:-cba   -A 2 2ac
    64:bca    -B 2 2bc
    64:a-cb   -B 2bc 2
    65        -C 2 2
    65:cab    -A 2 2
    65:bca    -B 2 2
    66        -C 2 2c
    66:cab    -A 2a 2
    66:bca    -B 2b 2b
    67        -C 2b 2
    67:ba-c   -C 2b 2b
    67:cab    -A 2c 2c
    67:-cba   -A 2 2c
    67:bca    -B 2 2c
    67:a-cb   -B 2c 2
    68:1       C 2 2 -1bc
    68:2      -C 2b 2bc
    68:1ba-c   C 2 2 -1bc
    68:2ba-c  -C 2b 2c
    68:1cab    A 2 2 -1ac
    68:2cab   -A 2a 2c
    68:1-cba   A 2 2 -1ac
    68:2-cba  -A 2ac 2c
    68:1bca    B 2 2 -1bc
    68:2bca   -B 2bc 2b
    68:1a-cb   B 2 2 -1bc
    68:2a-cb  -B 2b 2bc
    69        -F 2 2
    70:1       F 2 2 -1d
    70:2      -F 2uv 2vw
    71        -I 2 2
    72        -I 2 2c
    72:cab    -I 2a 2
    72:bca    -I 2b 2b
    73        -I 2b 2c
    73:ba-c   -I 2a 2b
    74        -I 2b 2
    74:ba-c   -I 2a 2a
    74:cab    -I 2c 2c
    74:-cba   -I 2 2b
    74:bca    -I 2 2a
    74:a-cb   -I 2c 2
    75         P 4
    76         P 4w
    77         P 4c
    78         P 4cw
    79         I 4
    80         I 4bw
    81         P -4
    82         I -4
    83        -P 4
    84        -P 4c
    85:1       P 4ab -1ab
    85:2      -P 4a
    86:1       P 4n -1n
    86:2      -P 4bc
    87        -I 4
    88:1       I 4bw -1bw
    88:2      -I 4ad
    89         P 4 2
    90         P 4ab 2ab
    91         P 4w 2c
    92         P 4abw 2nw
    93         P 4c 2
    94         P 4n 2n
    95         P 4cw 2c
    96         P 4nw 2abw
    97         I 4 2
    98         I 4bw 2bw
    99         P 4 -2
   100         P 4 -2ab
   101         P 4c -2c
   102         P 4n -2n
   103         P 4 -2c
   104         P 4 -2n
   105         P 4c -2
   106         P 4c -2ab
   107         I 4 -2
   108         I 4 -2c
   109         I 4bw -2
   110         I 4bw -2c
   111         P -4 2
   112         P -4 2c
   113         P -4 2ab
   114         P -4 2n
   115         P -4 -2
   116         P -4 -2c
   117         P -4 -2ab
   118         P -4 -2n
   119         I -4 -2
   120         I -4 -2c
   121         I -4 2
   122         I -4 2bw
   123        -P 4 2
   124        -P 4 2c
   125:1       P 4 2 -1ab
   125:2      -P 4a 2b
   126:1       P 4 2 -1n
   126:2      -P 4a 2bc
   127        -P 4 2ab
   128        -P 4 2n
   129:1       P 4ab 2ab -1ab
   129:2      -P 4a 2a
   130:1       P 4ab 2n -1ab
   130:2      -P 4a 2ac
   131        -P 4c 2
   132        -P 4c 2c
   133:1       P 4n 2c -1n
   133:2      -P 4ac 2b
   134:1       P 4n 2 -1n
   134:2      -P 4ac 2bc
   135        -P 4c 2ab
   136        -P 4n 2n
   137:1       P 4n 2n -1n
   137:2      -P 4ac 2a
   138:1       P 4n 2ab -1n
   138:2      -P 4ac 2ac
   139        -I 4 2
   140        -I 4 2c
   141:1       I 4bw 2bw -1bw
   141:2      -I 4bd 2
   142:1       I 4bw 2aw -1bw
   142:2      -I 4bd 2c
   143         P 3
   144         P 31
   145         P 32
   146:H       R 3
   146:R       P 3*
   147        -P 3
   148:H      -R 3
   148:R      -P 3*
   149         P 3 2
   150         P 3 2"
   151         P 31 2c (0 0 1)
   152         P 31 2"
   153         P 32 2c (0 0 -1)
   154         P 32 2"
   155:H       R 3 2"
   155:R       P 3* 2
   156         P 3 -2"
   157         P 3 -2
   158         P 3 -2"c
   159         P 3 -2c
   160:H       R 3 -2"
   160:R       P 3* -2
   161:H       R 3 -2"c
   161:R       P 3* -2n
   162        -P 3 2
   163        -P 3 2c
   164        -P 3 2"
   165        -P 3 2"c
   166:H      -R 3 2"
   166:R      -P 3* 2
   167:H      -R 3 2"c
   167:R      -P 3* 2n
   168         P 6
   169         P 61
   170         P 65
   171         P 62
   172         P 64
   173         P 6c
   174         P -6
   175        -P 6
   176        -P 6c
   177         P 6 2
   178         P 61 2 (0 0 -1)
   179         P 65 2 (0 0 1)
   180         P 62 2c (0 0 1)
   181         P 64 2c (0 0 -1)
   182         P 6c 2c
   183         P 6 -2
   184         P 6 -2c
   185         P 6c -2
   186         P 6c -2c
   187         P -6 2
   188         P -6c 2
   189         P -6 -2
   190         P -6c -2c
   191        -P 6 2
   192        -P 6 2c
   193        -P 6c 2
   194        -P 6c 2c
   195         P 2 2 3
   196         F 2 2 3
   197         I 2 2 3
   198         P 2ac 2ab 3
   199         I 2b 2c 3
   200        -P 2 2 3
   201:1       P 2 2 3 -1n
   201:2      -P 2ab 2bc 3
   202        -F 2 2 3
   203:1       F 2 2 3 -1d
   203:2      -F 2uv 2vw 3
   204        -I 2 2 3
   205        -P 2ac 2ab 3
   206        -I 2b 2c 3
   207         P 4 2 3
   208         P 4n 2 3
   209         F 4 2 3
   210         F 4d 2 3
   211         I 4 2 3
   212         P 4acd 2ab 3
   213         P 4bd 2ab 3
   214         I 4bd 2c 3
   215         P -4 2 3
   216         F -4 2 3
   217         I -4 2 3
   218         P -4n 2 3
   219         F -4c 2 3
   220         I -4bd 2c 3
   221        -P 4 2 3
   222:1       P 4 2 3 -1n
   222:2      -P 4a 2bc 3
   223        -P 4n 2 3
   224:1       P 4n 2 3 -1n
   224:2      -P 4bc 2bc 3
   225        -F 4 2 3
   226        -F 4c 2 3
   227:1       F 4d 2 3 -1d
   227:2      -F 4vw 2vw 3
   228:1       F 4d 2 3 -1cd
   228:2      -F 4cvw 2vw 3
   229        -I 4 2 3
   230        -I 4bd 2c 3
"""
# Hermann-Mauguin notation
_hmStr = r"""
     1        P 1
     2        P -1
     3:b      P 1 2 1
     3:c      P 1 1 2
     3:a      P 2 1 1
     4:b      P 1 21 1
     4:c      P 1 1 21
     4:a      P 21 1 1
     5:b1     C 1 2 1
     5:b2     A 1 2 1
     5:b3     I 1 2 1
     5:c1     A 1 1 2
     5:c2     B 1 1 2
     5:c3     I 1 1 2
     5:a1     B 2 1 1
     5:a2     C 2 1 1
     5:a3     I 2 1 1
     6:b      P 1 m 1
     6:c      P 1 1 m
     6:a      P m 1 1
     7:b1     P 1 c 1
     7:b2     P 1 n 1
     7:b3     P 1 a 1
     7:c1     P 1 1 a
     7:c2     P 1 1 n
     7:c3     P 1 1 b
     7:a1     P b 1 1
     7:a2     P n 1 1
     7:a3     P c 1 1
     8:b1     C 1 m 1
     8:b2     A 1 m 1
     8:b3     I 1 m 1
     8:c1     A 1 1 m
     8:c2     B 1 1 m
     8:c3     I 1 1 m
     8:a1     B m 1 1
     8:a2     C m 1 1
     8:a3     I m 1 1
     9:b1     C 1 c 1
     9:b2     A 1 n 1
     9:b3     I 1 a 1
     9:-b1    A 1 a 1
     9:-b2    C 1 n 1
     9:-b3    I 1 c 1
     9:c1     A 1 1 a
     9:c2     B 1 1 n
     9:c3     I 1 1 b
     9:-c1    B 1 1 b
     9:-c2    A 1 1 n
     9:-c3    I 1 1 a
     9:a1     B b 1 1
     9:a2     C n 1 1
     9:a3     I c 1 1
     9:-a1    C c 1 1
     9:-a2    B n 1 1
     9:-a3    I b 1 1
    10:b      P 1 2/m 1
    10:c      P 1 1 2/m
    10:a      P 2/m 1 1
    11:b      P 1 21/m 1
    11:c      P 1 1 21/m
    11:a      P 21/m 1 1
    12:b1     C 1 2/m 1
    12:b2     A 1 2/m 1
    12:b3     I 1 2/m 1
    12:c1     A 1 1 2/m
    12:c2     B 1 1 2/m
    12:c3     I 1 1 2/m
    12:a1     B 2/m 1 1
    12:a2     C 2/m 1 1
    12:a3     I 2/m 1 1
    13:b1     P 1 2/c 1
    13:b2     P 1 2/n 1
    13:b3     P 1 2/a 1
    13:c1     P 1 1 2/a
    13:c2     P 1 1 2/n
    13:c3     P 1 1 2/b
    13:a1     P 2/b 1 1
    13:a2     P 2/n 1 1
    13:a3     P 2/c 1 1
    14:b1     P 1 21/c 1
    14:b2     P 1 21/n 1
    14:b3     P 1 21/a 1
    14:c1     P 1 1 21/a
    14:c2     P 1 1 21/n
    14:c3     P 1 1 21/b
    14:a1     P 21/b 1 1
    14:a2     P 21/n 1 1
    14:a3     P 21/c 1 1
    15:b1     C 1 2/c 1
    15:b2     A 1 2/n 1
    15:b3     I 1 2/a 1
    15:-b1    A 1 2/a 1
    15:-b2    C 1 2/n 1
    15:-b3    I 1 2/c 1
    15:c1     A 1 1 2/a
    15:c2     B 1 1 2/n
    15:c3     I 1 1 2/b
    15:-c1    B 1 1 2/b
    15:-c2    A 1 1 2/n
    15:-c3    I 1 1 2/a
    15:a1     B 2/b 1 1
    15:a2     C 2/n 1 1
    15:a3     I 2/c 1 1
    15:-a1    C 2/c 1 1
    15:-a2    B 2/n 1 1
    15:-a3    I 2/b 1 1
    16        P 2 2 2
    17        P 2 2 21
    17:cab    P 21 2 2
    17:bca    P 2 21 2
    18        P 21 21 2
    18:cab    P 2 21 21
    18:bca    P 21 2 21
    19        P 21 21 21
    20        C 2 2 21
    20:cab    A 21 2 2
    20:bca    B 2 21 2
    21        C 2 2 2
    21:cab    A 2 2 2
    21:bca    B 2 2 2
    22        F 2 2 2
    23        I 2 2 2
    24        I 21 21 21
    25        P m m 2
    25:cab    P 2 m m
    25:bca    P m 2 m
    26        P m c 21
    26:ba-c   P c m 21
    26:cab    P 21 m a
    26:-cba   P 21 a m
    26:bca    P b 21 m
    26:a-cb   P m 21 b
    27        P c c 2
    27:cab    P 2 a a
    27:bca    P b 2 b
    28        P m a 2
    28:ba-c   P b m 2
    28:cab    P 2 m b
    28:-cba   P 2 c m
    28:bca    P c 2 m
    28:a-cb   P m 2 a
    29        P c a 21
    29:ba-c   P b c 21
    29:cab    P 21 a b
    29:-cba   P 21 c a
    29:bca    P c 21 b
    29:a-cb   P b 21 a
    30        P n c 2
    30:ba-c   P c n 2
    30:cab    P 2 n a
    30:-cba   P 2 a n
    30:bca    P b 2 n
    30:a-cb   P n 2 b
    31        P m n 21
    31:ba-c   P n m 21
    31:cab    P 21 m n
    31:-cba   P 21 n m
    31:bca    P n 21 m
    31:a-cb   P m 21 n
    32        P b a 2
    32:cab    P 2 c b
    32:bca    P c 2 a
    33        P n a 21
    33:ba-c   P b n 21
    33:cab    P 21 n b
    33:-cba   P 21 c n
    33:bca    P c 21 n
    33:a-cb   P n 21 a
    34        P n n 2
    34:cab    P 2 n n
    34:bca    P n 2 n
    35        C m m 2
    35:cab    A 2 m m
    35:bca    B m 2 m
    36        C m c 21
    36:ba-c   C c m 21
    36:cab    A 21 m a
    36:-cba   A 21 a m
    36:bca    B b 21 m
    36:a-cb   B m 21 b
    37        C c c 2
    37:cab    A 2 a a
    37:bca    B b 2 b
    38        A m m 2
    38:ba-c   B m m 2
    38:cab    B 2 m m
    38:-cba   C 2 m m
    38:bca    C m 2 m
    38:a-cb   A m 2 m
    39        A b m 2
    39:ba-c   B m a 2
    39:cab    B 2 c m
    39:-cba   C 2 m b
    39:bca    C m 2 a
    39:a-cb   A c 2 m
    40        A m a 2
    40:ba-c   B b m 2
    40:cab    B 2 m b
    40:-cba   C 2 c m
    40:bca    C c 2 m
    40:a-cb   A m 2 a
    41        A b a 2
    41:ba-c   B b a 2
    41:cab    B 2 c b
    41:-cba   C 2 c b
    41:bca    C c 2 a
    41:a-cb   A c 2 a
    42        F m m 2
    42:cab    F 2 m m
    42:bca    F m 2 m
    43        F d d 2
    43:cab    F 2 d d
    43:bca    F d 2 d
    44        I m m 2
    44:cab    I 2 m m
    44:bca    I m 2 m
    45        I b a 2
    45:cab    I 2 c b
    45:bca    I c 2 a
    46        I m a 2
    46:ba-c   I b m 2
    46:cab    I 2 m b
    46:-cba   I 2 c m
    46:bca    I c 2 m
    46:a-cb   I m 2 a
    47        P m m m
    48:1      P n n n:1
    48:2      P n n n:2
    49        P c c m
    49:cab    P m a a
    49:bca    P b m b
    50:1      P b a n:1
    50:2      P b a n:2
    50:1cab   P n c b:1
    50:2cab   P n c b:2
    50:1bca   P c n a:1
    50:2bca   P c n a:2
    51        P m m a
    51:ba-c   P m m b
    51:cab    P b m m
    51:-cba   P c m m
    51:bca    P m c m
    51:a-cb   P m a m
    52        P n n a
    52:ba-c   P n n b
    52:cab    P b n n
    52:-cba   P c n n
    52:bca    P n c n
    52:a-cb   P n a n
    53        P m n a
    53:ba-c   P n m b
    53:cab    P b m n
    53:-cba   P c n m
    53:bca    P n c m
    53:a-cb   P m a n
    54        P c c a
    54:ba-c   P c c b
    54:cab    P b a a
    54:-cba   P c a a
    54:bca    P b c b
    54:a-cb   P b a b
    55        P b a m
    55:cab    P m c b
    55:bca    P c m a
    56        P c c n
    56:cab    P n a a
    56:bca    P b n b
    57        P b c m
    57:ba-c   P c a m
    57:cab    P m c a
    57:-cba   P m a b
    57:bca    P b m a
    57:a-cb   P c m b
    58        P n n m
    58:cab    P m n n
    58:bca    P n m n
    59:1      P m m n:1
    59:2      P m m n:2
    59:1cab   P n m m:1
    59:2cab   P n m m:2
    59:1bca   P m n m:1
    59:2bca   P m n m:2
    60        P b c n
    60:ba-c   P c a n
    60:cab    P n c a
    60:-cba   P n a b
    60:bca    P b n a
    60:a-cb   P c n b
    61        P b c a
    61:ba-c   P c a b
    62        P n m a
    62:ba-c   P m n b
    62:cab    P b n m
    62:-cba   P c m n
    62:bca    P m c n
    62:a-cb   P n a m
    63        C m c m
    63:ba-c   C c m m
    63:cab    A m m a
    63:-cba   A m a m
    63:bca    B b m m
    63:a-cb   B m m b
    64        C m c a
    64:ba-c   C c m b
    64:cab    A b m a
    64:-cba   A c a m
    64:bca    B b c m
    64:a-cb   B m a b
    65        C m m m
    65:cab    A m m m
    65:bca    B m m m
    66        C c c m
    66:cab    A m a a
    66:bca    B b m b
    67        C m m a
    67:ba-c   C m m b
    67:cab    A b m m
    67:-cba   A c m m
    67:bca    B m c m
    67:a-cb   B m a m
    68:1      C c c a:1
    68:2      C c c a:2
    68:1ba-c  C c c b:1
    68:2ba-c  C c c b:2
    68:1cab   A b a a:1
    68:2cab   A b a a:2
    68:1-cba  A c a a:1
    68:2-cba  A c a a:2
    68:1bca   B b c b:1
    68:2bca   B b c b:2
    68:1a-cb  B b a b:1
    68:2a-cb  B b a b:2
    69        F m m m
    70:1      F d d d:1
    70:2      F d d d:2
    71        I m m m
    72        I b a m
    72:cab    I m c b
    72:bca    I c m a
    73        I b c a
    73:ba-c   I c a b
    74        I m m a
    74:ba-c   I m m b
    74:cab    I b m m
    74:-cba   I c m m
    74:bca    I m c m
    74:a-cb   I m a m
    75        P 4
    76        P 41
    77        P 42
    78        P 43
    79        I 4
    80        I 41
    81        P -4
    82        I -4
    83        P 4/m
    84        P 42/m
    85:1      P 4/n:1
    85:2      P 4/n:2
    86:1      P 42/n:1
    86:2      P 42/n:2
    87        I 4/m
    88:1      I 41/a:1
    88:2      I 41/a:2
    89        P 4 2 2
    90        P 42 1 2
    91        P 41 2 2
    92        P 41 21 2
    93        P 42 2 2
    94        P 42 21 2
    95        P 43 2 2
    96        P 43 21 2
    97        I 4 2 2
    98        I 41 2 2
    99        P 4 m m
   100        P 4 b m
   101        P 42 c m
   102        P 42 n m
   103        P 4 c c
   104        P 4 n c
   105        P 42 m c
   106        P 42 b c
   107        I 4 m m
   108        I 4 c m
   109        I 41 m d
   110        I 41 c d
   111        P -4 2 m
   112        P -4 2 c
   113        P -4 21 m
   114        P -4 21 c
   115        P -4 m 2
   116        P -4 c 2
   117        P -4 b 2
   118        P -4 n 2
   119        I -4 m 2
   120        I -4 c 2
   121        I -4 2 m
   122        I -4 2 d
   123        P 4/m m m
   124        P 4/m c c
   125:1      P 4/n b m:1
   125:2      P 4/n b m:2
   126:1      P 4/n n c:1
   126:2      P 4/n n c:2
   127        P 4/m b m
   128        P 4/m n c
   129:1      P 4/n m m:1
   129:2      P 4/n m m:2
   130:1      P 4/n c c:1
   130:2      P 4/n c c:2
   131        P 42/m m c
   132        P 42/m c m
   133:1      P 42/n b c:1
   133:2      P 42/n b c:2
   134:1      P 42/n n m:1
   134:2      P 42/n n m:2
   135        P 42/m b c
   136        P 42/m n m
   137:1      P 42/n m c:1
   137:2      P 42/n m c:2
   138:1      P 42/n c m:1
   138:2      P 42/n c m:2
   139        I 4/m m m
   140        I 4/m c m
   141:1      I 41/a m d:1
   141:2      I 41/a m d:2
   142:1      I 41/a c d:1
   142:2      I 41/a c d:2
   143        P 3
   144        P 31
   145        P 32
   146:H      R 3
   146:R      R 3
   147        P -3
   148:H      R -3
   148:R      R -3
   149        P 3 1 2
   150        P 3 2 1
   151        P 31 1 2
   152        P 31 2 1
   153        P 32 1 2
   154        P 32 2 1
   155:H      R 32
   155:R      R 32
   156        P 3 m 1
   157        P 3 1 m
   158        P 3 c 1
   159        P 3 1 c
   160:H      R 3 m
   160:R      R 3 m
   161:H      R 3 c
   161:R      R 3 c
   162        P -3 1 m
   163        P -3 1 c
   164        P -3 m 1
   165        P -3 c 1
   166:H      R -3 m
   166:R      R -3 m
   167:H      R -3 c
   167:R      R -3 c
   168        P 6
   169        P 61
   170        P 65
   171        P 62
   172        P 64
   173        P 63
   174        P -6
   175        P 6/m
   176        P 63/m
   177        P 6 2 2
   178        P 61 2 2
   179        P 65 2 2
   180        P 62 2 2
   181        P 64 2 2
   182        P 63 2 2
   183        P 6 m m
   184        P 6 c c
   185        P 63 c m
   186        P 63 m c
   187        P -6 m 2
   188        P -6 c 2
   189        P -6 2 m
   190        P -6 2 c
   191        P 6/m m m
   192        P 6/m c c
   193        P 63/m c m
   194        P 63/m m c
   195        P 2 3
   196        F 2 3
   197        I 2 3
   198        P 21 3
   199        I 21 3
   200        P m -3
   201:1      P n -3:1
   201:2      P n -3:2
   202        F m -3
   203:1      F d -3:1
   203:2      F d -3:2
   204        I m -3
   205        P a -3
   206        I a -3
   207        P 4 3 2
   208        P 42 3 2
   209        F 4 3 2
   210        F 41 3 2
   211        I 4 3 2
   212        P 43 3 2
   213        P 41 3 2
   214        I 41 3 2
   215        P -4 3 m
   216        F -4 3 m
   217        I -4 3 m
   218        P -4 3 n
   219        F -4 3 c
   220        I -4 3 d
   221        P m -3 m
   222:1      P n -3 n:1
   222:2      P n -3 n:2
   223        P m -3 n
   224:1      P n -3 m:1
   224:2      P n -3 m:2
   225        F m -3 m
   226        F m -3 c
   227:1      F d -3 m:1
   227:2      F d -3 m:2
   228:1      F d -3 c:1
   228:2      F d -3 c:2
   229        I m -3 m
   230        I a -3 d
"""

def _buildDict(hstr):
    """build the dictionaries from the notation string

    Returns two dictionaries:  one taking sg number to string
    and the inverse

    Takes the first Hall symbol it finds.  This is desired
    for the rhombohedral lattices so that they use the hexagonal
    convention.
    """
    d  = dict()
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
                pass
            di[hstr] = n
            pass
        pass

    return d, di


lookupHall, Hall_to_sgnum = _buildDict(_hallStr)
lookupHM,   HM_to_sgnum   = _buildDict(_hmStr)


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


sgid_to_hall, hall_to_sgid = _map_sg_info(_hallStr)
sgid_to_hm, hm_to_sgid = _map_sg_info(_hmStr)


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

_sgrange = lambda min, max: tuple(range(min, max + 1)) # inclusive range
_pgDict = {
    _sgrange(  1,   1): ('c1', laue_1),  # Triclinic
    _sgrange(  2,   2): ('ci', laue_1),  #                    laue 1
    _sgrange(  3,   5): ('c2', laue_2),  # Monoclinic
    _sgrange(  6,   9): ('cs', laue_2),
    _sgrange( 10,  15): ('c2h',laue_2),  #                    laue 2
    _sgrange( 16,  24): ('d2', laue_3),  # Orthorhombic
    _sgrange( 25,  46): ('c2v',laue_3),
    _sgrange( 47,  74): ('d2h',laue_3),  #                    laue 3
    _sgrange( 75,  80): ('c4', laue_4),  # Tetragonal
    _sgrange( 81,  82): ('s4', laue_4),
    _sgrange( 83,  88): ('c4h',laue_4),  #                    laue 4
    _sgrange( 89,  98): ('d4', laue_5),
    _sgrange( 99, 110): ('c4v',laue_5),
    _sgrange(111, 122): ('d2d',laue_5),
    _sgrange(123, 142): ('d4h',laue_5),  #                    laue 5
    _sgrange(143, 146): ('c3', laue_6),  # Trigonal
    _sgrange(147, 148): ('s6', laue_6),  #                    laue 6 [also c3i]
    _sgrange(149, 155): ('d3', laue_7),
    _sgrange(156, 161): ('c3v',laue_7),
    _sgrange(162, 167): ('d3d',laue_7),  #                    laue 7
    _sgrange(168, 173): ('c6', laue_8),  # Hexagonal
    _sgrange(174, 174): ('c3h',laue_8),
    _sgrange(175, 176): ('c6h',laue_8),  #                    laue 8
    _sgrange(177, 182): ('d6', laue_9),
    _sgrange(183, 186): ('c6v',laue_9),
    _sgrange(187, 190): ('d3h',laue_9),
    _sgrange(191, 194): ('d6h',laue_9),   #                    laue 9
    _sgrange(195, 199): ('t',  laue_10),  # Cubic
    _sgrange(200, 206): ('th', laue_10),  #                    laue 10
    _sgrange(207, 214): ('o',  laue_11),
    _sgrange(215, 220): ('td', laue_11),
    _sgrange(221, 230): ('oh', laue_11),  #                    laue 11
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
    laue_11: ltype_7
    }


# Required parameters by lattice type
# * dictionary provides list of required indices with
#   a function that takes the reduced set of lattice parameters
#   to the full set
# * consistent with lparm.latticeVectors
#
_rqpDict = {
    ltype_1: (tuple(range(6)), lambda p: p),  # all 6
    ltype_2: ((0,1,2,4), lambda p: (p[0], p[1], p[2], 90, p[3], 90)), # note beta
    ltype_3: ((0,1,2),   lambda p: (p[0], p[1], p[2], 90, 90,   90)),
    ltype_4: ((0,2),     lambda p: (p[0], p[0], p[1], 90, 90,   90)),
    ltype_5: ((0,2),     lambda p: (p[0], p[0], p[1], 90, 90,  120)),
    ltype_6: ((0,2),     lambda p: (p[0], p[0], p[1], 90, 90,  120)),
    ltype_7: ((0,),      lambda p: (p[0], p[0], p[0], 90, 90,   90)),
    }

def Allowed_HKLs(sgnum, hkllist):
    """
    this function checks if a particular g vector is allowed
    by lattice centering, screw axis or glide plane
    """
    sg_hmsymbol = symbols.pstr_spacegroup[sgnum-1].strip()
    symmorphic = False
    if(sgnum in constants.sgnum_symmorphic):
        symmorphic = True

    hkllist = np.atleast_2d(hkllist)

    centering = sg_hmsymbol[0]
    if(centering == 'P'):
        # all reflections are allowed
        mask = np.ones([hkllist.shape[0], ], dtype=np.bool)
    elif(centering == 'F'):
        # same parity
        seo = np.sum(np.mod(hkllist+100, 2), axis=1)
        mask = np.logical_not(np.logical_or(seo == 1, seo == 2))
    elif(centering == 'I'):
        # sum is even
        seo = np.mod(np.sum(hkllist, axis=1)+100, 2)
        mask = (seo == 0)
    elif(centering == 'A'):
        # k+l is even
        seo = np.mod(np.sum(hkllist[:, 1:3], axis=1)+100, 2)
        mask = seo == 0
    elif(centering == 'B'):
        # h+l is even
        seo = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2)
        mask = seo == 0
    elif(centering == 'C'):
        # h+k is even
        seo = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2)
        mask = seo == 0
    elif(centering == 'R'):
        # -h+k+l is divisible by 3
        seo = np.mod(-hkllist[:, 0]+hkllist[:, 1]+hkllist[:, 2]+90, 3)
        mask = seo == 0
    else:
        raise RuntimeError(
            'IsGAllowed: unknown lattice centering encountered.')

    hkls = hkllist[mask, :]
    if(not symmorphic):
        hkls = NonSymmorphicAbsences(hkls)
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

    if(latticeType == 'triclinic'):
        """
            no systematic absences for the triclinic crystals
        """
        pass

    elif(latticeType == 'monoclinic'):
        if(ax != '2_1'):
            raise RuntimeError(
                'omitscrewaxisabsences: monoclinic systems\
                 can only have 2_1 screw axis')
        """
            only unique b-axis will be encoded
            it is the users responsibility to input
            lattice parameters in the standard setting
            with b-axis having the 2-fold symmetry
        """
        if(iax == 1):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
            mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        else:
            raise RuntimeError(
                'omitscrewaxisabsences: only b-axis\
                 can have 2_1 screw axis')

    elif(latticeType == 'orthorhombic'):
        if(ax != '2_1'):
            raise RuntimeError(
                'omitscrewaxisabsences: orthorhombic systems\
                 can only have 2_1 screw axis')
        """
        2_1 screw on primary axis
        h00 ; h = 2n
        """
        if(iax == 0):
            mask1 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
            mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif(iax == 1):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
            mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif(iax == 2):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

    elif(latticeType == 'tetragonal'):
        if(iax == 0):
            mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
            if(ax == '4_2'):
                mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
            elif(ax in ['4_1', '4_3']):
                mask2 = np.mod(hkllist[:, 2]+100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]
        elif(iax == 1):
            mask1 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
            mask2 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
            if(ax == '2_1'):
                mask3 = np.mod(hkllist[:, 0]+100, 2) != 0
                mask4 = np.mod(hkllist[:, 1]+100, 2) != 0
            mask1 = np.logical_not(np.logical_and(mask1, mask3))
            mask2 = np.logical_not(np.logical_and(mask2, mask4))
            mask = ~np.logical_or(~mask1, ~mask2)
            hkllist = hkllist[mask, :]

    elif(latticeType == 'trigonal'):
        mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
        if(iax == 0):
            if(ax in ['3_1', '3_2']):
                mask2 = np.mod(hkllist[:, 2]+90, 3) != 0
        else:
            raise RuntimeError(
                'omitscrewaxisabsences: trigonal \
                systems can only have screw axis')
        mask = np.logical_not(np.logical_and(mask1, mask2))
        hkllist = hkllist[mask, :]

    elif(latticeType == 'hexagonal'):
        mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
        if(iax == 0):
            if(ax == '6_3'):
                mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
            elif(ax in ['3_1', '3_2', '6_2', '6_4']):
                mask2 = np.mod(hkllist[:, 2]+90, 3) != 0
            elif(ax in ['6_1', '6_5']):
                mask2 = np.mod(hkllist[:, 2]+120, 6) != 0
        else:
            raise RuntimeError(
                'omitscrewaxisabsences: hexagonal \
                systems can only have screw axis')
        mask = np.logical_not(np.logical_and(mask1, mask2))
        hkllist = hkllist[mask, :]

    elif(latticeType == 'cubic'):
        mask1 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 1] == 0)
        mask2 = np.logical_and(hkllist[:, 0] == 0, hkllist[:, 2] == 0)
        mask3 = np.logical_and(hkllist[:, 1] == 0, hkllist[:, 2] == 0)
        if(ax in ['2_1', '4_2']):
            mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
            mask5 = np.mod(hkllist[:, 1]+100, 2) != 0
            mask6 = np.mod(hkllist[:, 0]+100, 2) != 0
        elif(ax in ['4_1', '4_3']):
            mask4 = np.mod(hkllist[:, 2]+100, 4) != 0
            mask5 = np.mod(hkllist[:, 1]+100, 4) != 0
            mask6 = np.mod(hkllist[:, 0]+100, 4) != 0
        mask1 = np.logical_not(np.logical_and(mask1, mask4))
        mask2 = np.logical_not(np.logical_and(mask2, mask5))
        mask3 = np.logical_not(np.logical_and(mask3, mask6))
        mask = ~np.logical_or(~mask1, np.logical_or(~mask2, ~mask3))
        hkllist = hkllist[mask, :]
    return hkllist.astype(np.int32)

def omitglideplaneabsences(self, hkllist, plane, ip):
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

    if(latticeType == 'triclinic'):
        pass

    elif(latticeType == 'monoclinic'):
        if(ip == 1):
            mask1 = hkllist[:, 1] == 0
            if(plane == 'c'):
                mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
            elif(plane == 'a'):
                mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
            elif(plane == 'n'):
                mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

    elif(latticeType == 'orthorhombic'):
        if(ip == 0):
            mask1 = hkllist[:, 0] == 0
            if(plane == 'b'):
                mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
            elif(plane == 'c'):
                mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
            elif(plane == 'n'):
                mask2 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
            elif(plane == 'd'):
                mask2 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

        elif(ip == 1):
            mask1 = hkllist[:, 1] == 0
            if(plane == 'c'):
                mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
            elif(plane == 'a'):
                mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
            elif(plane == 'n'):
                mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
            elif(plane == 'd'):
                mask2 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

        elif(ip == 2):
            mask1 = hkllist[:, 2] == 0
            if(plane == 'a'):
                mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
            elif(plane == 'b'):
                mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
            elif(plane == 'n'):
                mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
            elif(plane == 'd'):
                mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

    elif(latticeType == 'tetragonal'):
        if(ip == 0):
            mask1 = hkllist[:, 2] == 0
            if(plane == 'a'):
                mask2 = np.mod(hkllist[:, 0]+100, 2) != 0
            elif(plane == 'b'):
                mask2 = np.mod(hkllist[:, 1]+100, 2) != 0
            elif(plane == 'n'):
                mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
            elif(plane == 'd'):
                mask2 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

        elif(ip == 1):
            mask1 = hkllist[:, 0] == 0
            mask2 = hkllist[:, 1] == 0
            if(plane in ['a', 'b']):
                mask3 = np.mod(hkllist[:, 1]+100, 2) != 0
                mask4 = np.mod(hkllist[:, 0]+100, 2) != 0
            elif(plane == 'c'):
                mask3 = np.mod(hkllist[:, 2]+100, 2) != 0
                mask4 = mask3
            elif(plane == 'n'):
                mask3 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
                mask4 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
            elif(plane == 'd'):
                mask3 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
                mask4 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
            mask1 = np.logical_not(np.logical_and(mask1, mask3))
            mask2 = np.logical_not(np.logical_and(mask2, mask4))
            mask = ~np.logical_or(~mask1, ~mask2)
            hkllist = hkllist[mask, :]

        elif(ip == 2):
            mask1 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 1])
            if(plane in ['c', 'n']):
                mask2 = np.mod(hkllist[:, 2]+100, 2) != 0
            elif(plane == 'd'):
                mask2 = np.mod(2*hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
            mask = np.logical_not(np.logical_and(mask1, mask2))
            hkllist = hkllist[mask, :]

    elif(latticeType == 'trigonal'):
        if(plane != 'c'):
            raise RuntimeError(
                'omitglideplaneabsences: only c-glide \
                allowed for trigonal systems')
        if(ip == 1):
            mask1 = hkllist[:, 0] == 0
            mask2 = hkllist[:, 1] == 0
            mask3 = hkllist[:, 0] == -hkllist[:, 1]
            if(plane == 'c'):
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
            else:
                raise RuntimeError(
                    'omitglideplaneabsences: only c-glide \
                    allowed for trigonal systems')

        elif(ip == 2):
            mask1 = hkllist[:, 1] == hkllist[:, 0]
            mask2 = hkllist[:, 0] == -2*hkllist[:, 1]
            mask3 = -2*hkllist[:, 0] == hkllist[:, 1]
            if(plane == 'c'):
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
            else:
                raise RuntimeError(
                    'omitglideplaneabsences: only c-glide \
                    allowed for trigonal systems')
        mask1 = np.logical_and(mask1, mask4)
        mask2 = np.logical_and(mask2, mask4)
        mask3 = np.logical_and(mask3, mask4)
        mask = np.logical_not(np.logical_or(
            mask1, np.logical_or(mask2, mask3)))
        hkllist = hkllist[mask, :]

    elif(latticeType == 'hexagonal'):
        if(plane != 'c'):
            raise RuntimeError(
                'omitglideplaneabsences: only c-glide \
                allowed for hexagonal systems')
        if(ip == 2):
            mask1 = hkllist[:, 0] == hkllist[:, 1]
            mask2 = hkllist[:, 0] == -2*hkllist[:, 1]
            mask3 = -2*hkllist[:, 0] == hkllist[:, 1]
            mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
            mask1 = np.logical_and(mask1, mask4)
            mask2 = np.logical_and(mask2, mask4)
            mask3 = np.logical_and(mask3, mask4)
            mask = np.logical_not(np.logical_or(
                mask1, np.logical_or(mask2, mask3)))

        elif(ip == 1):
            mask1 = hkllist[:, 1] == 0
            mask2 = hkllist[:, 0] == 0
            mask3 = hkllist[:, 1] == -hkllist[:, 0]
            mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
        mask1 = np.logical_and(mask1, mask4)
        mask2 = np.logical_and(mask2, mask4)
        mask3 = np.logical_and(mask3, mask4)
        mask = np.logical_not(np.logical_or(
            mask1, np.logical_or(mask2, mask3)))
        hkllist = hkllist[mask, :]

    elif(latticeType == 'cubic'):
        if(ip == 0):
            mask1 = hkllist[:, 0] == 0
            mask2 = hkllist[:, 1] == 0
            mask3 = hkllist[:, 2] == 0
            mask4 = np.mod(hkllist[:, 0]+100, 2) != 0
            mask5 = np.mod(hkllist[:, 1]+100, 2) != 0
            mask6 = np.mod(hkllist[:, 2]+100, 2) != 0
            if(plane == 'a'):
                mask1 = np.logical_or(np.logical_and(
                    mask1, mask5), np.logical_and(mask1, mask6))
                mask2 = np.logical_or(np.logical_and(
                    mask2, mask4), np.logical_and(mask2, mask6))
                mask3 = np.logical_and(mask3, mask4)
                mask = np.logical_not(np.logical_or(
                    mask1, np.logical_or(mask2, mask3)))
            elif(plane == 'b'):
                mask1 = np.logical_and(mask1, mask5)
                mask3 = np.logical_and(mask3, mask5)
                mask = np.logical_not(np.logical_or(mask1, mask3))
            elif(plane == 'c'):
                mask1 = np.logical_and(mask1, mask6)
                mask2 = np.logical_and(mask2, mask6)
                mask = np.logical_not(np.logical_or(mask1, mask2))
            elif(plane == 'n'):
                mask4 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 2) != 0
                mask5 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 2) != 0
                mask6 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 2) != 0
                mask1 = np.logical_not(np.logical_and(mask1, mask4))
                mask2 = np.logical_not(np.logical_and(mask2, mask5))
                mask3 = np.logical_not(np.logical_and(mask3, mask6))
                mask = ~np.logical_or(
                    ~mask1, np.logical_or(~mask2, ~mask3))
            elif(plane == 'd'):
                mask4 = np.mod(hkllist[:, 1]+hkllist[:, 2]+100, 4) != 0
                mask5 = np.mod(hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                mask6 = np.mod(hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
                mask1 = np.logical_not(np.logical_and(mask1, mask4))
                mask2 = np.logical_not(np.logical_and(mask2, mask5))
                mask3 = np.logical_not(np.logical_and(mask3, mask6))
                mask = ~np.logical_or(
                    ~mask1, np.logical_or(~mask2, ~mask3))
            else:
                raise RuntimeError(
                    'omitglideplaneabsences: unknown glide \
                    plane encountered.')
            hkllist = hkllist[mask, :]

        if(ip == 2):
            mask1 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 1])
            mask2 = np.abs(hkllist[:, 1]) == np.abs(hkllist[:, 2])
            mask3 = np.abs(hkllist[:, 0]) == np.abs(hkllist[:, 2])
            if(plane in ['a', 'b', 'c', 'n']):
                mask4 = np.mod(hkllist[:, 2]+100, 2) != 0
                mask5 = np.mod(hkllist[:, 0]+100, 2) != 0
                mask6 = np.mod(hkllist[:, 1]+100, 2) != 0
            elif(plane == 'd'):
                mask4 = np.mod(2*hkllist[:, 0]+hkllist[:, 2]+100, 4) != 0
                mask5 = np.mod(hkllist[:, 0]+2*hkllist[:, 1]+100, 4) != 0
                mask6 = np.mod(2*hkllist[:, 0]+hkllist[:, 1]+100, 4) != 0
            else:
                raise RuntimeError(
                    'omitglideplaneabsences: unknown glide \
                    plane encountered.')
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
        if(p != ''):
            hkllist = omitglideplaneabsences(sgnum, hkllist, p, ip)
    axes = constants.SYS_AB[sgnum][1]
    for iax, ax in enumerate(axes):
        if(ax != ''):
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
    pmrange = lambda n: list(range(n, -(n+1), -1)) # plus/minus range
    iroot   = lambda n: int(floor(sqrt(n)))  # integer square root

    hkls = []
    hmax = iroot(ss)
    for h in pmrange(hmax):
        ss2 = ss - h*h
        kmax = iroot(ss2)
        for k in pmrange(kmax):
            rem   = ss2 - k*k
            if rem == 0:
                hkls.append((h, k, 0))
            else:
                l = iroot(rem)
                if l*l == rem:
                    hkls += [(h, k, l), (h, k, -l)]
                    pass
                pass
            pass
        pass

    return hkls


#
# ================================================== Test Functions
#


def testHKLs():
    #
    #  Check reduced HKLs
    #
    #  1. Titanium (sg 194)
    #
    sg = SpaceGroup(194)
    print('==================== Titanium (194)')
    ssmax = 20
    myHKLs = sg.getHKLs(ssmax)
    print('Number of HKLs with sum of square %d or less:  %d'
          % (ssmax, len(myHKLs)))
    for hkl in myHKLs:
        ss = hkl[0]**2 + hkl[1]**2 + hkl[2]**2
        print((hkl, ss))
        pass

    #
    #  2. Ruby (sg 167)
    #
    sg = SpaceGroup(167)
    print('==================== Ruby (167)')
    ssmax = 10
    myHKLs = sg.getHKLs(ssmax)
    print('Number of HKLs with sum of square %d or less:  %d'
          % (ssmax, len(myHKLs)))
    for hkl in myHKLs:
        ss = hkl[0]**2 + hkl[1]**2 + hkl[2]**2
        print((hkl, ss))
        pass
    #
    #  Test Generic HKLs
    #
    for ss in range(1, 10):
        print('==================== ss = %d' % ss)
        hkls = _getHKLsBySS(ss)
        print('                     number of hkls:  ', len(hkls))
        print(hkls)
        pass

    return


if __name__ == '__main__':
    #
    import sys
    #
    if 'testHKLs' in sys.argv:
        testHKLs()
        sys.exit()
    #
    #  Test Space groups:
    #
    for n in range(1, 231):
        try:
            sg = SpaceGroup(n)
            sg.getHKLs(10)
            print(sg)
            print('\n')
        except:
            print(('failed for space group number: ', n))
            print(('Hall symbol:  ', lookupHall[n]))
            pass
        pass
    #
    pass
