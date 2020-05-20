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
"""
Module for XRD material class

Use the Material class directly for new materials.  Known
materials are defined by name in materialDict.
"""
from configparser import SafeConfigParser as Parser
import numpy

from hexrd.crystallography import PlaneData as PData
from hexrd.spacegroup import SpaceGroup as SG

from hexrd.valunits import valWUnit

__all__ = ['Material', 'loadMaterialList']


#
# ================================================== Module Data
#


def _angstroms(x):
    return valWUnit('lp', 'length',  x, 'angstrom')


def _degrees(x):
    return valWUnit('lp', 'angle',   x, 'degrees')


def _key(x):
    return x.name


#
# ---------------------------------------------------CLASS:  Material
#


class Material(object):
    """Simple class for holding lattice parameters, accessible by name.

    The class references materials by name and contains lattice and
    space group data.
    """
    DFLT_NAME = 'material'
    DFLT_SGNUM = 230
    DFLT_LPARMS = [_angstroms(1.0), _angstroms(1.0), _angstroms(1.0),
                   _degrees(90.0), _degrees(90.0), _degrees(90.0)]
    DFLT_SSMAX = 50

    DFLT_KEV = valWUnit('wavelength', 'energy', 80.725e0, 'keV')
    DFLT_STR = 0.0025
    DFLT_TTH = numpy.radians(0.25)
    DFLT_ATOMINFO = numpy.array([[0, 0, 0, 1]])
    """Fractional Atom Position of an atom in the unit cell followed by the
    number of electrons within that atom. The max number of electrons is 96.
    """

    def __init__(self, name=DFLT_NAME, cfgP=None):
        """Constructor for Material

        name -- (str) name of material
        cfgP -- (instance) configuration file parser with
             -- the material name as a section
        """
        self.name = name
        self.description = ''
        if cfgP:
            # Get values from configuration
            self._readCfg(cfgP)
            pass
        else:
            # Use default values
            self._lparms = Material.DFLT_LPARMS
            self._hklMax = Material.DFLT_SSMAX
            #
            self._beamEnergy = Material.DFLT_KEV
            #
            self.description = ''
            #
            self.sgnum = Material.DFLT_SGNUM
            #
            self._atominfo = Material.DFLT_ATOMINFO
            #
            pass
        return

    def __str__(self):
        """String representation"""
        s = 'Material:  %s\n' % self.name
        if self.description:
            s += '   description:  %s\n' % self.description
            pass
        s += '   plane Data:  %s' % str(self.planeData)
        return s

    def _readCfg(self, p):
        """Read values from config parser"""

        # Lattice parameters

        lpStrings = (
            ('a-in-angstroms', _angstroms),
            ('b-in-angstroms', _angstroms),
            ('c-in-angstroms', _angstroms),
            ('alpha-in-degrees', _degrees),
            ('beta-in-degrees',  _degrees),
            ('gamma-in-degrees', _degrees)
            )

        sgnum = p.getint(self.name, 'space-group')
        tmpSG = SG(sgnum)

        try:
            hklMax = p.getint(self.name, 'hkls-ssmax')
        except:
            hklMax = Material.DFLT_SSMAX

        try:
            beamEnergy = p.getfloat(self.name, 'beam-energy')
        except:
            beamEnergy = Material.DFLT_KEV

        lparams = []
        for ind in tmpSG.reqParams:
            param, unit = lpStrings[ind]
            lparams.append(unit(p.getfloat(self.name, param)))
            pass

        # Initialize
        self._hklMax = hklMax
        self._beamEnergy = beamEnergy
        self._lparms = self._toSixLP(sgnum, lparams)
        self.sgnum = sgnum
        self.description = p.get(self.name, 'description')

        return

    def _newPdata(self):
        """Create a new plane data instance"""
        # spaceGroup module calculates forbidden reflections
        hkls = numpy.array(self.spaceGroup.getHKLs(self.hklMax)).T
        lprm = [self._lparms[i] for i in self.spaceGroup.reqParams]
        laue = self.spaceGroup.laueGroup
        self._pData = PData(hkls, lprm, laue,
                            self._beamEnergy, Material.DFLT_STR,
                            tThWidth=Material.DFLT_TTH)
        #
        #  Set default exclusions
        #
        dflt_excl = numpy.array(
            [1 for i in range(len(self._pData.exclusions))], dtype=bool)
        dflt_excl[:5] = False
        self._pData.exclusions = dflt_excl

        return

    def _toSixLP(self, sgn, lp):
        """
        Generate all six lattice parameters, making sure units are attached.
        """
        tmpSG = SG(sgn)
        lp6 = list(tmpSG.sixLatticeParams(lp))

        # make sure angles have attached units
        for i in range(6):
            if not hasattr(lp6[i], 'getVal'):
                if i in range(3):
                    lp6[i] = _angstroms(lp6[i])
                else:
                    lp6[i] = _degrees(lp6[i])
                    pass
                pass
            pass

        return lp6

    #
    # ============================== API
    #
    #  ========== Properties
    #

    # property:  spaceGroup

    @property
    def spaceGroup(self):
        """(read only) Space group"""
        return self._spaceGroup

    # property:  sgnum

    def _get_sgnum(self):
        """Get method for sgnum"""
        return self._sgnum

    def _set_sgnum(self, v):
        """Set method for sgnum"""
        self._sgnum = v
        self._spaceGroup = SG(self._sgnum)
        self._newPdata()

        return

    sgnum = property(_get_sgnum, _set_sgnum, None,
                     "Space group number")
    # property:  beamEnergy

    def _get_beamEnergy(self):
        """Get method for beamEnergy"""
        return self._beamEnergy

    def _set_beamEnergy(self, keV):
        """
        Set method for beamEnergy

        * note that units are assumed to be keV for
          float arguments.  Also can take a valWUnit
          instance
        """
        self._beamEnergy = keV
        self.planeData.wavelength = keV

        return

    beamEnergy = property(_get_beamEnergy, _set_beamEnergy, None,
                          "Beam energy in keV")

    # property:  hklMax

    def _get_hklMax(self):
        """Get method for hklMax"""
        return self._hklMax

    def _set_hklMax(self, v):
        """Set method for hklMax"""
        self._hklMax = v
        self._newPdata()  # update planeData
        return

    hklMax = property(_get_hklMax, _set_hklMax, None,
                      "Max sum of squares for HKLs")
    # property:  planeData

    @property
    def planeData(self):
        """(read only) Return the planeData attribute (lattice parameters)"""
        return self._pData

    # property:  latticeParameters

    def _get_latticeParameters(self):
        """Get method for latticeParameters"""
        return self._lparms

    def _set_latticeParameters(self, v):
        """Set method for latticeParameters"""
        self._lparms = self._toSixLP(self.sgnum, v)
        # self._newPdata()
        self.planeData.lparms = v

        return

    lpdoc = r"""Lattice parameters

On output, all six paramters are returned.

On input, either all six or a minimal set is accepted.

The values have units attached, i.e. they are valWunit instances.
"""
    latticeParameters = property(
            _get_latticeParameters, _set_latticeParameters,
            None, lpdoc)

    # property:  "name"

    def _get_name(self):
        """Set method for name"""
        return self._name

    def _set_name(self, v):
        """Set method for name"""
        self._name = v

        return

    name = property(_get_name, _set_name, None,
                    "Name of material")

    # property: "atominfo"
    def _get_atominfo(self):
        """Set method for name"""
        return self._atominfo

    def _set_atominfo(self, v):
        """Set method for name"""
        if v.shape[1] == 4:
            self._atominfo = v
        else:
            print("Improper syntax, array must be n x 4")

        return

    atominfo = property(
        _get_atominfo, _set_atominfo, None,
        "Information about atomic positions and electron number")

    #
    #  ========== Methods
    #
    #
    pass  # end class


#
#  -----------------------------------------------END CLASS:  Material
#
#  Utility Functions
#


def loadMaterialList(cfgFile):
    """Load a list of materials from a file

    The file uses the config file format.  See ConfigParser module.
"""
    p = Parser()
    p.read(cfgFile)
    #
    #  Each section defines a material
    #
    names = p.sections()
    matList = [Material(n, p) for n in names]
    # Sort the list
    matList = sorted(matList, key=_key)

    return matList


#
#  ============================== Executable section for testing
#


if __name__ == '__main__':
    #
    #  For testing
    #
    import sys

    if len(sys.argv) == 1:
        print("need argument:  materials.cfg")
        sys.exit()
        pass

    ml = loadMaterialList(sys.argv[1])

    print('MATERIAL LIST\n')
    print(('   from file:  ', sys.argv[1]))
    for m in ml:
        print(m)
        pass
    pass
