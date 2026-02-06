#! /usr/bin/env python
# ============================================================
# Copyright (c) 2007-2012, Lawrence Livermore National Security, LLC.
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
# ============================================================
#
"""Module for associating units with scalar quantities

This module has been modified from its original form
by removing the call to the "units" executable and restricting
the units to only those used by the heXRD package.

"""

import math

import numpy as np

from hexrd.core.constants import keVToAngstrom


__all__ = ['valWUnit', 'toFloat', 'valWithDflt']

# centralized unit types; chosen to have a match in the units command
energyUN = "ENERGY"
lengthUN = "LENGTH"
angleUN = "ANGLE"
#
# mapping to centralized unit types, for convenience
#
uTDict = dict(
    length=lengthUN,
    angle=angleUN,
    energy=energyUN,
)

cv_dict = {
    ('degrees', 'radians'): math.pi / 180.0,
    ('radians', 'degrees'): 180 / math.pi,
    ('m', 'mm'): 1.0e3,
    ('m', 'meter'): 1.0,
    ('m', 'nm'): 1.0e9,
    ('m', 'angstrom'): 1.0e10,
    ('meter', 'mm'): 1.0e3,
    ('meter', 'm'): 1.0,
    ('meter', 'nm'): 1.0e9,
    ('meter', 'angstrom'): 1.0e10,
    ('mm', 'm'): 1.0e-3,
    ('mm', 'meter'): 1.0e-3,
    ('mm', 'nm'): 1.0e6,
    ('mm', 'angstrom'): 1.0e7,
    ('angstrom', 'm'): 1.0e-10,
    ('angstrom', 'meter'): 1.0e-10,
    ('angstrom', 'mm'): 1.0e-7,
    ('angstrom', 'nm'): 1.0e-1,
    ('keV', 'J'): 1.60217646e-16,
    ('J', 'keV'): (1 / 1.60217646e-16),
}


# Value with units
class valWUnit:
    def __init__(self, name: str, unitType: str, value: float, unit: str):
        """
        Parameters
        ----------
        name : str
            Name of the item.
        unitType : str
            Class of units, e.g., 'length', 'angle', 'energy'.
        value : float | np.floating
            Numerical value.
        unit : str
            Name of the unit.
        """
        self.name = name
        if unitType in uTDict:
            self.uT = uTDict[unitType]
        else:
            # trust that unitType is correct -- may be a combined type
            self.uT = unitType

        self.value = value
        self.unit = unit

    def __str__(self):
        tmpl = """item named "%s" representing %g %s"""
        return tmpl % (self.name, self.value, self.unit)

    def __repr__(self):
        tmpl = 'valWUnit("%s","%s",%s,"%s")'
        return tmpl % (self.name, self.uT, self.value, self.unit)

    def __mul__(self, other: 'valWUnit | float') -> 'valWUnit':
        if isinstance(other, float):
            new = valWUnit(self.name, self.uT, self.value * other, self.unit)
            return new
        elif isinstance(other, valWUnit):
            new = valWUnit(
                '%s_times_%s' % (self.name, other.name),
                '%s %s' % (self.uT, other.uT),
                self.value * other.value,
                '(%s)*(%s)' % (self.unit, other.unit),
            )
            # really need to put in here something to resolve new.uT
            return new
        else:
            raise RuntimeError("mul with unsupported operand")

    def __add__(self, other: 'valWUnit | float') -> 'valWUnit':
        if isinstance(other, float):
            new = valWUnit(self.name, self.uT, self.value + other, self.unit)
            return new
        elif isinstance(other, valWUnit):
            new = valWUnit(
                self.name,
                self.uT,
                self.value + other.getVal(self.unit),
                self.unit,
            )
            return new
        else:
            raise RuntimeError("add with unsupported operand")

    def __sub__(self, other: 'valWUnit | float') -> 'valWUnit':
        if isinstance(other, float):
            new = valWUnit(self.name, self.uT, self.value - other, self.unit)
            return new
        elif isinstance(other, valWUnit):
            new = valWUnit(
                self.name,
                self.uT,
                self.value - other.getVal(self.unit),
                self.unit,
            )
            return new
        else:
            raise RuntimeError("add with unsupported operand")

    def _convert(self, toUnit: str) -> float:
        """
        Return the value of self in requested units.

        Parameters
        ----------
        toUnit : str
            The identifier for desired unit type.  Current choices are.
                degrees = 'degrees'
                radians = 'radians'

                meters = 'm' | 'meter'
                millimeters = 'mm'
                nanometers = 'nm'
                Ångstroms = 'angstrom'

                kilo electron-Volt = 'keV'
                Joule = 'J'

        Raises
        ------
        RuntimeError
            if `toUnit` is invalid.

        Returns
        -------
        saclar
            The converted unit value of self.

        """
        if self.unit == toUnit:
            return self.value
        #
        #  Needs conversion
        #
        from_to = (self.unit, toUnit)
        try:
            return cv_dict[from_to] * self.value
        except KeyError:
            special_case = ('keV', 'angstrom')
            if from_to == special_case or from_to == special_case[::-1]:
                return keVToAngstrom(self.value)
            raise RuntimeError(
                f"Unit conversion '{from_to[0]} --> " + f"{from_to[1]}' not recognized"
            )

    def isLength(self) -> bool:
        """Return true if quantity is a length"""
        return self.uT == uTDict['length']

    def isAngle(self) -> bool:
        """Return true if quantity is an angle"""
        return self.uT == uTDict['angle']

    def isEnergy(self) -> bool:
        """Return true if quantity  is an energy"""
        return self.uT == uTDict['energy']

    def getVal(self, toUnit):
        """
        Returns object value in requested units.

        Parameters
        ----------
        toUnit : str
            The identifier for desired unit type.  Current choices are
            .

        Raises
        ------
        RuntimeError
            Where the requested units are invalid

        Returns
        -------
        scalar
            The value of the object in the requested units.

        """
        return self._convert(toUnit)


def _toFloatScalar(v, u):
    if hasattr(v, 'getVal'):
        return v.getVal(u)
    else:
        return v


def toFloat(val, unitName):
    """Return the raw value of the object

    INPUTS

    val
       (float|valWUnit) object with value
    unitName
       (str) name of unit

    This function returns the raw value of the object, ignoring the
    unit, if it is numeric or converts it to the requested units and
    returns the magnitude if it is a valWUnit instance.

    For example:

    >>> print(toFloat(1.1, 'radians'))
    1.1
    >>> v = valWUnit('vee', 'angle', 1.1, 'radians')
    >>> print(toFloat(v, 'degrees'))
    63.02535746439056

    """

    if hasattr(val, '__len__'):
        retval = [_toFloatScalar(x, unitName) for x in val]
    else:
        retval = _toFloatScalar(val, unitName)
    return retval


def valWithDflt(val, dflt, toUnit=None):
    """Return value or default value"""
    retval = val
    if retval is None:
        retval = dflt
    if toUnit is not None:
        retval = toFloat(retval, toUnit)
    return retval


def _nm(x: float) -> valWUnit:
    return valWUnit("lp", "length", x, "nm")


def _kev(x: float) -> valWUnit:
    return valWUnit("kev", "energy", x, "keV")


def _angstrom(x: float) -> valWUnit:
    return valWUnit("lp", "length", x, "angstrom")


def _degrees(x: float) -> valWUnit:
    return valWUnit('lp', 'angle', x, 'degrees')


# Function alias
_angstroms = _angstrom
