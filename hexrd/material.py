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
import hexrd.spacegroup as sg
from hexrd.valunits import valWUnit
from hexrd.unitcell import unitcell

import copy

from os import path
from CifFile import ReadCif 
import  h5py

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
    default data is for nickel, but name is material
    """
    DFLT_NAME = 'material.xtal'
    DFLT_XTAL = 'Ni'
    DFLT_SGNUM = 225
    '''
    some materials have more than one space group setting. for ex
    the diamond cubic system has two settings with the origin either
    at (0,0,0) or at (1/4,1/4,1/4) etc. this handle takes care of these
    cases. but the defaiult is always 1
    '''
    DFLT_SGSETTING = 1

    DFLT_LPARMS = [_angstroms(3.61), _angstroms(3.61), _angstroms(3.61),
                   _degrees(90.0), _degrees(90.0), _degrees(90.0)]
    DFLT_SSMAX = 100

    DFLT_KEV = valWUnit('wavelength', 'energy', 80.725e0, 'keV')
    DFLT_STR = 0.0025
    DFLT_TTH = numpy.radians(0.25)

    """
    ATOMINFO    Fractional Atom Position of an atom in the unit cell followed by the
    site occupany and debye waller (B) factor in nm^(-2)

    ATOMTYPE    atomic number of all the different species in the unitcell
    """
    DFLT_ATOMINFO = numpy.array([[0, 0, 0, 1, 0.033]])    
    DFLT_ATOMTYPE = numpy.array([28])

    '''
    the dmin parameter is used to figure out the maximum sampling for g-vectors
    this parameter is in angstroms
    '''
    DFLT_DMIN = _angstroms(0.5)

    '''
    default space group setting
    '''
    DFLT_SGSETTING = 0

    def __init__(self, name=DFLT_NAME, cfgP=None, dmin=DFLT_DMIN, kev=DFLT_KEV, sgsetting=DFLT_SGSETTING):
        """Constructor for Material

        name -- (str) name of material
        cfgP -- (instance) configuration file parser with
             -- the material name as a section
        """
        self.name = name
        self.description = ''
        self._dmin = dmin
        self._beamEnergy = kev
        self.sgsetting = sgsetting
        if(self._dmin.unit == 'angstrom'):
            # convert to nm
            uc_dmin = self._dmin.value * 0.1

        elif(self._dmin.unit == 'nm'):
            uc_dmin = self._dmin.value

        if cfgP:
            # Get values from configuration
            # self._readCfg(cfgP)
            self._hklMax = Material.DFLT_SSMAX
            # self._beamEnergy = Material.DFLT_KEV
            self._readHDFxtal(fhdf=cfgP, xtal=name)
            # self._readCif(cfgP)
            pass
        else:
            # Use default values
            self._lparms = Material.DFLT_LPARMS
            self._hklMax = Material.DFLT_SSMAX
            #
            self.description = ''
            #
            self.sgnum = Material.DFLT_SGNUM
            self._sgsetting = Material.DFLT_SGSETTING
            #
            self._atominfo = Material.DFLT_ATOMINFO
            #
            self._atomtype = Material.DFLT_ATOMTYPE
            #

        self.unitcell = unitcell(self._lparms, self.sgnum, self._atomtype,
                                self._atominfo, self._dmin.value, 
                                self._beamEnergy.value, self.sgsetting)

        hkls = self.planeData.getHKLs(allHKLs=True)
        sf = numpy.zeros([hkls.shape[0],])
        for i,g in enumerate(hkls):
            sf[i] = self.unitcell.CalcXRSF(g)

        self.planeData.set_structFact(sf[~self.planeData.exclusions])

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
        '''
          Set default exclusions
          all reflections with two-theta smaller than 90 degrees
        '''
        tth = numpy.array([hkldata['tTheta'] for hkldata in self._pData.hklDataList])

        dflt_excl = numpy.ones(tth.shape,dtype=numpy.bool)
        dflt_excl[~numpy.isnan(tth)] = ~( (tth[~numpy.isnan(tth)] >= 0.0) & \
                                     (tth[~numpy.isnan(tth)] <= numpy.pi/2.0) )
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


    def _readCif(self, fcif=DFLT_NAME+'.cif'):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       10/16/2019 SS 1.0 original
        >> @DETAILS:    hexrd3 will have real structure factors and will require the overhaul
                        of the crystallography. In this effort, we will have a cif reader and
                        also the HDF5 format reader in the material class. We will be using 
                        pycifrw for i/o
        """

        # make sure file exists etc.
        if(fcif == Material.DFLT_NAME+'.cif'):
            try:
                cif = ReadCif(fcif)
            except(OSError):
                raise RuntimeError('OS Error: No file name supplied and default file name not found.')
        else:
            try:
                cif = ReadCif(fcif)
            except(OSError):
                raise RuntimeError('OS Error: File not found')

        # read the file
        cifdata = cif[cif.keys()[0]]

        # make sure the space group is present in the cif file, either as 
        # international table number, hermann-maguain or hall symbol
        sgkey = ['_space_group_IT_number', '_symmetry_space_group_name_h-m', \
                 '_symmetry_space_group_name_hall']

        sgdata = False
        for key in sgkey:
            sgdata = sgdata or (key in cifdata)

        if(not(sgdata)):
            raise RuntimeError(' No space group information in CIF file! ')

        sgnum = 0
        if sgkey[0] in cifdata:
            sgnum = int(cifdata[sgkey[0]])

        if (sgnum == 0):
            if (sgkey[1] in cifdata):
                HM = cifdata[sgkey[1]]
                sgnum = sg.HM_to_sgnum[HM]
            if (sgkey[2] in cifdata):
                hall = cifdata[sgkey[2]]
                sgnum = sg.Hall_to_sgnum[HM]  

        # lattice parameters
        lparms = []
        lpkey = ['_cell_length_a', '_cell_length_b', \
                 '_cell_length_c', '_cell_angle_alpha', \
                 '_cell_angle_beta', '_cell_angle_gamma']   
        for key in lpkey:
            lparms.append(float(cifdata[key]))

        for i in range(6):
                if(i < 3):
                    lparms[i] = _angstroms(lparms[i]*10.0)
                else:
                    lparms[i] = _degrees(lparms[i])

        self._lparms = lparms
        self.sgnum   = sgnum

        # fractional atomic site, occ and vibration amplitude
        fracsitekey = ['_atom_site_fract_x', '_atom_site_fract_y',\
                        '_atom_site_fract_z',]
        occ_U       = ['_atom_site_occupancy',\
                        '_atom_site_U_iso_or_equiv']
        sitedata = True
        for key in fracsitekey:
            sitedata = sitedata and (key in cifdata)

        if(not(sitedata)):
            raise RuntimeError(' fractional site position is not present or incomplete in the CIF file! ')

        atompos = []
        for key in fracsitekey:
            atompos.append(numpy.asarray(cifdata[key]).astype(numpy.float64))
        
        """note that the vibration amplitude, U is just the amplitude (in A)
            to convert to the typical B which occurs in the debye-waller factor,
            we will use the following formula
            B = 8 * pi ^2 * < U_av^2 >
            this will be done here so we dont have to worry about it later 
        """
        self._atominfo = numpy.asarray(atompos).T
        self._atominfo[:,4] = 8.0 * (numpy.pi**2) * (self._atominfo[:,4]**2)

    def _readHDFxtal(self, fhdf=DFLT_NAME, xtal=DFLT_NAME):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       10/17/2019 SS 1.0 original
        >> @DETAILS:    hexrd3 will have real structure factors and will require the overhaul
                        of the crystallography. In this effort, we will have a HDF file reader.
                        the file will be the same as the EMsoft xtal file. h5py will be used for
                        i/o
        """

        try:
            fid = h5py.File(fhdf, 'r')
        except IOError:
            IOError('material file does not exist.')

        try:
            fid["/"+xtal]
        except:
            KeyError('crystal doesn''t exist in material file.')

        for groups in list(fid.keys()):
            
            gid         = fid.get(groups)
            
            sgnum       = numpy.asscalar(numpy.array(gid.get('SpaceGroupNumber'), \
                                        dtype = numpy.int32))
            """ 
                IMPORTANT NOTE:
                note that the latice parameters in EMsoft is nm by default
                hexrd on the other hand uses A as the default units, so we
                need to be careful and convert it right here, so there is no
                confusion later on
            """
            lparms      = list(gid.get('LatticeParameters'))

            for i in range(6):
                if(i < 3):
                    lparms[i] = _angstroms(lparms[i]*10.0)
                else:
                    lparms[i] = _degrees(lparms[i])

        self._lparms    = lparms
        #self._lparms    = self._toSixLP(sgnum, lparms)
        # fill space group and lattice parameters
        self.sgnum      = sgnum

        # the last field in this is already the B factor, so no need to convert
        self._atominfo  = numpy.transpose(numpy.array(gid.get('AtomData'), dtype = numpy.float64))

        # read atom types (by atomic number, Z)
        self._atomtype = numpy.array(gid.get('Atomtypes'), dtype = numpy.int32)
        self._atom_ntype = self._atomtype.shape[0]
        self._sgsetting = gid.get('SpaceGroupSetting')

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
    matList = sorted(matList, _key)

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
