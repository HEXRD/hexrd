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
from hexrd.valunits import valWUnit
from hexrd import unitcell
from hexrd.constants import ptable, ptableinverse, chargestate
from hexrd import symmetry
import copy

from os import path
from pathlib import Path
from CifFile import ReadCif
import h5py
from warnings import warn
from hexrd.mksupport import Write2H5File
from hexrd.symbols import xtal_sys_dict
from hexrd.symbols import Hall_to_sgnum, HM_to_sgnum
from hexrd.utils.compatibility import h5py_read_string

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

    DFLT_LPARMS = [_angstroms(3.61), _angstroms(3.61), _angstroms(3.61),
                   _degrees(90.0), _degrees(90.0), _degrees(90.0)]
    DFLT_SSMAX = 100

    DFLT_KEV = valWUnit('wavelength', 'energy', 80.725e0, 'keV')
    DFLT_STR = 0.0025
    DFLT_TTH = numpy.radians(0.25)
    DFLT_TTHMAX = None
    """
    ATOMINFO    Fractional Atom Position of an atom in the unit cell followed by the
    site occupany and debye waller (U) factor in A^(-2)
    B is related to U by B = 8 pi^2 U

    ATOMTYPE    atomic number of all the different species in the unitcell
    """
    DFLT_ATOMINFO = numpy.array([[0., 0., 0., 1.]])
    DFLT_U = numpy.array([4.18e-7])
    DFLT_ATOMTYPE = numpy.array([28])
    DFLT_CHARGE = numpy.array(["0"])

    '''
    the dmin parameter is used to figure out the maximum sampling for g-vectors
    this parameter is in angstroms
    '''
    DFLT_DMIN = _angstroms(0.75)

    '''
    default stiffness tensor in voight notation
    '''
    DFLT_STIFFNESS = numpy.eye(6)

    '''
    some materials have more than one space group setting. for ex
    the diamond cubic system has two settings with the origin either
    at (0,0,0) or at (1/4,1/4,1/4) etc. this handle takes care of these
    cases. but the defaiult is always 0

    default space group setting
    '''
    DFLT_SGSETTING = 0

    def __init__(self, name=None,
                 material_file=None,
                 dmin=None,
                 kev=DFLT_KEV,
                 sgsetting=DFLT_SGSETTING):
        """Constructor for Material

        name -- (str) name of crystal
        material_file -- (str) name of the material file
        which contains the crystal. this could be either cif
        or hdf5
        """
        if name is None:
            self._name = Material.DFLT_XTAL
        else:
            assert isinstance(name, str), "name must be a str"
            self._name = name

        self.description = ''

        """
        set dmin to default value if its None
        if h5 file has a value, it will be substituted
        if not none, then set it to the value
        supplied and ignore the value in the h5 file
        """
        self.read_dmin_file = True
        if dmin is None:
            self._dmin = Material.DFLT_DMIN
        else:
            self._dmin = dmin
            self.read_dmin_file = False

        self._beamEnergy = kev

        self.sgsetting = sgsetting

        if material_file:
            # Get values from configuration
            # self._readCfg(material_file)
            # >> @ date 08/20/2020 SS removing dependence on hklmax
            #self._hklMax = Material.DFLT_SSMAX
            # self._beamEnergy = Material.DFLT_KEV
            form = Path(material_file).suffix[1:]

            if(form == 'cif'):
                self._readCif(material_file)
            elif(form in ['h5', 'hdf5', 'xtal']):
                self._readHDFxtal(fhdf=material_file, xtal=name)
        else:
            # default name
            self._name = Material.DFLT_XTAL
            # Use default values
            self._lparms = Material.DFLT_LPARMS
            # self._hklMax = Material.DFLT_SSMAX
            #
            self.description = ''
            #
            self.sgnum = Material.DFLT_SGNUM
            self._sgsetting = Material.DFLT_SGSETTING
            #
            self._atominfo = Material.DFLT_ATOMINFO
            #
            self._U = Material.DFLT_U
            #
            self._atomtype = Material.DFLT_ATOMTYPE
            self._charge = Material.DFLT_CHARGE
            #

        self._newUnitcell()
        self._newPdata()
        self.update_structure_factor()

    def __str__(self):
        """String representation"""
        s = 'Material:  %s\n' % self.name
        if self.description:
            s += '   description:  %s\n' % self.description
            pass
        s += '   plane Data:  %s' % str(self.planeData)
        return s

    def _reset_lparms(self):
        """
        @author Saransh Singh, Lawrence Livermore National Lab
        @date 03/11/2021 SS 1.0 original
        @details correctly initialize lattice parameters based
        on the space group number
        """
        lparms = [x.value for x in self._lparms]
        ltype = symmetry.latticeType(self.sgnum)
        lparms = [lparms[i] for i in unitcell._rqpDict[ltype][0]]
        lparms = unitcell._rqpDict[ltype][1](lparms)
        lparms_vu = []
        for i in range(6):
            if(i < 3):
                lparms_vu.append(_angstroms(lparms[i]))
            else:
                lparms_vu.append(_degrees(lparms[i]))

        self._lparms = lparms_vu

    def _newUnitcell(self):
        """
        @author Saransh Singh, Lawrence Livermore National Lab
        @date 03/11/2021 SS 1.0 original
        @details create a new unitcell class with everything initialized
        correctly
        """
        self._reset_lparms()
        self._unitcell = unitcell.unitcell(
            self._lparms, self.sgnum, self._atomtype, self._charge,
            self._atominfo, self._U,
            self._dmin.getVal('nm'), self._beamEnergy.value,
            self._sgsetting)

        if hasattr(self, 'stiffness'):
            self._unitcell.stiffness = self.stiffness
        else:
            self._unitcell.stiffness = Material.DFLT_STIFFNESS

        if pdata := getattr(self, '_pData', None):
            laue = self.unitcell._laueGroup
            reduced_lparms = self.reduced_lattice_parameters
            if pdata.laueGroup != laue or pdata.lparms != reduced_lparms:
                pdata.set_laue_and_lparms(laue, reduced_lparms)

    def _hkls_changed(self):
        # Call this when something happens that changes the hkls...
        self._newPdata()
        self.update_structure_factor()

    def _newPdata(self):
        """Create a new plane data instance if the hkls have changed"""
        # spaceGroup module calulates forbidden reflections
        '''
        >> @date 08/20/2020 SS removing dependence of planeData
        initialization on the spaceGroup module. everything is
        initialized using the unitcell module now

        >> @DATE 02/09/2021 SS adding initialization of exclusion
        from hdf5 file using a hkl list
        '''
        hkls = self.unitcell.getHKLs(self._dmin.getVal('nm')).T
        if old_pdata := getattr(self, '_pData', None):
            if hkls_match(hkls.T, old_pdata.getHKLs(allHKLs=True)):
                # There's no need to generate a new plane data object...
                return

            # Copy over attributes from the previous PlaneData object
            self._pData = PData(hkls, old_pdata, exclusions=None)

            # Get a mapping to new hkl indices
            old_indices, new_indices = map_hkls_to_new(old_pdata, self._pData)

            # Map the new exclusions to the old ones. New ones default to True.
            exclusions = numpy.ones(hkls.shape[1], dtype=bool)
            exclusions[new_indices] = old_pdata.exclusions[old_indices]

            if numpy.all(exclusions):
                # If they are all excluded, just set the default exclusions
                self.set_default_exclusions()
            else:
                self._pData.exclusions = exclusions
        else:
            # Make the PlaneData object from scratch...
            lprm = self.reduced_lattice_parameters
            laue = self.unitcell._laueGroup
            self._pData = PData(hkls, lprm, laue,
                                self._beamEnergy, Material.DFLT_STR,
                                tThWidth=Material.DFLT_TTH,
                                tThMax=Material.DFLT_TTHMAX)

            self.set_default_exclusions()

    def set_default_exclusions(self):
        if hasattr(self, 'hkl_from_file'):
            # If we loaded hkls from the file, use those
            self.enable_hkls_from_file()
        else:
            # Otherwise, enable only the first 5 by default
            self.enable_hkls_below_index(5)

    def enable_hkls_from_file(self):
        # Enable hkls from the file
        # 'hkl_from_file' must be an attribute on `self`
        exclusions = numpy.ones_like(self._pData.exclusions, dtype=bool)
        for i, g in enumerate(self._pData.hklDataList):
            if g['hkl'].tolist() in self.hkl_from_file.tolist():
                exclusions[i] = False

        self._pData.exclusions = exclusions

    def enable_hkls_below_index(self, index=5):
        # Enable hkls with indices less than @index
        exclusions = numpy.ones_like(self._pData.exclusions, dtype=bool)
        exclusions[:index] = False
        self._pData.exclusions = exclusions

    def enable_hkls_below_tth(self, tth_threshold=90.0):
        '''
          enable reflections with two-theta less than @tth_threshold degrees
        '''
        tth_threshold = numpy.radians(tth_threshold)

        tth = numpy.array([hkldata['tTheta']
                           for hkldata in self._pData.hklDataList])
        dflt_excl = numpy.ones(tth.shape, dtype=numpy.bool)
        dflt_excl2 = numpy.ones(tth.shape, dtype=numpy.bool)

        if(hasattr(self, 'hkl_from_file')):
            """
            hkls were read from the file so the exclusions will be set
            based on what hkls are present
            """
            for i, g in enumerate(self._pData.hklDataList):
                if(g['hkl'].tolist() in self.hkl_from_file.tolist()):
                    dflt_excl[i] = False

            dflt_excl2[~numpy.isnan(tth)] = \
                ~((tth[~numpy.isnan(tth)] >= 0.0) &
                  (tth[~numpy.isnan(tth)] <= tth_threshold))

            dflt_excl = numpy.logical_or(dflt_excl, dflt_excl2)

        else:
            dflt_excl[~numpy.isnan(tth)] = \
                ~((tth[~numpy.isnan(tth)] >= 0.0) &
                  (tth[~numpy.isnan(tth)] <= tth_threshold))
            dflt_excl[0] = False

        self._pData.exclusions = dflt_excl

    def update_structure_factor(self):
        hkls = self.planeData.getHKLs(allHKLs=True)
        sf = numpy.zeros([hkls.shape[0], ])
        for i, g in enumerate(hkls):
            sf[i] = self.unitcell.CalcXRSF(g)

        self.planeData.set_structFact(sf)

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
                raise RuntimeError(
                    'OS Error: No file name supplied \
                    and default file name not found.')
        else:
            try:
                cif = ReadCif(fcif)
            except(OSError):
                raise RuntimeError('OS Error: File not found')

        # read the file
        for k in cif.keys():
            if('_cell_length_a' in cif[k]):
                m = k
                break
        cifdata = cif[m]
        # cifdata = cif[cif.keys()[0]]

        # make sure the space group is present in the cif file, either as
        # international table number, hermann-maguain or hall symbol
        sgkey = ['_space_group_IT_number',
                 '_symmetry_space_group_name_h-m',
                 '_symmetry_space_group_name_hall',
                 '_symmetry_Int_Tables_number']

        sgdata = False
        for key in sgkey:
            sgdata = sgdata or (key in cifdata)
            if(sgdata):
                skey = key
                break

        if(not(sgdata)):
            raise RuntimeError(' No space group information in CIF file! ')

        sgnum = 0
        if skey is sgkey[0]:
            sgnum = int(cifdata[sgkey[0]])
        elif (skey is sgkey[1]):
            HM = cifdata[sgkey[1]]
            HM = HM.replace(" ", "")
            sgnum = HM_to_sgnum[HM]
        elif (skey is sgkey[2]):
            hall = cifdata[sgkey[2]]
            hall = hall.replace(" ", "")
            sgnum = Hall_to_sgnum[HM]
        elif(skey is sgkey[3]):
            sgnum = int(cifdata[sgkey[3]])

        # lattice parameters
        lparms = []
        lpkey = ['_cell_length_a', '_cell_length_b',
                 '_cell_length_c', '_cell_angle_alpha',
                 '_cell_angle_beta', '_cell_angle_gamma']

        for key in lpkey:
            n = cifdata[key].find('(')
            if(n != -1):
                lparms.append(float(cifdata[key][:n]))
            else:
                lparms.append(float(cifdata[key]))

        for i in range(6):
            if(i < 3):
                lparms[i] = _angstroms(lparms[i])
            else:
                lparms[i] = _degrees(lparms[i])

        self._lparms = lparms
        self.sgnum = sgnum

        # fractional atomic site, occ and vibration amplitude
        fracsitekey = ['_atom_site_fract_x', '_atom_site_fract_y',
                       '_atom_site_fract_z', ]

        occ_U = ['_atom_site_occupancy',
                 '_atom_site_u_iso_or_equiv', '_atom_site_U_iso_or_equiv']

        sitedata = True
        for key in fracsitekey:
            sitedata = sitedata and (key in cifdata)

        if(not(sitedata)):
            raise RuntimeError(
                ' fractional site position is not present \
                or incomplete in the CIF file! ')

        atompos = []
        for key in fracsitekey:
            slist = cifdata[key]
            pos = []

            for p in slist:
                n = p.find('(')

                if(n != -1):
                    pos.append(p[:n])
                else:
                    pos.append(p)

            '''
            sometimes cif files have negative values so need to
            bring them back to fractional coordinates between 0-1
            '''
            pos = numpy.asarray(pos).astype(numpy.float64)
            pos, _ = numpy.modf(pos+100.0)
            atompos.append(pos)

        """note that the vibration amplitude, U is just the amplitude (in A)
            to convert to the typical B which occurs in the debye-waller factor,
            we will use the following formula
            B = 8 * pi ^2 * < U_av^2 >
            this will be done here so we dont have to worry about it later
        """

        pocc = (occ_U[0] in cifdata.keys())
        pU = (occ_U[1] in cifdata.keys()) or (occ_U[2] in cifdata.keys())

        if(not pocc):
            warn('occupation fraction not present. setting it to 1')
            occ = numpy.ones(atompos[0].shape)
            atompos.append(occ)
        else:
            slist = cifdata[occ_U[0]]
            occ = []
            for p in slist:
                n = p.find('(')

                if(n != -1):
                    occ.append(p[:n])
                else:
                    occ.append(p)

            atompos.append(numpy.asarray(occ).astype(numpy.float64))

        if(not pU):
            warn('Debye-Waller factors not present. \
                setting to same values for all atoms.')
            U = 1.0/numpy.pi/2./numpy.sqrt(2.) * numpy.ones(atompos[0].shape)
            self._U = U
        else:
            if(occ_U[1] in cifdata.keys()):
                k = occ_U[1]
            else:
                k = occ_U[2]

            slist = cifdata[k]
            U = []
            for p in slist:
                n = p.find('(')

                if(n != -1):
                    U.append(p[:n])
                else:
                    U.append(p)

            self._U = numpy.asarray(U).astype(numpy.float64)
        '''
        format everything in the right shape etc.
        '''
        self._atominfo = numpy.asarray(atompos).T

        '''
        get atome types here i.e. the atomic number of atoms at each site
        '''
        atype = '_atom_site_type_symbol'
        patype = (atype in cifdata)
        if(not patype):
            raise RuntimeError('atom types not defined in cif file.')

        satype = cifdata[atype]
        atomtype = []

        for s in satype:
            atomtype.append(ptable[s])

        self._atomtype = numpy.asarray(atomtype).astype(numpy.int32)
        self._charge = ['0']*self._atomtype.shape[0]
        self._sgsetting = 0

    def _readHDFxtal(self, fhdf=DFLT_NAME, xtal=DFLT_NAME):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       10/17/2019 SS 1.0 original
                        01/07/2021 SS 1.1 check for optional stiffness in material file.
                        read and initialize unitcell stiffness field if present
        >> @DETAILS:    hexrd3 will have real structure factors and will require the overhaul
                        of the crystallography. In this effort, we will have a HDF file reader.
                        the file will be the same as the EMsoft xtal file. h5py will be used for
                        i/o
        """

        fexist = path.exists(fhdf)
        if(fexist):
            fid = h5py.File(fhdf, 'r')
            xtal = "/"+xtal
            if xtal not in fid:
                raise IOError('crystal doesn''t exist in material file.')
        else:
            raise IOError('material file does not exist.')

        gid = fid.get(xtal)

        sgnum = numpy.array(gid.get('SpaceGroupNumber'),
                            dtype=numpy.int32).item()
        """
            IMPORTANT NOTE:
            note that the latice parameters is nm by default
            hexrd on the other hand uses A as the default units, so we
            need to be careful and convert it right here, so there is no
            confusion later on
        """
        lparms = list(gid.get('LatticeParameters'))

        for i in range(6):
            if(i < 3):
                lparms[i] = _angstroms(lparms[i]*10.0)
            else:
                lparms[i] = _degrees(lparms[i])

        self._lparms = lparms
        #self._lparms    = self._toSixLP(sgnum, lparms)
        # fill space group and lattice parameters
        self.sgnum = sgnum

        # the U factors are related to B by the relation B = 8pi^2 U
        self._atominfo = numpy.transpose(numpy.array(
            gid.get('AtomData'), dtype=numpy.float64))
        self._U = numpy.transpose(numpy.array(
            gid.get('U'), dtype=numpy.float64))

        # read atom types (by atomic number, Z)
        self._atomtype = numpy.array(gid.get('Atomtypes'), dtype=numpy.int32)
        if 'ChargeStates' in gid:
            self._charge = h5py_read_string(gid['ChargeStates'])
        else:
            self._charge = ['0']*self._atomtype.shape[0]
        self._atom_ntype = self._atomtype.shape[0]

        self._sgsetting = numpy.array(gid.get('SpaceGroupSetting'),
                                      dtype=numpy.int32).item()

        if('stiffness' in gid):
            # we're assuming the stiffness is in units of GPa
            self.stiffness = numpy.array(gid.get('stiffness'))
        elif('compliance' in gid):
            # we're assuming the compliance is in units of TPa^-1
            self.stiffness = numpy.linalg.inv(
                numpy.array(gid.get('compliance'))) * 1.e3
        else:
            self.stiffness = numpy.zeros([6, 6])

        if self.read_dmin_file:
            if('dmin' in gid):
                # if dmin is present in the HDF5 file, then use that
                dmin = numpy.array(gid.get('dmin'),
                                   dtype=numpy.float64).item()
                self._dmin = _angstroms(dmin*10.)

        if('hkls' in gid):
            self.hkl_from_file = numpy.array(gid.get('hkls'),
                                             dtype=numpy.int32)

        fid.close()

    def dump_material(self, filename):
        '''
        get the atominfo dictionaary aand the lattice parameters
        '''
        AtomInfo = {}

        AtomInfo['file'] = filename
        AtomInfo['xtalname'] = self.name
        AtomInfo['xtal_sys'] = xtal_sys_dict[self.unitcell.latticeType.lower()]
        AtomInfo['Z'] = self.unitcell.atom_type
        AtomInfo['charge'] = self.unitcell.chargestates
        AtomInfo['SG'] = self.unitcell.sgnum
        AtomInfo['SGsetting'] = self.unitcell.sgsetting
        AtomInfo['APOS'] = self.unitcell.atom_pos
        AtomInfo['U'] = self.unitcell.U
        AtomInfo['stiffness'] = self.unitcell.stiffness
        AtomInfo['hkls'] = self.planeData.getHKLs()
        AtomInfo['dmin'] = self.unitcell.dmin
        '''
        lattice parameters
        '''
        lat_param = {'a': self.unitcell.a,
                     'b': self.unitcell.b,
                     'c': self.unitcell.c,
                     'alpha': self.unitcell.alpha,
                     'beta': self.unitcell.beta,
                     'gamma': self.unitcell.gamma}

        Write2H5File(AtomInfo, lat_param)

    # ============================== API
    #
    #  ========== Properties
    #

    # property:  spaceGroup

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, mat_name):
        assert isinstance(mat_name, str), "must set name to a str"
        self._name = mat_name

    @property
    def spaceGroup(self):
        """(read only) Space group"""
        return self._spaceGroup

    @property
    def vol(self):
        return self.unitcell.vol*1e3

    @property
    def lparms(self):
        return numpy.array([x.getVal("nm") if x.isLength() else
                            x.getVal("degrees") for x in self._lparms])

    @property
    def latticeType(self):
        return self.unitcell.latticeType

    @property
    def vol_per_atom(self):
        return self.unitcell.vol_per_atom

    @property
    def atom_pos(self):
        return self.unitcell.atom_pos

    @property
    def atom_type(self):
        return self.unitcell.atom_type

    @property
    def aniU(self):
        return self.unitcell.aniU

    @property
    def U(self):
        return self.unitcell.U

    @U.setter
    def U(self, Uarr):
        Uarr = numpy.array(Uarr)
        self._U = Uarr
        # self.unitcell.U = Uarr
        # if self.unitcell.U.shape == Uarr.shape:
        #     if not numpy.allclose(self.unitcell.U, Uarr):
        #         self.unitcell.U = Uarr
        #         self.update_structure_factor()
        #     else:
        #         return
        # else:
        #     self.unitcell.U = Uarr
        #     self.update_structure_factor()

    # property:  sgnum
    def _get_sgnum(self):
        """Get method for sgnum"""
        return self._sgnum

    def _set_sgnum(self, v):
        """Set method for sgnum
        >> @date 08/20/2020 SS removed planedata initialization
            everytime sgnum is updated singe everything is initialized
            using unitcell now
        """
        self._sgnum = v

        # Update the unit cell if there is one
        if hasattr(self, 'unitcell'):
            self._newUnitcell()
            self._hkls_changed()

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
        if(isinstance(keV, valWUnit)):
            self._beamEnergy = keV
        else:
            self._beamEnergy = valWUnit('kev', 'energy', keV, 'keV')

        '''
        acceleration potential is set in volts therefore the factor of 1e3
        @TODO make voltage valWUnit instance so this that we dont have to
        track this manually inn the future
        '''
        self.unitcell.voltage = self.beamEnergy.value*1e3
        self.planeData.wavelength = keV
        self._hkls_changed()

    beamEnergy = property(_get_beamEnergy, _set_beamEnergy, None,
                          "Beam energy in keV")

    """
    03/11/2021 SS 1.0 original
    """
    @property
    def unitcell(self):
        return self._unitcell

    @property
    def planeData(self):
        """(read only) Return the planeData attribute (lattice parameters)"""
        return self._pData

    # property:  latticeParameters
    @property
    def latticeParameters(self):
        return self._lparms

    @latticeParameters.setter
    def latticeParameters(self, v):
        """Set method for latticeParameters"""
        if(len(v) != 6):
            v = unitcell._rqpDict[self.unitcell.latticeType][1](v)
        lp = [_angstroms(v[i]) for i in range(3)]
        for i in range(3, 6):
            lp.append(_degrees(v[i]))
        self._lparms = lp
        self._newUnitcell()
        self._hkls_changed()

    @property
    def reduced_lattice_parameters(self):
        ltype = self.unitcell.latticeType
        return [self._lparms[i] for i in unitcell._rqpDict[ltype][0]]

    def _get_name(self):
        """Set method for name"""
        return self._name

    def _set_name(self, v):
        """Set method for name"""
        self._name = v

        return

    name = property(_get_name, _set_name, None,
                    "Name of material")

    @property
    def dmin(self):
        return self._dmin

    @dmin.setter
    def dmin(self, v):
        if self._dmin.getVal('angstrom') == v.getVal('angstrom'):
            return

        self._dmin = v

        # Update the unit cell
        self.unitcell.dmin = v.getVal('nm')

        self._hkls_changed()

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, vals):
        """
        first make sure the lengths are correct
        """
        if len(vals) != len(self.atomtype):
            msg = (f"incorrect size of charge. "
                f"must be same size as number of aom types.")
            raise ValueError(msg)
        """
        now check if charge states are actually allowed
        """
        for ii,z in enumerate(self.atomtype):
            elem = ptableinverse[z]
            cs = chargestate[elem]
            if not vals[ii] in cs:
                msg = (f"element {elem} does not allow "
                       f"charge state of {vals[ii]}. "
                       f"allowed charge states are {cs}.")
                raise ValueError(msg)

        self._charge = vals
        # self._newUnitcell()
        # self.update_structure_factor()

    @property
    def natoms(self):
        return self.atominfo.shape[0]

    # property: "atominfo"

    def _get_atominfo(self):
        """Set method for name"""
        return self._atominfo

    def _set_atominfo(self, v):
        """Set method for name"""
        if v.ndim != 2:
            raise ValueError("input must be 2-d.")
        if v.shape[1] != 4:
            raise ValueError("enter x, y, z, occ as nx4 array")

        self._atominfo = v

        # if self._atominfo.shape == v.shape:
        #     if not numpy.allclose(self._atominfo, v):
        #         self._atominfo = v
        #         #self.unitcell.atom_pos = v
        #         #self.update_structure_factor()

        #     else:
        #         return
        # else:
        #     self._atominfo = v
            #self.unitcell.atom_pos = v
            #self.update_structure_factor()

    atominfo = property(
        _get_atominfo, _set_atominfo, None,
        "Information about atomic positions and electron number")

    # property: "atomtype"
    def _get_atomtype(self):
        """Set method for name"""
        return self._atomtype

    def _set_atomtype(self, v):
        """Set method for atomtype"""
        """
        check to make sure number of atoms here is same as
        the atominfo
        """
        # if isinstance(v, list):
        #     if len(v) != self.natoms:
        #         raise ValueError("incorrect number of atoms")
        # elif isinstance(v, numpy.ndarray):
        #     if v.ndim != 1:
        #         if v.shape[0] != self.natoms:
        #             raise ValueError("incorrect number of atoms")

        # v = numpy.array(v)
        # if self._atomtype.shape == v.shape:
        #     if not numpy.allclose(self._atomtype, v):
        #         self._atomtype = numpy.array(v)
        #         s#elf._newUnitcell()
        #         #self.update_structure_factor()

        #     else:
        #         return
        # else:
        #     self._atomtype = numpy.array(v)
            #self._newUnitcell()
            #self.update_structure_factor()
        self._atomtype = v

    atomtype = property(
        _get_atomtype, _set_atomtype, None,
        "Information about atomic types")

    def _set_atomdata(self, atomtype, atominfo, U):
        """
        sometimes the number of atom types and their
        positions are changed when creating a material.
        this was leading to error in updating the material
        since the atominfo anf atomtype were separately updated
        with the unitcell updated for each of those calls.
        the error resulted when there was a size mismatch.
        this routine allows for simulataneous update of the two
        so there is no discrepancy and any discrepancy detected
        here is real

        the first entry is the atomtype array and the second is
        the atominfo array and the final is the U data.
        @todo pass charge state as the fourth input
        for now all charge set to zero
        """

        # check for consistency of sizes here
        atomtype = numpy.array(atomtype)
        atominfo = numpy.array(atominfo)
        U = numpy.array(U)

        if atomtype.shape[0] != atominfo.shape[0]:
            msg = (f"inconsistent shapes: number of atoms "
                   f"types passed = {atomtype.shape[0]} \n"
                   f" number of atom positions passed = {atominfo.shape[0]}" )
            raise ValueError(msg)

        if atomtype.shape[0] != U.shape[0]:
            msg = (f"inconsistent shapes: number of atoms "
                   f"types passed = {atomtype.shape[0]} \n"
                   f" U passed for {U.shape[0]} atoms." )
            raise ValueError(msg)

        if atominfo.shape[0] != U.shape[0]:
            msg = (f"inconsistent shapes: number of atom "
                   f"positions passed = {atominfo.shape[0]} \n"
                   f"U passed for {U.shape[0]} atoms." )
            raise ValueError(msg)

        self.atomtype = atomtype
        self.atominfo = atominfo
        self.U = U
        self.charge = ['0']*atomtype.shape[0]

        self._newUnitcell()
        self.update_structure_factor()
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


def load_materials_hdf5(f, dmin=Material.DFLT_DMIN, kev=Material.DFLT_KEV,
                        sgsetting=Material.DFLT_SGSETTING):
    """Load materials from an HDF5 file

    The file uses the HDF5 file format.
    """
    with h5py.File(f, 'r') as rf:
        names = list(rf)

    return {
        name: Material(name, f, dmin=dmin, kev=kev, sgsetting=sgsetting)
        for name in names
    }


def save_materials_hdf5(f, materials):
    """Save a dict of materials into an HDF5 file"""
    for material in materials.values():
        material.dump_material(f)


def hkls_match(a, b):
    # Check if hkls match. Expects inputs to have shape (x, 3).
    def sorted_hkls(x):
        return x[numpy.lexsort((x[:, 2], x[:, 1], x[:, 0]))]
    return numpy.array_equal(sorted_hkls(a), sorted_hkls(b))


def map_hkls_to_new(old_pdata, new_pdata):
    # Creates a mapping of old hkl indices to new ones.
    # Expects inputs to be PlaneData objects.
    def get_hkl_strings(pdata):
        return pdata.getHKLs(allHKLs=True, asStr=True)

    kwargs = {
        'ar1': get_hkl_strings(old_pdata),
        'ar2': get_hkl_strings(new_pdata),
        'return_indices': True,
    }
    return numpy.intersect1d(**kwargs)[1:]

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
