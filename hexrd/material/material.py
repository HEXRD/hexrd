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
import numpy as np

from hexrd.material.crystallography import PlaneData as PData
from hexrd.material import symmetry, unitcell
from hexrd.valunits import valWUnit
from hexrd.constants import ptable, ptableinverse, chargestate

from os import path
from pathlib import Path
from CifFile import ReadCif
import h5py
from warnings import warn
from hexrd.material.mksupport import Write2H5File
from hexrd.material.symbols import (
    xtal_sys_dict,
    Hall_to_sgnum,
    HM_to_sgnum,
)
from hexrd.utils.compatibility import h5py_read_string
from hexrd.fitting.peakfunctions import _unit_gaussian

__all__ = ['Material', 'loadMaterialList']


#
# ================================================== Module Data
#


def _angstroms(x):
    return valWUnit('lp', 'length', x, 'angstrom')


def _degrees(x):
    return valWUnit('lp', 'angle', x, 'degrees')


def _kev(x):
    return valWUnit('xrayenergy', 'energy', x, 'keV')


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

    DFLT_LPARMS = [
        _angstroms(3.61),
        _angstroms(3.61),
        _angstroms(3.61),
        _degrees(90.0),
        _degrees(90.0),
        _degrees(90.0),
    ]
    DFLT_SSMAX = 100

    DFLT_KEV = valWUnit('wavelength', 'energy', 80.725e0, 'keV')
    DFLT_STR = 0.0025
    DFLT_TTH = np.radians(0.25)
    DFLT_TTHMAX = None
    """
    ATOMINFO    Fractional Atom Position of an atom in the
    unit cell followed by the site occupany and debye waller
    (U) factor in A^(-2)
    B is related to U by B = 8 pi^2 U

    ATOMTYPE    atomic number of all the different species in the unitcell
    """
    DFLT_ATOMINFO = np.array([[0.0, 0.0, 0.0, 1.0]])
    DFLT_U = np.array([6.33e-3])
    DFLT_ATOMTYPE = np.array([28])
    DFLT_CHARGE = np.array(["0"])

    '''
    the dmin parameter is used to figure out the maximum sampling for g-vectors
    this parameter is in angstroms
    '''
    DFLT_DMIN = _angstroms(0.75)

    '''
    default stiffness tensor in voight notation
    '''
    DFLT_STIFFNESS = np.eye(6)

    '''
    some materials have more than one space group setting. for ex
    the diamond cubic system has two settings with the origin either
    at (0,0,0) or at (1/4,1/4,1/4) etc. this handle takes care of these
    cases. but the defaiult is always 0

    default space group setting
    '''
    DFLT_SGSETTING = 0

    def __init__(
        self,
        name=None,
        material_file=None,
        dmin=None,
        kev=None,
        sgsetting=DFLT_SGSETTING,
    ):
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

        self.sgsetting = sgsetting

        if material_file:
            # >> @ date 08/20/2020 SS removing dependence on hklmax
            if isinstance(material_file, (Path, str)):
                form = Path(material_file).suffix[1:]
            else:
                form = None

            h5_suffixes = ('h5', 'hdf5', 'xtal')

            if form == 'cif':
                self._readCif(material_file)
            elif isinstance(material_file, h5py.Group) or form in h5_suffixes:
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
            self._tThWidth = Material.DFLT_TTH
            #
            self._dmin = Material.DFLT_DMIN
            self._beamEnergy = Material.DFLT_KEV

            self.pressure = 0
            self.temperature = 298
            self.k0 = 100.0
            self.k0p = 0.0
            self.dk0dt = 0.0
            self.dk0pdt = 0.0
            self.alpha_t = 0.0
            self.dalpha_t_dt = 0.0

        # If these were specified, they override any other method of
        # obtaining them (including loading them from files).
        if dmin is not None:
            self._dmin = dmin

        if kev is not None:
            self._beamEnergy = kev

        self._newUnitcell()

        if not hasattr(self, 'v0'):
            self.reset_v0()

        self._newPdata()
        self.invalidate_structure_factor()

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
            if i < 3:
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
            self._lparms,
            self.sgnum,
            self._atomtype,
            self._charge,
            self._atominfo,
            self._U,
            self._dmin.getVal('nm'),
            self._beamEnergy.value,
            self._sgsetting,
        )

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
        self.invalidate_structure_factor()

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
            exclusions = np.ones(hkls.shape[1], dtype=bool)
            exclusions[new_indices] = old_pdata.exclusions[old_indices]

            if np.all(exclusions):
                # If they are all excluded, just set the default exclusions
                self.set_default_exclusions()
            else:
                self._pData.exclusions = exclusions
        else:
            # Make the PlaneData object from scratch...
            lprm = self.reduced_lattice_parameters
            laue = self.unitcell._laueGroup
            self._pData = PData(
                hkls,
                lprm,
                laue,
                self._beamEnergy,
                Material.DFLT_STR,
                tThWidth=self._tThWidth,
                tThMax=Material.DFLT_TTHMAX,
            )

            self.set_default_exclusions()

    def reset_v0(self):
        self.v0 = self.vol

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
        exclusions = np.ones_like(self._pData.exclusions, dtype=bool)
        for i, g in enumerate(self._pData.hklDataList):
            if g['hkl'].tolist() in self.hkl_from_file.tolist():
                exclusions[i] = False

        self._pData.exclusions = exclusions

    def enable_hkls_below_index(self, index=5):
        # Enable hkls with indices less than @index
        exclusions = np.ones_like(self._pData.exclusions, dtype=bool)
        exclusions[:index] = False
        self._pData.exclusions = exclusions

    def enable_hkls_below_tth(self, tth_threshold=90.0):
        '''
        enable reflections with two-theta less than @tth_threshold degrees
        '''
        tth_threshold = np.radians(tth_threshold)

        tth = np.array(
            [hkldata['tTheta'] for hkldata in self._pData.hklDataList]
        )
        dflt_excl = np.ones(tth.shape, dtype=np.bool)
        dflt_excl2 = np.ones(tth.shape, dtype=np.bool)

        if hasattr(self, 'hkl_from_file'):
            """
            hkls were read from the file so the exclusions will be set
            based on what hkls are present
            """
            for i, g in enumerate(self._pData.hklDataList):
                if g['hkl'].tolist() in self.hkl_from_file.tolist():
                    dflt_excl[i] = False

            dflt_excl2[~np.isnan(tth)] = ~(
                (tth[~np.isnan(tth)] >= 0.0)
                & (tth[~np.isnan(tth)] <= tth_threshold)
            )

            dflt_excl = np.logical_or(dflt_excl, dflt_excl2)

        else:
            dflt_excl[~np.isnan(tth)] = ~(
                (tth[~np.isnan(tth)] >= 0.0)
                & (tth[~np.isnan(tth)] <= tth_threshold)
            )
            dflt_excl[0] = False

        self._pData.exclusions = dflt_excl

    def invalidate_structure_factor(self):
        self.planeData.invalidate_structure_factor(self.unitcell)

    def compute_powder_overlay(
        self, ttharray=np.linspace(0, 80, 2000), fwhm=0.25, scale=1.0
    ):
        """
        this function computes a simulated spectra
        for using in place of lines for the powder
        overlay. inputs are simplified as compared
        to the typical LeBail/Rietveld computation.
        only a fwhm (in degrees) and scale are passed

        requested feature from Amy Jenei
        """
        tth = np.degrees(self.planeData.getTTh())  # convert to degrees
        Ip = self.planeData.powder_intensity
        self.powder_overlay = np.zeros_like(ttharray)
        for t, I in zip(tth, Ip):
            p = [t, fwhm]
            self.powder_overlay += scale * I * _unit_gaussian(p, ttharray)

    def remove_duplicate_atoms(self):
        """
        this function calls the same function in the
        unitcell class and updates planedata structure
        factors etc.
        """
        self.unitcell.remove_duplicate_atoms()
        self.atominfo = self.unitcell.atom_pos
        self.atomtype = self.unitcell.atom_type
        self.charge = self.unitcell.chargestates
        self._hkls_changed()

    def vt(self, temperature=None):
        '''calculate volume at high
        temperature
        '''
        alpha0 = self.thermal_expansion
        alpha1 = self.thermal_expansion_dt
        if temperature is None:
            vt = self.v0
        else:
            delT = temperature - 298
            delT2 = temperature**2 - 298**2
            vt = self.v0 * np.exp(alpha0 * delT + 0.5 * alpha1 * delT2)
        return vt

    def kt(self, temperature=None):
        '''calculate bulk modulus for
        high temperature
        '''
        k0 = self.k0
        if temperature is None:
            return k0
        else:
            delT = temperature - 298
            return k0 + self.dk0dt * delT

    def ktp(self, temperature=None):
        '''calculate bulk modulus derivative
        for high temperature
        '''
        k0p = self.k0p
        if temperature is None:
            return k0p
        else:
            delT = temperature - 298
            return k0p + self.dk0pdt * delT

    @property
    def pt_lp_factor(self):
        return (self.unitcell.vol * 1e3 / self.v0) ** (1 / 3)

    @property
    def lparms0(self):
        # Get the lattice parameters for 0 pressure and temperature (at v0)
        lparms = self.lparms
        return np.array([
            *(lparms[:3] / self.pt_lp_factor),
            *lparms[3:],
        ])

    def calc_pressure(self, volume=None, temperature=None):
        '''calculate the pressure given the volume
        and temperature using the third order
        birch-murnaghan equation of state.
        '''
        if volume is None:
            return 0
        else:
            vt = self.vt(temperature=temperature)
            kt = self.kt(temperature=temperature)
            ktp = self.ktp(temperature=temperature)
            f = 0.5 * ((vt / volume) ** (2.0 / 3.0) - 1)

            return (
                3.0 * kt * f * (1 - 1.5 * (4 - ktp) * f) * (1 + 2 * f) ** 2.5
            )

    def calc_volume(self, pressure=None, temperature=None):
        '''solve for volume in the birch-murnaghan EoS to
        compute the volume. this number will be propagated
        to the Material object as updated lattice constants.
        '''
        vt = self.vt(temperature=temperature)
        kt = self.kt(temperature=temperature)
        ktp = self.ktp(temperature=temperature)

        if pressure is None:
            return vt
        else:
            alpha = 0.75 * (ktp - 4)
            p = np.zeros(
                [
                    10,
                ]
            )
            p[0] = 1.0
            p[2] = (1 - 2 * alpha) / alpha
            p[4] = (alpha - 1) / alpha
            p[9] = -2 * pressure / 3 / kt / alpha
            res = np.roots(p)
            res = res[np.isreal(res)]
            res = 1 / np.real(res) ** 3

            mask = np.logical_and(res >= 0.0, res <= 1.0)
            res = res[mask]
            if len(res) != 1:
                msg = 'more than one physically ' 'reasonable solution found!'
                raise ValueError(msg)
            return res[0] * vt

    def calc_lp_factor(self, pressure=None, temperature=None):
        '''calculate the factor to multiply the lattice
        constants by. only the lengths will be modified, the
        angles will be kept constant.
        '''
        vt = self.vt(temperature=temperature)
        vpt = self.calc_volume(pressure=pressure, temperature=temperature)
        return (vpt / vt) ** (1.0 / 3.0)

    def calc_lp_at_PT(self, pressure=None, temperature=None):
        '''calculate the lattice parameters for a given
        pressure and temperature using the BM EoS. This
        is the main function which will be called from
        the GUI.
        '''
        f = self.calc_lp_factor(pressure=pressure, temperature=temperature)
        lparms0 = self.lparms0
        return np.array(
            [
                *(f * lparms0[:3]),
                *lparms0[3:],
            ]
        )

    def _readCif(self, fcif=DFLT_NAME + '.cif'):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab,
                        saransh1@llnl.gov
        >> @DATE:       10/16/2019 SS 1.0 original
        >> @DETAILS:    hexrd3 will have real structure factors and will
                        require the overhaul of the crystallography.
                        In this effort, we will have a cif reader and
                        also the HDF5 format reader in the material
                        class. We will be using pycifrw for i/o
        """

        try:
            cif = ReadCif(fcif)
        except RuntimeError:
            print("file not found")

        # read the file
        for k in cif.keys():
            if '_cell_length_a' in cif[k]:
                m = k
                break
        cifdata = cif[m]
        # cifdata = cif[cif.keys()[0]]

        # make sure the space group is present in the cif file, either as
        # international table number, hermann-maguain or hall symbol
        sgkey = [
            '_space_group_IT_number',
            '_symmetry_space_group_name_h-m',
            '_symmetry_space_group_name_hall',
            '_symmetry_Int_Tables_number',
        ]

        sgdata = False
        for key in sgkey:
            sgdata = sgdata or (key in cifdata)
            if sgdata:
                skey = key
                break

        if not (sgdata):
            raise RuntimeError(' No space group information in CIF file! ')

        sgnum = 0
        if skey is sgkey[0]:
            sgnum = int(cifdata[sgkey[0]])
        elif skey is sgkey[1]:
            HM = cifdata[sgkey[1]]
            HM = HM.replace(" ", "")
            HM = HM.replace("_", "")
            sgnum = HM_to_sgnum[HM]
        elif skey is sgkey[2]:
            hall = cifdata[sgkey[2]]
            hall = hall.replace(" ", "")
            sgnum = Hall_to_sgnum[HM]
        elif skey is sgkey[3]:
            sgnum = int(cifdata[sgkey[3]])

        # lattice parameters
        lparms = []
        lpkey = [
            '_cell_length_a',
            '_cell_length_b',
            '_cell_length_c',
            '_cell_angle_alpha',
            '_cell_angle_beta',
            '_cell_angle_gamma',
        ]

        for key in lpkey:
            n = cifdata[key].find('(')
            if n != -1:
                lparms.append(float(cifdata[key][:n]))
            else:
                lparms.append(float(cifdata[key]))

        for i in range(6):
            if i < 3:
                lparms[i] = _angstroms(lparms[i])
            else:
                lparms[i] = _degrees(lparms[i])

        self._lparms = lparms
        self.sgnum = sgnum

        # fractional atomic site, occ and vibration amplitude
        fracsitekey = [
            '_atom_site_fract_x',
            '_atom_site_fract_y',
            '_atom_site_fract_z',
        ]

        occ_U = [
            '_atom_site_occupancy',
            '_atom_site_u_iso_or_equiv',
            '_atom_site_U_iso_or_equiv',
        ]

        sitedata = True
        for key in fracsitekey:
            sitedata = sitedata and (key in cifdata)

        if not (sitedata):
            raise RuntimeError(
                ' fractional site position is not present \
                or incomplete in the CIF file! '
            )

        atompos = []
        for key in fracsitekey:
            slist = cifdata[key]
            pos = []

            for p in slist:
                n = p.find('(')

                if n != -1:
                    pos.append(p[:n])
                else:
                    pos.append(p)

            '''
            sometimes cif files have negative values so need to
            bring them back to fractional coordinates between 0-1
            '''
            pos = np.asarray(pos).astype(np.float64)
            pos, _ = np.modf(pos + 100.0)
            atompos.append(pos)

        """note that the vibration amplitude, U is just the amplitude (in A)
            to convert to the typical B which occurs in the debye-waller
            factor, we will use the following formula
            B = 8 * pi ^2 * < U_av^2 >
            this will be done here so we dont have to worry about it later
        """

        pocc = occ_U[0] in cifdata.keys()
        pU = (occ_U[1] in cifdata.keys()) or (occ_U[2] in cifdata.keys())

        if not pocc:
            warn('occupation fraction not present. setting it to 1')
            occ = np.ones(atompos[0].shape)
            atompos.append(occ)
        else:
            slist = cifdata[occ_U[0]]
            occ = []
            for p in slist:
                n = p.find('(')

                if n != -1:
                    occ.append(p[:n])
                else:
                    occ.append(p)

            chkstr = np.asarray([isinstance(x, str) for x in occ])
            occstr = np.array(occ)
            occstr[chkstr] = 1.0

            atompos.append(np.asarray(occstr).astype(np.float64))

        if not pU:
            msg = (
                "'Debye-Waller factors not present. "
                "setting to same values for all atoms.'"
            )
            warn(msg)
            U = self.DFLT_U[0] * np.ones(atompos[0].shape)
            self._U = U
        else:
            if occ_U[1] in cifdata.keys():
                k = occ_U[1]
            else:
                k = occ_U[2]

            slist = cifdata[k]
            U = []
            for p in slist:
                n = p.find('(')

                if n != -1:
                    U.append(p[:n])
                else:
                    U.append(p)

            chkstr = np.asarray([isinstance(x, str) for x in U])

            for ii, x in enumerate(chkstr):
                if x:
                    try:
                        U[ii] = float(U[ii])
                    except ValueError:
                        U[ii] = self.DFLT_U[0]

            self._U = np.asarray(U)
        '''
        format everything in the right shape etc.
        '''
        self._atominfo = np.asarray(atompos).T

        '''
        get atome types here i.e. the atomic number of atoms at each site
        '''
        atype = '_atom_site_type_symbol'
        patype = atype in cifdata
        if not patype:
            raise RuntimeError('atom types not defined in cif file.')

        satype = cifdata[atype]
        atomtype = []
        charge = []
        for s in satype:
            if "+" in s:
                ss = s[:-2]
                c = s[-2:]
                if c[0] == '+':
                    c = c[::-1]
            elif "-" in s:
                ss = s[:-2]
                c = s[-2:]
                if c[0] == '-':
                    c = c[::-1]
            else:
                ss = s
                c = "0"

            atomtype.append(ptable[ss])
            charge.append(c)

        self._atomtype = np.asarray(atomtype).astype(np.int32)
        self._charge = charge
        self._sgsetting = 0

        self._dmin = Material.DFLT_DMIN
        self._beamEnergy = Material.DFLT_KEV
        self._tThWidth = Material.DFLT_TTH

        '''set the Birch-Murnaghan equation of state
        parameters to default values. These values can
        be updated by user or by reading a JCPDS file
        '''
        self.pressure = 0
        self.temperature = 298
        self.k0 = 100.0
        self.k0p = 0.0
        self.dk0dt = 0.0
        self.dk0pdt = 0.0
        self.alpha_t = 0.0
        self.dalpha_t_dt = 0.0

    def _readHDFxtal(self, fhdf=DFLT_NAME, xtal=DFLT_NAME):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab,
                        saransh1@llnl.gov
        >> @DATE:       10/17/2019 SS 1.0 original
                        01/07/2021 SS 1.1 check for optional
                        stiffness in material file.
                        read and initialize unitcell stiffness field if present
        >> @DETAILS:    hexrd3 will have real structure factors and will
                        require the overhaul of the crystallography.
                        In this effort, we will have a HDF file reader
                        the file will be the same as the EMsoft xtal
                        file. h5py will be used for i/o
        """

        if isinstance(fhdf, (Path, str)):
            if not path.exists(fhdf):
                raise IOError('material file does not exist.')

            root_gid = h5py.File(fhdf, 'r')
            xtal = f'/{xtal}'
        elif isinstance(fhdf, h5py.Group):
            root_gid = fhdf
        else:
            raise Exception(f'Unknown type for fhdf: {fhdf}')

        if xtal not in root_gid:
            raise IOError('crystal doesn' 't exist in material file.')

        gid = root_gid.get(xtal)

        sgnum = np.array(gid.get('SpaceGroupNumber'), dtype=np.int32).item()
        """
            IMPORTANT NOTE:
            note that the latice parameters is nm by default
            hexrd on the other hand uses A as the default units, so we
            need to be careful and convert it right here, so there is no
            confusion later on
        """
        lparms = list(gid.get('LatticeParameters'))

        for i in range(6):
            if i < 3:
                lparms[i] = _angstroms(lparms[i] * 10.0)
            else:
                lparms[i] = _degrees(lparms[i])

        self._lparms = lparms

        # fill space group and lattice parameters
        self.sgnum = sgnum

        # the U factors are related to B by the relation B = 8pi^2 U
        self._atominfo = np.transpose(
            np.array(gid.get('AtomData'), dtype=np.float64)
        )
        self._U = np.transpose(np.array(gid.get('U'), dtype=np.float64))

        # read atom types (by atomic number, Z)
        self._atomtype = np.array(gid.get('Atomtypes'), dtype=np.int32)
        if 'ChargeStates' in gid:
            self._charge = h5py_read_string(gid['ChargeStates'])
        else:
            self._charge = ['0'] * self._atomtype.shape[0]
        self._atom_ntype = self._atomtype.shape[0]

        self._sgsetting = np.array(
            gid.get('SpaceGroupSetting'), dtype=np.int32
        ).item()

        if 'stiffness' in gid:
            # we're assuming the stiffness is in units of GPa
            self.stiffness = np.array(gid.get('stiffness'))
        elif 'compliance' in gid:
            # we're assuming the compliance is in units of TPa^-1
            self.stiffness = (
                np.linalg.inv(np.array(gid.get('compliance'))) * 1.0e3
            )
        else:
            self.stiffness = np.zeros([6, 6])

        '''start reading the Birch-Murnaghan equation of state
        parameters
        '''
        self.pressure = 0
        if 'pressure' in gid:
            self.pressure = np.array(gid.get('pressure'),
                                     dtype=np.float64).item()

        self.temperature = 298
        if 'temperature' in gid:
            self.temperature = np.array(gid.get('temperature'),
                                        dtype=np.float64).item()

        self.k0 = 100.0
        if 'k0' in gid:
            # this is the isotropic bulk modulus
            k0 = np.array(gid.get('k0'), dtype=np.float64).item()
            self.k0 = k0

        self.k0p = 0.0
        if 'k0p' in gid:
            # this is the pressure derivation of
            # the isotropic bulk modulus
            k0p = np.array(gid.get('k0p'), dtype=np.float64).item()
            self.k0p = k0p

        self.dk0dt = 0.0
        if 'dk0dt' in gid:
            # this is the temperature derivation of
            # the isotropic bulk modulus
            dk0dt = np.array(gid.get('dk0dt'), dtype=np.float64).item()
            self.dk0dt = dk0dt

        self.dk0pdt = 0.0
        if 'dk0pdt' in gid:
            # this is the temperature derivation of
            # the pressure derivative of isotropic bulk modulus
            dk0pdt = np.array(gid.get('dk0pdt'), dtype=np.float64).item()
            self.dk0pdt = dk0pdt

        self.alpha_t = 0.0
        if 'alpha_t' in gid:
            # this is the temperature derivation of
            # the pressure derivative of isotropic bulk modulus
            alpha_t = np.array(gid.get('alpha_t'), dtype=np.float64).item()
            self.alpha_t = alpha_t

        self.dalpha_t_dt = 0.0
        if 'dalpha_t_dt' in gid:
            # this is the temperature derivation of
            # the pressure derivative of isotropic bulk modulus
            dalpha_t_dt = np.array(gid.get('dalpha_t_dt'),
                                   dtype=np.float64).item()
            self.dalpha_t_dt = dalpha_t_dt

        '''Finished with the BM EOS
        '''

        if 'v0' in gid:
            # this is the isotropic ambient unitcell
            # volume
            v0 = np.array(gid.get('v0'), dtype=np.float64).item()
            self.v0 = v0

        # if dmin is present in file:
        if 'dmin' in gid:
            # if dmin is present in the HDF5 file, then use that
            dmin = np.array(gid.get('dmin'), dtype=np.float64).item()
            self._dmin = _angstroms(dmin * 10.0)
        else:
            self._dmin = Material.DFLT_DMIN

        # if kev is present in file
        if 'kev' in gid:
            kev = np.array(gid.get('kev'), dtype=np.float64).item()
            self._beamEnergy = _kev(kev)
        else:
            self._beamEnergy = Material.DFLT_KEV

        if 'tThWidth' in gid:
            tThWidth = np.array(gid.get('tThWidth'), dtype=np.float64).item()
            tThWidth = np.radians(tThWidth)
        else:
            tThWidth = Material.DFLT_TTH

        self._tThWidth = tThWidth

        if 'hkls' in gid:
            self.hkl_from_file = np.array(gid.get('hkls'), dtype=np.int32)

        if isinstance(fhdf, (Path, str)):
            # Close the file...
            root_gid.close()

    def dump_material(self, file, path=None):
        '''
        get the atominfo dictionaary aand the lattice parameters
        '''
        AtomInfo = {}

        AtomInfo['file'] = file
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
        AtomInfo['kev'] = self.beamEnergy.getVal("keV")
        if self.planeData.tThWidth is None:
            AtomInfo['tThWidth'] = np.degrees(Material.DFLT_TTH)
        else:
            AtomInfo['tThWidth'] = np.degrees(self.planeData.tThWidth)

        AtomInfo['pressure'] = self.pressure
        AtomInfo['temperature'] = self.temperature

        AtomInfo['k0'] = 100.0
        if hasattr(self, 'k0'):
            AtomInfo['k0'] = self.k0

        AtomInfo['k0p'] = 0.0
        if hasattr(self, 'k0p'):
            AtomInfo['k0p'] = self.k0p

        AtomInfo['v0'] = self.vol
        if hasattr(self, 'v0'):
            AtomInfo['v0'] = self.v0

        AtomInfo['dk0dt'] = 0.0
        if hasattr(self, 'dk0dt'):
            AtomInfo['dk0dt'] = self.dk0dt

        AtomInfo['dk0pdt'] = 0.0
        if hasattr(self, 'dk0pdt'):
            AtomInfo['dk0pdt'] = self.dk0pdt

        AtomInfo['alpha_t'] = 0.0
        if hasattr(self, 'alpha_t'):
            AtomInfo['alpha_t'] = self.alpha_t

        AtomInfo['dalpha_t_dt'] = 0.0
        if hasattr(self, 'dalpha_t_dt'):
            AtomInfo['dalpha_t_dt'] = self.dalpha_t_dt
        '''
        lattice parameters
        '''
        lat_param = {
            'a': self.unitcell.a,
            'b': self.unitcell.b,
            'c': self.unitcell.c,
            'alpha': self.unitcell.alpha,
            'beta': self.unitcell.beta,
            'gamma': self.unitcell.gamma,
        }

        Write2H5File(AtomInfo, lat_param, path)

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
        return self.unitcell.vol * 1e3

    @property
    def lparms(self):
        return np.array(
            [
                x.getVal("nm") if x.isLength() else x.getVal("degrees")
                for x in self._lparms
            ]
        )

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
        Uarr = np.array(Uarr)
        self._U = Uarr

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

    sgnum = property(_get_sgnum, _set_sgnum, None, "Space group number")

    @property
    def pgnum(self):
        return self.unitcell.pgnum

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
        if isinstance(keV, valWUnit):
            self._beamEnergy = keV
        else:
            self._beamEnergy = valWUnit('kev', 'energy', keV, 'keV')

        '''
        acceleration potential is set in volts therefore the factor of 1e3
        @TODO make voltage valWUnit instance so this that we dont have to
        track this manually inn the future
        '''
        self.unitcell.voltage = self.beamEnergy.value * 1e3
        self.planeData.wavelength = keV
        self._hkls_changed()

    beamEnergy = property(
        _get_beamEnergy, _set_beamEnergy, None, "Beam energy in keV"
    )

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
        if len(v) != 6:
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

    name = property(_get_name, _set_name, None, "Name of material")

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
            msg = (
                "incorrect size of charge. "
                "must be same size as number of aom types."
            )
            raise ValueError(msg)
        """
        now check if charge states are actually allowed
        """
        for ii, z in enumerate(self.atomtype):
            elem = ptableinverse[z]
            cs = chargestate[elem]
            if not vals[ii] in cs:
                msg = (
                    f"element {elem} does not allow "
                    f"charge state of {vals[ii]}. "
                    f"allowed charge states are {cs}."
                )
                raise ValueError(msg)

        self._charge = vals
        # self._newUnitcell()
        # self.invalidate_structure_factor()

    @property
    def absorption_length(self):
        return self.unitcell.absorption_length

    @property
    def natoms(self):
        return self.atominfo.shape[0]

    # property: "atominfo"

    def _get_atominfo(self):
        """get method for name"""
        return self._atominfo

    def _set_atominfo(self, v):
        """set method for name"""
        if v.ndim != 2:
            raise ValueError("input must be 2-d.")
        if v.shape[1] != 4:
            raise ValueError("enter x, y, z, occ as nx4 array")

        self._atominfo = v

    atominfo = property(
        _get_atominfo,
        _set_atominfo,
        None,
        "Information about atomic positions and electron number",
    )

    # property: "atomtype"
    def _get_atomtype(self):
        """get method for name"""
        return self._atomtype

    def _set_atomtype(self, v):
        """set method for atomtype"""
        self._atomtype = v

    atomtype = property(
        _get_atomtype, _set_atomtype, None, "Information about atomic types"
    )

    @property
    def thermal_expansion(self):
        return self.alpha_t

    @thermal_expansion.setter
    def thermal_expansion(self, val):
        self.alpha_t = val

    @property
    def thermal_expansion_dt(self):
        return self.dalpha_t_dt

    @thermal_expansion_dt.setter
    def thermal_expansion_dt(self, val):
        self.dalpha_t_dt = val

    def _set_atomdata(self, atomtype, atominfo, U, charge):
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
        atomtype = np.array(atomtype)
        atominfo = np.array(atominfo)
        U = np.array(U)

        if atomtype.shape[0] != atominfo.shape[0]:
            msg = (
                f"inconsistent shapes: number of atoms "
                f"types passed = {atomtype.shape[0]} \n"
                f" number of atom positions passed = {atominfo.shape[0]}"
            )
            raise ValueError(msg)

        if atomtype.shape[0] != U.shape[0]:
            msg = (
                f"inconsistent shapes: number of atoms "
                f"types passed = {atomtype.shape[0]} \n"
                f" U passed for {U.shape[0]} atoms."
            )
            raise ValueError(msg)

        if atominfo.shape[0] != U.shape[0]:
            msg = (
                f"inconsistent shapes: number of atom "
                f"positions passed = {atominfo.shape[0]} \n"
                f"U passed for {U.shape[0]} atoms."
            )
            raise ValueError(msg)

        if len(charge) != atomtype.shape[0]:
            msg = (
                f"inconsistent shapes: number of atoms "
                f"types passed = {atomtype.shape[0]} \n"
                f"charge value passed for {len(charge)} atoms."
            )
            raise ValueError(msg)

        self.atomtype = atomtype
        self.atominfo = atominfo
        self.U = U
        self.charge = charge

        self._newUnitcell()
        self.invalidate_structure_factor()

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

    The file uses the config file format.  See ConfigParser module."""
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


def load_materials_hdf5(
    f, dmin=None, kev=None, sgsetting=Material.DFLT_SGSETTING
):
    """Load materials from an HDF5 file

    The file uses the HDF5 file format.
    """
    if isinstance(f, (Path, str)):
        with h5py.File(f, 'r') as rf:
            names = list(rf)
    elif isinstance(f, h5py.Group):
        names = list(f)
    else:
        raise Exception(f'Unknown type for materials file: {f}')

    if isinstance(dmin, float):
        dmin = _angstroms(dmin)

    if isinstance(kev, float):
        kev = _kev(kev)

    kwargs = {
        'material_file': f,
        'dmin': dmin,
        'kev': kev,
        'sgsetting': sgsetting,
    }
    return {name: Material(name, **kwargs) for name in names}


def save_materials_hdf5(f, materials, path=None):
    """Save a dict of materials into an HDF5 file"""
    for material in materials.values():
        material.dump_material(f, path)


def hkls_match(a, b):
    # Check if hkls match. Expects inputs to have shape (x, 3).
    def sorted_hkls(x):
        return x[np.lexsort((x[:, 2], x[:, 1], x[:, 0]))]

    if a.shape != b.shape:
        return False

    return np.array_equal(sorted_hkls(a), sorted_hkls(b))


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
    return np.intersect1d(**kwargs)[1:]


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
