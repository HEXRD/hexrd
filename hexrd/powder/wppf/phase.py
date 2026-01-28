from abc import ABC, abstractmethod
import copy
import importlib.resources
from pathlib import Path
import warnings

import h5py
import numpy as np
import yaml

from hexrd.core import constants
from hexrd.core.material import Material, symmetry, symbols
from hexrd.core.material.spacegroup import Allowed_HKLs, SpaceGroup
from hexrd.core.material.unitcell import _calcstar, _rqpDict
from hexrd.core.valunits import _nm, valWUnit
from hexrd.powder.wppf.xtal import (
    _calc_dspacing,
    _get_tth,
    _calcxrsf,
    _calc_extinction_factor,
    _calc_absorption_factor,
    _get_sf_hkl_factors,
)
import hexrd.core.resources


class AbstractMaterial:
    # This only exists as a placeholder for return annotation
    pass


class Material_LeBail(AbstractMaterial):
    """
    ========================================================================================
    ========================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
                    09/14/2020 SS 1.1 class can now be initialized using
                    a material.Material class instance
    >> @DETAILS:    Material_LeBail class is a stripped down version of the
                    material.Material class.this is done to keep the class
                    lightweight
                    and make sure only the information necessary for the lebail fit is kept

    =========================================================================================
    =========================================================================================
    """

    def __init__(self, fhdf=None, xtal=None, dmin=None, material_obj=None):
        self._shkl = np.zeros((15,))

        if isinstance(material_obj, Material):
            self._init_from_materials(material_obj)
            return
        elif isinstance(material_obj, Material_Rietveld):
            self._init_from_rietveld(material_obj)
            return
        elif material_obj is not None:
            raise ValueError(
                "Invalid material_obj argument. \
                only Material class can be passed here."
            )

        # Default initialization without a material
        """
        dmin in nm
        """
        self.dmin = dmin.value
        self._readHDF(fhdf, xtal)
        self._calcrmt()
        self.sf_and_twin_probability()
        (
            self.SYM_SG,
            self.SYM_PG_d,
            self.SYM_PG_d_laue,
            self.centrosymmetric,
            self.symmorphic,
        ) = symmetry.GenerateSGSym(self.sgnum, self.sgsetting)
        self.latticeType = symmetry.latticeType(self.sgnum)
        self.sg_hmsymbol = symbols.pstr_spacegroup[self.sgnum - 1].strip()
        self.GenerateRecipPGSym()
        self.CalcMaxGIndex()
        self._calchkls()
        self.sg = SpaceGroup(self.sgnum)

    def _init_from_materials(self, material_obj):
        """
        this function is used to initialize the materials_lebail class
        from an instance of the material.Material class. this option is
        provided for easy integration of the hexrdgui with WPPF.

        O7/01/2021 SS ADDED DIRECT AND RECIPROCAL STRUCTURE MATRIX AS
        FIELDS IN THE CLASS
        """
        self.name = material_obj.name.replace('-', '_')
        self.dmin = material_obj.dmin.getVal('nm')
        self.sgnum = material_obj.unitcell.sgnum
        self.sgsetting = material_obj.sgsetting
        self.sg = SpaceGroup(self.sgnum)
        self.sf_and_twin_probability()

        # lattice parameters
        self.lparms = np.array(
            [
                material_obj.unitcell.a,
                material_obj.unitcell.b,
                material_obj.unitcell.c,
                material_obj.unitcell.alpha,
                material_obj.unitcell.beta,
                material_obj.unitcell.gamma,
            ]
        )

        self.latticeType = material_obj.unitcell.latticeType
        self.sg_hmsymbol = material_obj.unitcell.sg_hmsymbol

        # get maximum indices for sampling hkl
        self.ih = material_obj.unitcell.ih
        self.ik = material_obj.unitcell.ik
        self.il = material_obj.unitcell.il

        """ get all space and point group symmetry operators
         in direct space, including the laue group. reciprocal
         space point group symmetries also included """
        self.SYM_SG = material_obj.unitcell.SYM_SG
        self.SYM_PG_d = material_obj.unitcell.SYM_PG_d
        self.SYM_PG_d_laue = material_obj.unitcell.SYM_PG_d_laue
        self.SYM_PG_r = material_obj.unitcell.SYM_PG_r
        self.SYM_PG_r_laue = material_obj.unitcell.SYM_PG_r_laue

        self.centrosymmetric = material_obj.unitcell.centrosymmetric
        self.symmorphic = material_obj.unitcell.symmorphic

        self.hkls = material_obj.planeData.getHKLs()

        self._calcrmt()

    def _init_from_rietveld(self, mat: 'Material_Rietveld'):
        # Just copy over the attributes we need
        attrs_to_copy = [
            'name',
            'dmin',
            'sgnum',
            'sgsetting',
            'sg',
            'lparms',
            'latticeType',
            'sg_hmsymbol',
            'ih',
            'ik',
            'il',
            'sf_alpha',
            'twin_beta',
            'SYM_SG',
            'SYM_PG_d',
            'SYM_PG_d_laue',
            'SYM_PG_r',
            'SYM_PG_r_laue',
            'centrosymmetric',
            'symmorphic',
            'hkls',
        ]
        for name in attrs_to_copy:
            setattr(self, name, copy.deepcopy(getattr(mat, name)))

        self._calcrmt()

    def _readHDF(self, fhdf, xtal):
        with h5py.File(fhdf, 'r') as f:
            name = xtal
            if xtal not in f:
                raise IOError("crystal doesn't exist in material file.")

            group = f[xtal]

            self.sgnum = group['SpaceGroupNumber']
            self.sgsetting = group['SpaceGroupSetting']
            """
                IMPORTANT NOTE:
                note that the latice parameters in EMsoft is nm by default
                hexrd on the other hand uses A as the default units, so we
                need to be careful and convert it right here, so there is no
                confusion later on
            """
            self.lparms = list(group['LatticeParameters'])
            self.name = name.replace('-', '_')

    def _calcrmt(self):
        """
        O7/01/2021 SS ADDED DIRECT AND RECIPROCAL STRUCTURE MATRIX AS
        FIELDS IN THE CLASS
        """
        a = self.lparms[0]
        b = self.lparms[1]
        c = self.lparms[2]

        alpha = np.radians(self.lparms[3])
        beta = np.radians(self.lparms[4])
        gamma = np.radians(self.lparms[5])

        ca = np.cos(alpha)
        cb = np.cos(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
        tg = np.tan(gamma)

        """
            direct metric tensor
        """
        self.dmt = np.array(
            [
                [a**2, a * b * cg, a * c * cb],
                [a * b * cg, b**2, b * c * ca],
                [a * c * cb, b * c * ca, c**2],
            ]
        )
        self.vol = np.sqrt(np.linalg.det(self.dmt))

        if self.vol < 1e-5:
            warnings.warn('unitcell volume is suspiciously small')

        """
            reciprocal metric tensor
        """
        self.rmt = np.linalg.inv(self.dmt)

        """
            direct structure matrix
        """
        self.dsm = np.array(
            [
                [a, b * cg, c * cb],
                [0.0, b * sg, -c * (cb * cg - ca) / sg],
                [0.0, 0.0, self.vol / (a * b * sg)],
            ]
        )
        """
            reciprocal structure matrix
        """
        self.rsm = np.array(
            [
                [1.0 / a, 0.0, 0.0],
                [-1.0 / (a * tg), 1.0 / (b * sg), 0.0],
                [
                    b * c * (cg * ca - cb) / (self.vol * sg),
                    a * c * (cb * cg - ca) / (self.vol * sg),
                    a * b * sg / self.vol,
                ],
            ]
        )

    """ calculate dot product of two vectors in any space 'd' 'r' or 'c' """

    def CalcLength(self, u, space):
        if space == 'c':
            return np.linalg.norm(u)

        if space not in ('d', 'r'):
            raise ValueError('incorrect space argument')

        lhs = self.dmt if space == 'd' else self.rmt
        return np.sqrt(np.dot(u, np.dot(lhs, u)))

    def CalcDot(self, u, v, space):
        if space == 'c':
            return np.dot(u, v)

        if space not in ('d', 'r'):
            raise ValueError('space is unidentified')

        lhs = self.dmt if space == 'd' else self.rmt
        return np.dot(u, np.dot(lhs, v))

    def getTTh(self, wavelength):
        self.dsp = _calc_dspacing(
            self.rmt.astype(np.float64),
            self.hkls.astype(np.float64),
        )
        tth, wavelength_allowed_hkls = _get_tth(self.dsp, wavelength)
        self.wavelength_allowed_hkls = wavelength_allowed_hkls.astype(bool)
        return tth

    def get_sf_hkl_factors(self):
        """
        this function calculates the prefactor for
        each hkl used to calculate the 2theta shifts
        due to stacking faults. for details see EQ. 10
        Velterop et. al., Stacking and twin faults
        J. Appl. Cryst. (2000). 33, 296-306

        currently only doing fcc. will be adding hcp and
        bcc in the future
        adding a guard rail so that the function only
        returns for sgnum 225
        """
        if self.sgnum != 225:
            return None, None

        sym = self.SYM_PG_r.astype(float)
        mat = self.rmt.astype(float)
        return _get_sf_hkl_factors(self.hkls, sym, mat)

    def sf_and_twin_probability(self):
        self.sf_alpha = None
        self.twin_beta = None
        if self.sgnum == 225:
            self.sf_alpha = 0.0
            self.twin_beta = 0.0

    def GenerateRecipPGSym(self):
        self.SYM_PG_r = self.SYM_PG_d[0, :, :]
        self.SYM_PG_r = np.broadcast_to(self.SYM_PG_r, [1, 3, 3])
        self.SYM_PG_r_laue = self.SYM_PG_d[0, :, :]
        self.SYM_PG_r_laue = np.broadcast_to(self.SYM_PG_r_laue, [1, 3, 3])

        for i in range(1, self.SYM_PG_d.shape[0]):
            g = self.SYM_PG_d[i, :, :]
            g = np.dot(self.dmt, np.dot(g, self.rmt))
            g = np.round(np.broadcast_to(g, [1, 3, 3]))
            self.SYM_PG_r = np.concatenate((self.SYM_PG_r, g))

        for i in range(1, self.SYM_PG_d_laue.shape[0]):
            g = self.SYM_PG_d_laue[i, :, :]
            g = np.dot(self.dmt, np.dot(g, self.rmt))
            g = np.round(np.broadcast_to(g, [1, 3, 3]))
            self.SYM_PG_r_laue = np.concatenate((self.SYM_PG_r_laue, g))

        self.SYM_PG_r = self.SYM_PG_r.astype(np.int32)
        self.SYM_PG_r_laue = self.SYM_PG_r_laue.astype(np.int32)

    def CalcMaxGIndex(self):
        self.ih = 1
        while (
            1.0 / self.CalcLength(np.array([self.ih, 0, 0], dtype=np.float64), 'r')
            > self.dmin
        ):
            self.ih = self.ih + 1
        self.ik = 1
        while (
            1.0 / self.CalcLength(np.array([0, self.ik, 0], dtype=np.float64), 'r')
            > self.dmin
        ):
            self.ik = self.ik + 1
        self.il = 1
        while (
            1.0 / self.CalcLength(np.array([0, 0, self.il], dtype=np.float64), 'r')
            > self.dmin
        ):
            self.il = self.il + 1

    def CalcStar(self, v, space, applyLaue=False):
        """
        this function calculates the symmetrically equivalent hkls (or uvws)
        for the reciprocal (or direct) point group symmetry.
        """
        if space not in ('d', 'r'):
            raise ValueError('CalcStar: unrecognized space.')

        suffix = '_laue' if applyLaue else ''
        name = f'SYM_PG_{space}{suffix}'
        sym = getattr(self, name).astype(float)
        v = np.asarray(v).astype(float)
        mat = (self.dmt if space == 'd' else self.rmt).astype(float)
        return _calcstar(v, sym, mat)

    def removeinversion(self, ksym):
        """
        this function chooses a subset from a list
        of symmetrically equivalent reflections such
        that there are no g and -g present.
        """
        klist = []
        for i in range(ksym.shape[0]):
            k = ksym[i, :]
            kk = list(k)
            nkk = list(-k)
            if not klist:
                if np.sum(k) > np.sum(-k):
                    klist.append(kk)
                else:
                    klist.append(nkk)

            else:
                if kk in klist or nkk in klist:
                    pass
                else:
                    klist.append(kk)
        klist = np.array(klist)
        return klist

    def ChooseSymmetric(self, hkllist, InversionSymmetry=True):
        """
        this function takes a list of hkl vectors and
        picks out a subset of the list picking only one
        of the symmetrically equivalent one. The convention
        is to choose the hkl with the most positive components.
        """
        mask = np.ones(hkllist.shape[0], dtype=bool)
        laue = InversionSymmetry
        for i, g in enumerate(hkllist):
            if mask[i]:
                geqv = self.CalcStar(g, 'r', applyLaue=laue)
                for r in geqv[1:,]:
                    rid = np.where(np.all(r == hkllist, axis=1))
                    mask[rid] = False
        hkl = hkllist[mask, :].astype(np.int32)
        hkl_max = []
        for g in hkl:
            geqv = self.CalcStar(g, 'r', applyLaue=laue)
            loc = np.argmax(np.sum(geqv, axis=1))
            gmax = geqv[loc, :]
            hkl_max.append(gmax)
        return np.array(hkl_max).astype(np.int32)

    def SortHKL(self, hkllist):
        """
        this function sorts the hkllist by increasing |g|
        i.e. decreasing d-spacing. If two vectors are same
        length, then they are ordered with increasing
        priority to l, k and h
        """
        glen = []
        for g in hkllist:
            glen.append(np.round(self.CalcLength(g, 'r'), 8))
        # glen = np.atleast_2d(np.array(glen,dtype=float)).T
        dtype = [
            ('glen', float),
            ('max', int),
            ('sum', int),
            ('h', int),
            ('k', int),
            ('l', int),
        ]
        a = []
        for i, gl in enumerate(glen):
            g = hkllist[i, :]
            a.append((gl, np.max(g), np.sum(g), g[0], g[1], g[2]))
        a = np.array(a, dtype=dtype)
        isort = np.argsort(a, order=['glen', 'max', 'sum', 'l', 'k', 'h'])
        return hkllist[isort, :]

    def _calchkls(self):
        self.hkls = self.getHKLs(self.dmin)

    def getHKLs(self, dmin):
        """
        this function generates the symetrically unique set of
        hkls up to a given dmin.
        dmin is in nm
        """
        """
        always have the centrosymmetric condition because of
        Friedels law for xrays so only 4 of the 8 octants
        are sampled for unique hkls. By convention we will
        ignore all l < 0
        """
        hmin = -self.ih - 1
        hmax = self.ih
        kmin = -self.ik - 1
        kmax = self.ik
        lmin = -1
        lmax = self.il
        hkllist = np.array(
            [
                [ih, ik, il]
                for ih in np.arange(hmax, hmin, -1)
                for ik in np.arange(kmax, kmin, -1)
                for il in np.arange(lmax, lmin, -1)
            ]
        )
        hkl_allowed = Allowed_HKLs(self.sgnum, hkllist)
        hkl = []
        hkl_dsp = []
        for g in hkl_allowed:
            # ignore [0 0 0] as it is the direct beam
            if np.sum(np.abs(g)) != 0:
                dspace = 1.0 / self.CalcLength(g, 'r')
                if dspace >= dmin:
                    hkl_dsp.append(g)
        """
        we now have a list of g vectors which are all within dmin range
        plus the systematic absences due to lattice centering and glide
        planes/screw axis has been taken care of
        the next order of business is to go through the list and only pick
        out one of the symetrically equivalent hkls from the list.
        """
        hkl_dsp = np.array(hkl_dsp).astype(np.int32)
        """
        the inversionsymmetry switch enforces the application of the inversion
        symmetry regradless of whether the crystal has the symmetry or not
        this is necessary in the case of xrays due to friedel's law
        """
        hkl = self.ChooseSymmetric(hkl_dsp, InversionSymmetry=True)
        """
        finally sort in order of decreasing dspacing
        """
        return self.SortHKL(hkl)

    def Required_lp(self, p):
        return _rqpDict[self.latticeType][1](p)

    @property
    def shkl(self):
        return self._shkl

    @shkl.setter
    def shkl(self, val):
        """
        set the shkl as array
        """
        if len(val) != 15:
            raise ValueError("shkl shape must be (15, )")

        self._shkl = val


class Material_Rietveld(Material_LeBail):
    """
    ===========================================================================================
    ===========================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
                    02/01/2021 SS 1.1 class can now be initialized using a
                    material.Material class instance
    >> @DETAILS:    Material_LeBail class is a stripped down version of the
                    material.Material class. This is done to keep the class
                    lightweight and make sure only the information necessary
                    for the Rietveld fit is kept
    ===========================================================================================
     ==========================================================================================
    """

    def __init__(self, fhdf=None, xtal=None, dmin=None, kev=None, material_obj=None):
        # First, initialize the LeBail-specific stuff
        super().__init__(fhdf, xtal, dmin, material_obj)

        # Now initialize Rietveld-specific stuff
        # If `material_object` is not `None`, then
        # `_init_from_materials()` will have already been called.
        self.abs_fact = 1e4
        if material_obj is None:
            """
            voltage in ev
            """
            self.voltage = kev.value * 1000.0

            if self.aniU:
                self.calcBetaij()

            self.InitializeInterpTable()
            self.CalcWavelength()
            self.CalcPositions()

    def _init_from_materials(self, material_obj):
        # Initialize the same stuff as LeBail
        super()._init_from_materials(material_obj)

        # Now grab Rietveld-specific stuff

        # inverse of absorption length
        self.abs_fact = 1e-4 * (1.0 / material_obj.absorption_length)

        # acceleration voltage and wavelength
        self.voltage = material_obj.unitcell.voltage
        self.wavelength = material_obj.unitcell.wavelength

        # asymmetric atomic positions
        self.atom_pos = material_obj.unitcell.atom_pos

        # Debye-Waller factors including anisotropic ones
        self.U = material_obj.unitcell.U
        self.aniU = False
        if self.U.ndim > 1:
            self.aniU = True
            self.betaij = material_obj.unitcell.betaij

        # atom types i.e. Z and number of different atom types
        self.atom_type = material_obj.unitcell.atom_type
        self.atom_ntype = material_obj.unitcell.atom_ntype

        # calculate the multiplicities
        self.multiplicity = self.getMultiplicity(self.hkls)

        # interpolation tables and anomalous form factors
        # self.f1 = material_obj.unitcell.f1
        # self.f2 = material_obj.unitcell.f2
        # self.f_anam = material_obj.unitcell.f_anam

        # final step is to calculate the asymmetric positions in
        # the unit cell
        self.numat = material_obj.unitcell.numat
        self.asym_pos = material_obj.unitcell.asym_pos
        self.InitializeInterpTable()

    def _readHDF(self, fhdf, xtal):
        # First, read the same things as LeBail
        super()._readHDF(fhdf, xtal)

        # Now read in Rietveld-specific stuff
        with h5py.File(fhdf, 'r') as f:
            group = f[xtal]
            # the last field in this is already
            self.atom_pos = group['AtomData'].T

            # the U factors are related to B by the relation B = 8pi^2 U
            self.U = group['U'].T

            # read atom types (by atomic number, Z)
            self.atom_type = group['Atomtypes']
            self.atom_ntype = self.atom_type.shape[0]

    def calcBetaij(self):
        self.betaij = np.zeros([3, 3, self.atom_ntype])
        for i in range(self.U.shape[0]):
            U = self.U[i, :]
            self.betaij[:, :, i] = np.array(
                [[U[0], U[3], U[4]], [U[3], U[1], U[5]], [U[4], U[5], U[2]]]
            )

            self.betaij[:, :, i] *= 2.0 * np.pi**2 * self.aij

    def CalcWavelength(self):
        # wavelength in nm
        self.wavelength = (
            constants.cPlanck * constants.cLight / constants.cCharge / self.voltage
        )
        self.wavelength *= 1e9
        # self.CalcAnomalous()

    def CalcKeV(self):
        self.kev = (
            constants.cPlanck * constants.cLight / constants.cCharge / self.wavelength
        )

        self.kev *= 1e-3

    def _calcrmt(self):
        super()._calcrmt()
        ast = self.CalcLength([1, 0, 0], 'r')
        bst = self.CalcLength([0, 1, 0], 'r')
        cst = self.CalcLength([0, 0, 1], 'r')

        self.aij = np.array(
            [
                [ast**2, ast * bst, ast * cst],
                [bst * ast, bst**2, bst * cst],
                [cst * ast, cst * bst, cst**2],
            ]
        )

    def _calchkls(self):
        super()._calc_hkls()
        self.multiplicity = self.getMultiplicity(self.hkls)

    ''' transform between any crystal space to any other space.
        choices are 'd' (direct), 'r' (reciprocal) and 'c' (cartesian)'''

    def TransSpace(self, v_in, inspace, outspace):
        if inspace == 'd':
            if outspace == 'r':
                v_out = np.dot(v_in, self.dmt)
            elif outspace == 'c':
                v_out = np.dot(self.dsm, v_in)
            else:
                raise ValueError(
                    'inspace in ' 'd' ' but outspace can' 't be identified'
                )
        elif inspace == 'r':
            if outspace == 'd':
                v_out = np.dot(v_in, self.rmt)
            elif outspace == 'c':
                v_out = np.dot(self.rsm, v_in)
            else:
                raise ValueError(
                    'inspace in ' 'r' ' but outspace can' 't be identified'
                )
        elif inspace == 'c':
            if outspace == 'r':
                v_out = np.dot(v_in, self.rsm)
            elif outspace == 'd':
                v_out = np.dot(v_in, self.dsm)
            else:
                raise ValueError(
                    'inspace in ' 'c' ' but outspace can' 't be identified'
                )
        else:
            raise ValueError('incorrect inspace argument')
        return v_out

    def getMultiplicity(self, hkls):
        return np.array([self.CalcStar(g, 'r').shape[0] for g in hkls])

    def CalcPositions(self):
        """
        calculate the asymmetric positions in the fundamental unitcell
        used for structure factor calculations
        """
        numat = []
        asym_pos = []

        # using the wigner-seitz notation
        for i in range(self.atom_ntype):
            n = 1
            r = self.atom_pos[i, 0:3]
            r = np.hstack((r, 1.0))

            asym_pos.append(np.broadcast_to(r[0:3], [1, 3]))

            for symmat in self.SYM_SG:
                # get new position
                rnew = np.dot(symmat, r)

                # reduce to fundamental unitcell with fractional
                # coordinates between 0-1
                rr = rnew[0:3]
                rr = np.modf(rr)[0]
                rr[rr < 0.0] += 1.0
                rr[np.abs(rr) < 1.0e-6] = 0.0

                # check if this is new
                isnew = True
                for j in range(n):
                    if np.sum(np.abs(rr - asym_pos[i][j, :])) < 1e-4:
                        isnew = False
                        break

                # if its new add this to the list
                if isnew:
                    asym_pos[i] = np.vstack((asym_pos[i], rr))
                    n += 1

            numat.append(n)

        self.numat = np.array(numat)
        self.asym_pos = asym_pos

    def InitializeInterpTable(self):
        f_anomalous_data = []
        data = importlib.resources.open_binary(hexrd.core.resources, 'Anomalous.h5')
        with h5py.File(data, 'r') as fid:
            for i in range(0, self.atom_ntype):
                Z = self.atom_type[i]
                elem = constants.ptableinverse[Z]
                gid = fid.get('/' + elem)
                data = np.array(gid.get('data'))
                data = data[:, [7, 1, 2]]
                f_anomalous_data.append(data)

        n = max([x.shape[0] for x in f_anomalous_data])
        self.f_anomalous_data = np.zeros([self.atom_ntype, n, 3])
        self.f_anomalous_data_sizes = np.zeros(
            [
                self.atom_ntype,
            ],
            dtype=np.int32,
        )

        for i in range(self.atom_ntype):
            nd = f_anomalous_data[i].shape[0]
            self.f_anomalous_data_sizes[i] = nd
            self.f_anomalous_data[i, :nd, :] = f_anomalous_data[i]

    def CalcXRSF(self, wavelength, w_int):
        """
        the 1E-2 is to convert to A^-2
        since the fitting is done in those units
        """
        fNT = np.zeros([self.atom_ntype])
        frel = np.zeros([self.atom_ntype])
        scatfac = np.zeros([self.atom_ntype, 11])
        f_anomalous_data = self.f_anomalous_data

        aniU = self.aniU
        occ = self.atom_pos[:, 3]
        if aniU:
            betaij = self.betaij
        else:
            betaij = self.U

        self.numat = np.zeros(self.atom_ntype, dtype=np.int32)
        for i in range(0, self.atom_ntype):
            self.numat[i] = self.asym_pos[i].shape[0]
            Z = self.atom_type[i]
            elem = constants.ptableinverse[Z]
            scatfac[i, :] = constants.scatfac[elem]
            frel[i] = constants.frel[elem]
            fNT[i] = constants.fNT[elem]

        self.asym_pos_arr = np.zeros([self.numat.max(), self.atom_ntype, 3])
        for i in range(0, self.atom_ntype):
            nn = self.numat[i]
            self.asym_pos_arr[:nn, i, :] = self.asym_pos[i]

        nref = self.hkls.shape[0]

        return _calcxrsf(
            self.hkls.astype(np.float64),
            nref,
            self.multiplicity,
            w_int,
            wavelength,
            self.rmt.astype(np.float64),
            self.atom_type,
            self.atom_ntype,
            betaij,
            occ,
            self.asym_pos_arr,
            self.numat,
            scatfac,
            fNT,
            frel,
            f_anomalous_data,
            self.f_anomalous_data_sizes,
        )

    def calc_extinction(self, wavelength, tth, f_sqr, shape_factor_K, particle_size_D):
        return _calc_extinction_factor(
            self.hkls,
            tth,
            self.vol * 1e3,
            wavelength,
            f_sqr,
            shape_factor_K,
            particle_size_D,
        )

    def calc_absorption(self, tth, phi, wavelength):
        abs_fact = self.abs_fact
        absorption = _calc_absorption_factor(abs_fact, tth, phi, wavelength)

        return absorption

    def calc_temperature(self, T_debye):
        '''use the classical debye model to convert U factor to
        an equivalent temperature. The T_debye is a dictionary
        with the keys as atom types and the values are the debye
        temperatures for the atom type.

        M = B = 8 * pi^2 * U = 6 h^2 /(M_atomic*K_B) * (T/T_debye^2)
        '''
        T = dict.fromkeys(T_debye)
        for U, a in zip(self.U, self.atom_type):
            key = constants.ptableinverse[a]
            mass = constants.ATOM_WEIGHTS_DICT[k]
            # the standard are in m^2 in S.I. units and U is in A^2
            # so the 1E20 factor accounts for that
            pre = (
                6
                * constants.cPlanck**2
                / (mass * 1e-3 / constants.cAvogadro)
                / constants.cBoltzmann
            ) * 1e20
            if key in T_debye:
                T[key] = (8 * np.pi**2 * U / pre) * T_debye[key] ** 2

        return T


class AbstractPhases(ABC):
    """
    ========================================================================================
    ========================================================================================
    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/20/2020 SS 1.0 original
    >> @DETAILS:    class to handle different phases in the fits. this is a stripped down
                    version of main Phase class for efficiency. only the
                    components necessary for calculating peak positions are retained. further
                    this will have a slight modification to account for different wavelengths
                    in the same phase name
    =========================================================================================
    =========================================================================================
    """

    # Abstract methods which must be defined for each phase type
    @abstractmethod
    def _get_phase(self, material_key: str, wavelength_name: str) -> AbstractMaterial:
        pass

    @abstractmethod
    def add(self, material_file, material_key):
        pass

    # Shared methods which each phase uses
    def __init__(
        self,
        material_file=None,
        material_keys=None,
        dmin=_nm(0.05),
        wavelength={'alpha1': [_nm(0.15406), 1.0], 'alpha2': [_nm(0.154443), 0.52]},
    ):
        self.phase_dict = {}

        """
        set wavelength. check if wavelength is supplied in A, if it is
        convert to nm since the rest of the code assumes those units
        """
        wavelength_nm = {}
        for k, v in wavelength.items():
            if v[0].unit == 'angstrom':
                wavelength_nm[k] = [
                    valWUnit('lp', 'length', v[0].getVal("nm"), 'nm'),
                    v[1],
                ]
            else:
                wavelength_nm[k] = v

        self.wavelength = wavelength_nm

        self.dmin = dmin

        if material_file is not None and material_keys is not None:
            keys = material_keys
            keys = keys if isinstance(keys, list) else [keys]
            self.add_many(material_file, keys)

    @property
    def num_phases(self) -> int:
        return len(self)

    def __str__(self):
        resstr = 'Phases in calculation:\n'
        for i, k in enumerate(self.phase_dict):
            resstr += f'\t{i+1}. {k}\n'
        return resstr

    def __getitem__(self, key):
        # Always sanitize the material name since lmfit won't accept '-'
        key = key.replace('-', '_')
        return self.phase_dict[key]

    def __setitem__(self, key, mat_cls):
        # Always sanitize the material name since lmfit won't accept '-'
        key = key.replace('-', '_')
        self.phase_dict[key] = mat_cls

    def __iter__(self):
        return iter(self.phase_dict)

    def __len__(self):
        return len(self.phase_dict)

    def reset_phase_fractions(self):
        pf = 1.0 / self.num_phases
        for k in self:
            for l in self.wavelength:
                mat = self._get_phase(k, l)
                mat.pf = pf

    def add_many(self, material_file, material_keys):
        for k in material_keys:
            self.add(material_file, k, update_pf=False)

        self.reset_phase_fractions()

        self.material_file = material_file
        self.material_keys = [k.replace('-', '_') for k in material_keys]

    def load(self, fname):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    load parameters from yaml file
        """
        with open(fname) as file:
            dic = yaml.load(file, Loader=yaml.SafeLoader)

        for mfile in dic.keys():
            mat_keys = list(dic[mfile])
            self.add_many(mfile, mat_keys)

    def dump(self, fname):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    dump parameters to yaml file
        """
        dic = {self.material_file: [m for m in self]}
        with open(fname, 'w') as f:
            yaml.safe_dump(dic, f, sort_keys=False)

    def dump_hdf5(self, file):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       01/15/2021 SS 1.0 original
        >> @ DETAILS    dumps the information from each material in the phase class
                        to a hdf5 file specified by filename or h5py.File object
        """
        if isinstance(file, str):
            mode = 'r+' if Path(file).exists() else 'x'
            fid = h5py.File(mode)
        elif isinstance(file, h5py.File):
            fid = file
        else:
            raise RuntimeError(
                'Parameters: dump_hdf5 Pass in a filename \
                string or h5py.File object'
            )

        if "/Phases" in fid:
            del fid["Phases"]

        gid_top = fid.create_group("Phases")
        # Only write out the first material
        l = next(iter(self.wavelength))
        for p in self:
            mat = self._get_phase(p, l)
            gid = gid_top.create_group(p)
            gid["SpaceGroupNumber"] = mat.sgnum
            gid["SpaceGroupSetting"] = mat.sgsetting
            gid["LatticeParameters"] = np.asarray(mat.lparms)
            gid["dmin"] = mat.dmin
            gid["dmin"].attrs["units"] = "nm"
            gid["hkls"] = np.asarray(mat.hkls)


class Phases_LeBail(AbstractPhases):
    def _get_phase(self, material_key: str, wavelength_name: str) -> Material_LeBail:
        return self[material_key]

    def add(self, material_file, material_key, update_pf=True):
        self[material_key] = Material_LeBail(
            fhdf=material_file, xtal=material_key, dmin=self.dmin
        )

        if update_pf:
            self.reset_phase_fractions()


class Phases_Rietveld(AbstractPhases):
    def _get_phase(self, material_key: str, wavelength_name: str) -> Material_Rietveld:
        return self[material_key][wavelength_name]

    def add(self, material_file, material_key, update_pf=True):
        self[material_key] = {}
        for l in self.wavelength:
            lam = self.wavelength[l][0].getVal('nm') * 1e-9
            E = constants.cPlanck * constants.cLight / constants.cCharge / lam
            E *= 1e-3
            kev = valWUnit('beamenergy', 'energy', E * 1e-3, 'keV')
            self[material_key][l] = Material_Rietveld(
                material_file, material_key, dmin=self.dmin, kev=kev
            )

        if update_pf:
            self.reset_phase_fractions()

    @property
    def phase_fraction(self):
        l = next(iter(self.wavelength))
        pf = np.array([self[k][l].pf for k in self])
        return pf / pf.sum()

    @phase_fraction.setter
    def phase_fraction(self, val):
        if len(val) != len(self):
            raise ValueError("number of phases does not match size of input")

        for ii, k in enumerate(self):
            for l in self.wavelength:
                self[k][l].pf = val[ii]
