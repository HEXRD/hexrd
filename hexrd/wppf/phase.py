0import numpy as np
from hexrd.valunits import valWUnit
from hexrd.spacegroup import Allowed_HKLs, SpaceGroup
from hexrd import symmetry, symbols, constants
from hexrd.material import Material
from hexrd.unitcell import _rqpDict
from hexrd.wppf import wppfsupport
from hexrd.wppf.xtal import _calc_dspacing, _get_tth, _calcxrsf,\
_calc_extinction_factor, _calc_absorption_factor
import h5py
import importlib.resources
import hexrd.resources

class Material_LeBail:
    """ 
    ========================================================================================
    ========================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
                    09/14/2020 SS 1.1 class can now be initialized using
                    a material.Material class instance
    >> @DETAILS:    Material_LeBail class is a stripped down version of the 
                    materials.Material class.this is done to keep the class lightweight 
                    and make sure only the information necessary for the lebail fit is kept

    =========================================================================================
    =========================================================================================
    """

    def __init__(self,
                 fhdf=None,
                 xtal=None,
                 dmin=None,
                 material_obj=None):

        if(material_obj is None):
            self.dmin = dmin.value
            self._readHDF(fhdf, xtal)
            self._calcrmt()

            _, self.SYM_PG_d, self.SYM_PG_d_laue, \
                self.centrosymmetric, self.symmorphic = \
                symmetry.GenerateSGSym(self.sgnum, self.sgsetting)
            self.latticeType = symmetry.latticeType(self.sgnum)
            self.sg_hmsymbol = symbols.pstr_spacegroup[self.sgnum-1].strip()
            self.GenerateRecipPGSym()
            self.CalcMaxGIndex()
            self._calchkls()
            self.sg = SpaceGroup(self.sgnum)

        else:
            if(isinstance(material_obj, Material)):
                self._init_from_materials(material_obj)
            else:
                raise ValueError(
                    "Invalid material_obj argument. \
                    only Material class can be passed here.")
        self._shkl = np.zeros((15,))

    def _readHDF(self, fhdf, xtal):

        # fexist = path.exists(fhdf)
        # if(fexist):
        fid = h5py.File(fhdf, 'r')
        name = xtal
        xtal = "/"+xtal
        if xtal not in fid:
            raise IOError('crystal doesn''t exist in material file.')
        # else:
        #   raise IOError('material file does not exist.')

        gid = fid.get(xtal)

        self.sgnum = np.asscalar(np.array(gid.get('SpaceGroupNumber'),
                                          dtype=np.int32))
        self.sgsetting = np.asscalar(np.array(gid.get('SpaceGroupSetting'),
                                              dtype=np.int32))
        """
            IMPORTANT NOTE:
            note that the latice parameters is nm by default
            hexrd on the other hand uses A as the default units, so we
            need to be careful and convert it right here, so there is no
            confusion later on
        """
        self.lparms = list(gid.get('LatticeParameters'))
        self.name = name
        fid.close()

    def _init_from_materials(self, material_obj):
        """
        this function is used to initialize the materials_lebail class
        from an instance of the material.Material class. this option is
        provided for easy integration of the hexrdgui with WPPF.

        O7/01/2021 SS ADDED DIRECT AND RECIPROCAL STRUCTURE MATRIX AS
        FIELDS IN THE CLASS
        """
        self.name = material_obj.name
        self.dmin = material_obj.dmin.getVal('nm')
        self.sgnum = material_obj.unitcell.sgnum
        self.sgsetting = material_obj.sgsetting
        self.sg = SpaceGroup(self.sgnum)

        if(material_obj.latticeParameters[0].unit == 'nm'):
            self.lparms = [x.value for x in material_obj.latticeParameters]
        elif(material_obj.latticeParameters[0].unit == 'angstrom'):
            lparms = [x.value for x in material_obj.latticeParameters]
            for i in range(3):
                lparms[i] /= 10.0
            self.lparms = lparms

        self.dmt = material_obj.unitcell.dmt
        self.rmt = material_obj.unitcell.rmt
        self.dsm = material_obj.unitcell.dsm
        self.rsm = material_obj.unitcell.rsm
        self.vol = material_obj.unitcell.vol

        self.centrosymmetric = material_obj.unitcell.centrosymmetric
        self.symmorphic = material_obj.unitcell.symmorphic

        self.latticeType = material_obj.unitcell.latticeType
        self.sg_hmsymbol = material_obj.unitcell.sg_hmsymbol

        self.ih = material_obj.unitcell.ih
        self.ik = material_obj.unitcell.ik
        self.il = material_obj.unitcell.il

        self.SYM_PG_d = material_obj.unitcell.SYM_PG_d
        self.SYM_PG_d_laue = material_obj.unitcell.SYM_PG_d_laue
        self.SYM_PG_r = material_obj.unitcell.SYM_PG_r
        self.SYM_PG_r_laue = material_obj.unitcell.SYM_PG_r_laue

        self.hkls = material_obj.planeData.getHKLs()

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
        sa = np.sin(alpha)
        sb = np.sin(beta)
        sg = np.sin(gamma)
        tg = np.tan(gamma)

        """
            direct metric tensor
        """
        self.dmt = np.array([[a**2, a*b*cg, a*c*cb],
                             [a*b*cg, b**2, b*c*ca],
                             [a*c*cb, b*c*ca, c**2]])
        self.vol = np.sqrt(np.linalg.det(self.dmt))

        if(self.vol < 1e-5):
            warnings.warn('unitcell volume is suspiciously small')

        """
            reciprocal metric tensor
        """
        self.rmt = np.linalg.inv(self.dmt)

        """
            direct structure matrix
        """
        self.dsm = np.array([[a, b*cg, c*cb],
                              [0., b*sg, -c*(cb*cg - ca)/sg],
                              [0., 0., self.vol/(a*b*sg)]])

        """
            reciprocal structure matrix
        """
        self.rsm = np.array([[1./a, 0., 0.],
                              [-1./(a*tg), 1./(b*sg), 0.],
                              [b*c*(cg*ca - cb)/(self.vol*sg),
                               a*c*(cb*cg - ca)/(self.vol*sg),
                               a*b*sg/self.vol]])

    def _calchkls(self):
        self.hkls = self.getHKLs(self.dmin)

    """ calculate dot product of two vectors in any space 'd' 'r' or 'c' """

    def CalcLength(self, u, space):

        if(space == 'd'):
            vlen = np.sqrt(np.dot(u, np.dot(self.dmt, u)))
        elif(space == 'r'):
            vlen = np.sqrt(np.dot(u, np.dot(self.rmt, u)))
        elif(spec == 'c'):
            vlen = np.linalg.norm(u)
        else:
            raise ValueError('incorrect space argument')

        return vlen

    def CalcDot(self, u, v, space):
        if(space == 'd'):
            dot = np.dot(u, np.dot(self.dmt, v))
        elif(space == 'r'):
            dot = np.dot(u, np.dot(self.rmt, v))
        elif(space == 'c'):
            dot = np.dot(u, v)
        else:
            raise ValueError('space is unidentified')
        return dot

    def getTTh(self, wavelength):

        tth = []
        self.dsp = _calc_dspacing(self.rmt.astype(np.float64),
            self.hkls.astype(np.float64))
        tth, wavelength_allowed_hkls = \
        _get_tth(self.dsp, wavelength)
        self.wavelength_allowed_hkls = \
        wavelength_allowed_hkls.astype(bool)
        return tth

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
        while (1.0 / self.CalcLength(
                np.array([self.ih, 0, 0], dtype=np.float64), 'r')
                > self.dmin):
            self.ih = self.ih + 1
        self.ik = 1
        while (1.0 / self.CalcLength(
                np.array([0, self.ik, 0], dtype=np.float64), 'r')
                > self.dmin):
            self.ik = self.ik + 1
        self.il = 1
        while (1.0 / self.CalcLength(
                np.array([0, 0, self.il], dtype=np.float64), 'r')
                > self.dmin):
            self.il = self.il + 1

    def CalcStar(self, v, space, applyLaue=False):
        """
        this function calculates the symmetrically equivalent hkls (or uvws)
        for the reciprocal (or direct) point group symmetry.
        """
        if(space == 'd'):
            if(applyLaue):
                sym = self.SYM_PG_d_laue
            else:
                sym = self.SYM_PG_d
        elif(space == 'r'):
            if(applyLaue):
                sym = self.SYM_PG_r_laue
            else:
                sym = self.SYM_PG_r
        else:
            raise ValueError('CalcStar: unrecognized space.')
        vsym = np.atleast_2d(v)
        for s in sym:
            vp = np.dot(s, v)
            # check if this is new
            isnew = True
            for vec in vsym:
                if(np.sum(np.abs(vp - vec)) < 1E-4):
                    isnew = False
                    break
            if(isnew):
                vsym = np.vstack((vsym, vp))
        return vsym

    def ChooseSymmetric(self, hkllist, InversionSymmetry=True):
        """
        this function takes a list of hkl vectors and
        picks out a subset of the list picking only one
        of the symmetrically equivalent one. The convention
        is to choose the hkl with the most positive components.
        """
        mask = np.ones(hkllist.shape[0], dtype=np.bool)
        laue = InversionSymmetry
        for i, g in enumerate(hkllist):
            if(mask[i]):
                geqv = self.CalcStar(g, 'r', applyLaue=laue)
                for r in geqv[1:, ]:
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
        # glen = np.atleast_2d(np.array(glen,dtype=np.float)).T
        dtype = [('glen', float), ('max', int), ('sum', int),
                 ('h', int), ('k', int), ('l', int)]
        a = []
        for i, gl in enumerate(glen):
            g = hkllist[i, :]
            a.append((gl, np.max(g), np.sum(g), g[0], g[1], g[2]))
        a = np.array(a, dtype=dtype)
        isort = np.argsort(a, order=['glen', 'max', 'sum', 'l', 'k', 'h'])
        return hkllist[isort, :]

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
        hmin = -self.ih-1
        hmax = self.ih
        kmin = -self.ik-1
        kmax = self.ik
        lmin = -1
        lmax = self.il
        hkllist = np.array([[ih, ik, il] for ih in np.arange(hmax, hmin, -1)
                            for ik in np.arange(kmax, kmin, -1)
                            for il in np.arange(lmax, lmin, -1)])
        hkl_allowed = Allowed_HKLs(self.sgnum, hkllist)
        hkl = []
        dsp = []
        hkl_dsp = []
        for g in hkl_allowed:
            # ignore [0 0 0] as it is the direct beam
            if(np.sum(np.abs(g)) != 0):
                dspace = 1./self.CalcLength(g, 'r')
                if(dspace >= dmin):
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
        self.hkl = self.SortHKL(hkl)
        return self.hkl

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
        if(len(val) != 15):
            msg = (f"incorrect shape for shkl. "
                f"shape should be (15, ).")
            raise ValueError(msg)

        self._shkl = val

class Phases_LeBail:
    """
    ========================================================================================
    ========================================================================================
    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/20/2020 SS 1.0 original
    >> @DETAILS:    class to handle different phases in the LeBail fit. this is a stripped down
                    version of main Phase class for efficiency. only the 
                    components necessary for calculating peak positions are retained. further 
                    this will have a slight modification to account for different wavelengths 
                    in the same phase name
    =========================================================================================
    =========================================================================================
    """
    def _kev(x):
        return valWUnit('beamenergy', 'energy', x, 'keV')

    def _nm(x):
        return valWUnit('lp', 'length', x, 'nm')

    def __init__(self, material_file=None,
                 material_keys=None,
                 dmin=_nm(0.05),
                 wavelength={'alpha1': [_nm(0.15406), 1.0],
                             'alpha2': [_nm(0.154443), 0.52]}
                 ):

        self.phase_dict = {}
        self.num_phases = 0

        """
        set wavelength. check if wavelength is supplied in A, if it is
        convert to nm since the rest of the code assumes those units
        """
        wavelength_nm = {}
        for k, v in wavelength.items():
            wavelength_nm[k] = [valWUnit('lp', 'length',
                                         v[0].getVal('nm'), 'nm'), v[1]]

        self.wavelength = wavelength_nm

        self.dmin = dmin

        if(material_file is not None):
            if(material_keys is not None):
                if(type(material_keys) is not list):
                    self.add(material_file, material_keys)
                else:
                    self.add_many(material_file, material_keys)

    def __str__(self):
        resstr = 'Phases in calculation:\n'
        for i, k in enumerate(self.phase_dict.keys()):
            resstr += '\t'+str(i+1)+'. '+k+'\n'
        return resstr

    def __getitem__(self, key):
        if(key in self.phase_dict.keys()):
            return self.phase_dict[key]
        else:
            raise ValueError('phase with name not found')

    def __setitem__(self, key, mat_cls):

        if(key in self.phase_dict.keys()):
            warnings.warn('phase already in parameter \
                list. overwriting ...')
        if(isinstance(mat_cls, Material_LeBail)):
            self.phase_dict[key] = mat_cls
        else:
            raise ValueError('input not a material class')

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if(self.n < len(self.phase_dict.keys())):
            res = list(self.phase_dict.keys())[self.n]
            self.n += 1
            return res
        else:
            raise StopIteration

    def __len__(self):
        return len(self.phase_dict)

    def add(self, material_file, material_key):

        self[material_key] = Material_LeBail(
            fhdf=material_file, xtal=material_key, dmin=self.dmin)

    def add_many(self, material_file, material_keys):

        for k in material_keys:

            self[k] = Material_LeBail(
                fhdf=material_file, xtal=k, dmin=self.dmin)

            self.num_phases += 1

        for k in self:
            self[k].pf = 1.0/len(self)

        self.material_file = material_file
        self.material_keys = material_keys

    def load(self, fname):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       06/08/2020 SS 1.0 original
            >> @DETAILS:    load parameters from yaml file
        """
        with open(fname) as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        for mfile in dic.keys():
            mat_keys = list(dic[mfile])
            self.add_many(mfile, mat_keys)

    def dump(self, fname):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       06/08/2020 SS 1.0 original
            >> @DETAILS:    dump parameters to yaml file
        """
        dic = {}
        k = self.material_file
        dic[k] = [m for m in self]

        with open(fname, 'w') as f:
            data = yaml.dump(dic, f, sort_keys=False)

    def dump_hdf5(self, file):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       01/15/2021 SS 1.0 original
        >> @ DETAILS    dumps the information from each material in the phase class
                        to a hdf5 file specified by filename or h5py.File object
        """
        if(isinstance(file, str)):
            fexist = path.isfile(file)
            if(fexist):
                fid = h5py.File(file, 'r+')
            else:
                fid = h5py.File(file, 'x')

        elif(isinstance(file, h5py.File)):
            fid = file

        else:
            raise RuntimeError(
                'Parameters: dump_hdf5 Pass in a filename \
                string or h5py.File object')

        if("/Phases" in fid):
            del(fid["Phases"])
        gid_top = fid.create_group("Phases")

        for p in self:
            mat = self[p]

            sgnum = mat.sgnum
            sgsetting = mat.sgsetting
            lparms = mat.lparms
            dmin = mat.dmin
            hkls = mat.hkls

            gid = gid_top.create_group(p)

            did = gid.create_dataset("SpaceGroupNumber", (1, ), dtype=np.int32)
            did.write_direct(np.array(sgnum, dtype=np.int32))

            did = gid.create_dataset(
                "SpaceGroupSetting", (1, ), dtype=np.int32)
            did.write_direct(np.array(sgsetting, dtype=np.int32))

            did = gid.create_dataset(
                "LatticeParameters", (6, ), dtype=np.float64)
            did.write_direct(np.array(lparms, dtype=np.float64))

            did = gid.create_dataset("dmin", (1, ), dtype=np.float64)
            did.attrs["units"] = "nm"
            did.write_direct(np.array(dmin, dtype=np.float64))

class Material_Rietveld:
    """
    ===========================================================================================
    ===========================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/18/2020 SS 1.0 original
                    02/01/2021 SS 1.1 class can now be initialized using a 
                    material.Material class instance
    >> @DETAILS:    Material_LeBail class is a stripped down version of the materials.Material
                    class.this is done to keep the class lightweight and make sure only the 
                    information necessary for the Rietveld fit is kept
    ===========================================================================================
     ==========================================================================================
    """

    def __init__(self,
                 fhdf=None,
                 xtal=None,
                 dmin=None,
                 kev=None,
                 material_obj=None):

        self._shkl = np.zeros((15,))
        self.abs_fact = 1e4
        if(material_obj is None):
            """
            dmin in nm
            """
            self.dmin = dmin.value
            """
            voltage in ev
            """
            self.voltage = kev.value * 1000.0

            self._readHDF(fhdf, xtal)
            self._calcrmt()

            if(self.aniU):
                self.calcBetaij()

            self.SYM_SG, self.SYM_PG_d, self.SYM_PG_d_laue, \
                self.centrosymmetric, self.symmorphic = \
                symmetry.GenerateSGSym(self.sgnum, self.sgsetting)
            self.latticeType = symmetry.latticeType(self.sgnum)
            self.sg_hmsymbol = symbols.pstr_spacegroup[self.sgnum-1].strip()
            self.GenerateRecipPGSym()
            self.CalcMaxGIndex()
            self._calchkls()
            self.InitializeInterpTable()
            self.CalcWavelength()
            self.CalcPositions()
            self.sg = SpaceGroup(self.sgnum)

        else:
            if(isinstance(material_obj, Material)):
                self._init_from_materials(material_obj)
            else:
                raise ValueError(
                    "Invalid material_obj argument. \
                    only Material class can be passed here.")

    def _init_from_materials(self, material_obj):
        """

        """
        # name
        self.name = material_obj.name

        # min d-spacing for sampling hkl
        self.dmin = material_obj.dmin

        # acceleration voltage and wavelength
        self.voltage = material_obj.unitcell.voltage
        self.wavelength = material_obj.unitcell.wavelength

        # space group number
        self.sgnum = material_obj.sgnum
        self.sg = SpaceGroup(self.sgnum)
        
        # space group setting
        self.sgsetting = material_obj.sgsetting

        # lattice type from sgnum
        self.latticeType = material_obj.unitcell.latticeType

        # Herman-Maugauin symbol
        self.sg_hmsymbol = material_obj.unitcell.sg_hmsymbol

        # lattice parameters
        self.lparms = np.array([material_obj.unitcell.a,
                                material_obj.unitcell.b,
                                material_obj.unitcell.c,
                                material_obj.unitcell.alpha,
                                material_obj.unitcell.beta,
                                material_obj.unitcell.gamma])

        # asymmetric atomic positions
        self.atom_pos = material_obj.unitcell.atom_pos

        # Debye-Waller factors including anisotropic ones
        self.U = material_obj.unitcell.U
        self.aniU = False
        if(self.U.ndim > 1):
            self.aniU = True
            self.betaij = material_obj.unitcell.betaij

        # atom types i.e. Z and number of different atom types
        self.atom_type = material_obj.unitcell.atom_type
        self.atom_ntype = material_obj.unitcell.atom_ntype

        self._calcrmt()

        """ get all space and point group symmetry operators
         in direct space, including the laue group. reciprocal
         space point group symmetries also included """
        self.SYM_SG = material_obj.unitcell.SYM_SG
        self.SYM_PG_d = material_obj.unitcell.SYM_PG_d
        self.SYM_PG_d_laue = material_obj.unitcell.SYM_PG_d_laue
        self.centrosymmetric = material_obj.unitcell.centrosymmetric
        self.symmorphic = material_obj.unitcell.symmorphic
        self.SYM_PG_r = material_obj.unitcell.SYM_PG_r
        self.SYM_PG_r_laue = material_obj.unitcell.SYM_PG_r_laue

        # get maximum indices for sampling hkl
        self.ih = material_obj.unitcell.ih
        self.ik = material_obj.unitcell.ik
        self.il = material_obj.unitcell.il

        # copy over the hkl but calculate the multiplicities
        self.hkls = material_obj.planeData.getHKLs()
        multiplicity = []
        for g in self.hkls:
            multiplicity.append(self.CalcStar(g, 'r').shape[0])

        multiplicity = np.array(multiplicity)
        self.multiplicity = multiplicity

        # interpolation tables and anomalous form factors
        self.f1 = material_obj.unitcell.f1
        self.f2 = material_obj.unitcell.f2
        self.f_anam = material_obj.unitcell.f_anam

        # final step is to calculate the asymmetric positions in
        # the unit cell
        self.numat = material_obj.unitcell.numat
        self.asym_pos = material_obj.unitcell.asym_pos
        self.InitializeInterpTable()

    def _readHDF(self, fhdf, xtal):

        # fexist = path.exists(fhdf)
        # if(fexist):
        fid = h5py.File(fhdf, 'r')
        name = xtal
        xtal = "/"+xtal
        if xtal not in fid:
            raise IOError('crystal doesn''t exist in material file.')
        # else:
        #   raise IOError('material file does not exist.')

        gid = fid.get(xtal)

        self.sgnum = np.asscalar(np.array(gid.get('SpaceGroupNumber'),
                                          dtype=np.int32))
        self.sgsetting = np.asscalar(np.array(gid.get('SpaceGroupSetting'),
                                              dtype=np.int32))
        """
            IMPORTANT NOTE:
            note that the latice parameters in EMsoft is nm by default
            hexrd on the other hand uses A as the default units, so we
            need to be careful and convert it right here, so there is no
            confusion later on
        """
        self.lparms = list(gid.get('LatticeParameters'))

        # the last field in this is already
        self.atom_pos = np.transpose(
            np.array(gid.get('AtomData'), dtype=np.float64))

        # the U factors are related to B by the relation B = 8pi^2 U
        self.U = np.transpose(np.array(gid.get('U'), dtype=np.float64))

        # read atom types (by atomic number, Z)
        self.atom_type = np.array(gid.get('Atomtypes'), dtype=np.int32)
        self.atom_ntype = self.atom_type.shape[0]
        self.name = name

        fid.close()

    def calcBetaij(self):

        self.betaij = np.zeros([3, 3, self.atom_ntype])
        for i in range(self.U.shape[0]):
            U = self.U[i, :]
            self.betaij[:, :, i] = np.array([[U[0], U[3], U[4]],
                                             [U[3], U[1], U[5]],
                                             [U[4], U[5], U[2]]])

            self.betaij[:, :, i] *= 2. * np.pi**2 * self.aij

    def CalcWavelength(self):
        # wavelength in nm
        self.wavelength = constants.cPlanck * \
            constants.cLight /  \
            constants.cCharge / \
            self.voltage
        self.wavelength *= 1e9
        self.CalcAnomalous()

    def CalcKeV(self):
        self.kev = constants.cPlanck * \
            constants.cLight /  \
            constants.cCharge / \
            self.wavelength

        self.kev *= 1e-3

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
        sa = np.sin(alpha)
        sb = np.sin(beta)
        sg = np.sin(gamma)
        tg = np.tan(gamma)

        """
            direct metric tensor
        """
        self.dmt = np.array([[a**2, a*b*cg, a*c*cb],
                             [a*b*cg, b**2, b*c*ca],
                             [a*c*cb, b*c*ca, c**2]])
        self.vol = np.sqrt(np.linalg.det(self.dmt))

        if(self.vol < 1e-5):
            warnings.warn('unitcell volume is suspiciously small')

        """
            reciprocal metric tensor
        """
        self.rmt = np.linalg.inv(self.dmt)

        """
            direct structure matrix
        """
        self.dsm = np.array([[a, b*cg, c*cb],
                              [0., b*sg, -c*(cb*cg - ca)/sg],
                              [0., 0., self.vol/(a*b*sg)]])
        """
            reciprocal structure matrix
        """
        self.rsm = np.array([[1./a, 0., 0.],
                              [-1./(a*tg), 1./(b*sg), 0.],
                              [b*c*(cg*ca - cb)/(self.vol*sg),
                               a*c*(cb*cg - ca)/(self.vol*sg),
                               a*b*sg/self.vol]])

        ast = self.CalcLength([1, 0, 0], 'r')
        bst = self.CalcLength([0, 1, 0], 'r')
        cst = self.CalcLength([0, 0, 1], 'r')

        self.aij = np.array([[ast**2, ast*bst, ast*cst],
                             [bst*ast, bst**2, bst*cst],
                             [cst*ast, cst*bst, cst**2]])

    def _calchkls(self):
        self.hkls, self.multiplicity = self.getHKLs(self.dmin)

    """ calculate dot product of two vectors in any space 'd' 'r' or 'c' """

    def CalcLength(self, u, space):

        if(space == 'd'):
            vlen = np.sqrt(np.dot(u, np.dot(self.dmt, u)))
        elif(space == 'r'):
            vlen = np.sqrt(np.dot(u, np.dot(self.rmt, u)))
        elif(spec == 'c'):
            vlen = np.linalg.norm(u)
        else:
            raise ValueError('incorrect space argument')

        return vlen

    def getTTh(self, wavelength):

        tth = []
        self.dsp = _calc_dspacing(self.rmt.astype(np.float64),
            self.hkls.astype(np.float64))
        tth, wavelength_allowed_hkls = \
        _get_tth(self.dsp, wavelength)
        self.wavelength_allowed_hkls = wavelength_allowed_hkls.astype(bool)
        return tth

    ''' transform between any crystal space to any other space.
        choices are 'd' (direct), 'r' (reciprocal) and 'c' (cartesian)'''
    def TransSpace(self, v_in, inspace, outspace):
        if(inspace == 'd'):
            if(outspace == 'r'):
                v_out = np.dot(v_in, self.dmt)
            elif(outspace == 'c'):
                v_out = np.dot(self.dsm, v_in)
            else:
                raise ValueError(
                    'inspace in ''d'' but outspace can''t be identified')
        elif(inspace == 'r'):
            if(outspace == 'd'):
                v_out = np.dot(v_in, self.rmt)
            elif(outspace == 'c'):
                v_out = np.dot(self.rsm, v_in)
            else:
                raise ValueError(
                    'inspace in ''r'' but outspace can''t be identified')
        elif(inspace == 'c'):
            if(outspace == 'r'):
                v_out = np.dot(v_in, self.rsm)
            elif(outspace == 'd'):
                v_out = np.dot(v_in, self.dsm)
            else:
                raise ValueError(
                    'inspace in ''c'' but outspace can''t be identified')
        else:
            raise ValueError('incorrect inspace argument')
        return v_out

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
        while (1.0 / self.CalcLength(
                np.array([self.ih, 0, 0], dtype=np.float64), 'r')
                > self.dmin):
            self.ih = self.ih + 1
        self.ik = 1
        while (1.0 / self.CalcLength(
                np.array([0, self.ik, 0], dtype=np.float64), 'r') >
                self.dmin):
            self.ik = self.ik + 1
        self.il = 1
        while (1.0 / self.CalcLength(
                np.array([0, 0, self.il], dtype=np.float64), 'r') >
                self.dmin):
            self.il = self.il + 1

    def CalcStar(self, v, space, applyLaue=False):
        """
        this function calculates the symmetrically equivalent hkls (or uvws)
        for the reciprocal (or direct) point group symmetry.
        """
        if(space == 'd'):
            if(applyLaue):
                sym = self.SYM_PG_d_laue
            else:
                sym = self.SYM_PG_d
        elif(space == 'r'):
            if(applyLaue):
                sym = self.SYM_PG_r_laue
            else:
                sym = self.SYM_PG_r
        else:
            raise ValueError('CalcStar: unrecognized space.')
        vsym = np.atleast_2d(v)
        for s in sym:
            vp = np.dot(s, v)
            # check if this is new
            isnew = True
            for vec in vsym:
                if(np.sum(np.abs(vp - vec)) < 1E-4):
                    isnew = False
                    break
            if(isnew):
                vsym = np.vstack((vsym, vp))
        return vsym

    def ChooseSymmetric(self, hkllist, InversionSymmetry=True):
        """
        this function takes a list of hkl vectors and
        picks out a subset of the list picking only one
        of the symmetrically equivalent one. The convention
        is to choose the hkl with the most positive components.
        """
        mask = np.ones(hkllist.shape[0], dtype=np.bool)
        laue = InversionSymmetry
        for i, g in enumerate(hkllist):
            if(mask[i]):
                geqv = self.CalcStar(g, 'r', applyLaue=laue)
                for r in geqv[1:, ]:
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
        # glen = np.atleast_2d(np.array(glen,dtype=np.float)).T
        dtype = [('glen', float), ('max', int), ('sum', int),
                 ('h', int), ('k', int), ('l', int)]
        a = []
        for i, gl in enumerate(glen):
            g = hkllist[i, :]
            a.append((gl, np.max(g), np.sum(g), g[0], g[1], g[2]))
        a = np.array(a, dtype=dtype)
        isort = np.argsort(a, order=['glen', 'max', 'sum', 'l', 'k', 'h'])
        return hkllist[isort, :]

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
        hmin = -self.ih-1
        hmax = self.ih
        kmin = -self.ik-1
        kmax = self.ik
        lmin = -1
        lmax = self.il
        hkllist = np.array([[ih, ik, il] for ih in np.arange(hmax, hmin, -1)
                            for ik in np.arange(kmax, kmin, -1)
                            for il in np.arange(lmax, lmin, -1)])
        hkl_allowed = Allowed_HKLs(self.sgnum, hkllist)
        hkl = []
        dsp = []
        hkl_dsp = []
        for g in hkl_allowed:
            # ignore [0 0 0] as it is the direct beam
            if(np.sum(np.abs(g)) != 0):
                dspace = 1./self.CalcLength(g, 'r')
                if(dspace >= dmin):
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
        hkls = self.SortHKL(hkl)

        multiplicity = []
        for g in hkls:
            multiplicity.append(self.CalcStar(g, 'r').shape[0])

        multiplicity = np.array(multiplicity)
        return hkls, multiplicity

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
            r = np.hstack((r, 1.))

            asym_pos.append(np.broadcast_to(r[0:3], [1, 3]))

            for symmat in self.SYM_SG:
                # get new position
                rnew = np.dot(symmat, r)

                # reduce to fundamental unitcell with fractional
                # coordinates between 0-1
                rr = rnew[0:3]
                rr = np.modf(rr)[0]
                rr[rr < 0.] += 1.
                rr[np.abs(rr) < 1.0E-6] = 0.

                # check if this is new
                isnew = True
                for j in range(n):
                    if(np.sum(np.abs(rr - asym_pos[i][j, :])) < 1E-4):
                        isnew = False
                        break

                # if its new add this to the list
                if(isnew):
                    asym_pos[i] = np.vstack((asym_pos[i], rr))
                    n += 1

            numat.append(n)

        self.numat = np.array(numat)
        self.asym_pos = asym_pos

    def InitializeInterpTable(self):

        f_anomalous_data = []
        data = importlib.resources.open_binary(hexrd.resources, 'Anomalous.h5')
        with h5py.File(data, 'r') as fid:
            for i in range(0, self.atom_ntype):

                Z = self.atom_type[i]
                elem = constants.ptableinverse[Z]
                gid = fid.get('/'+elem)
                data = np.array(gid.get('data'))
                data = data[:,[7,1,2]]
                f_anomalous_data.append(data)

        n = max([x.shape[0] for x in f_anomalous_data])
        self.f_anomalous_data = np.zeros([self.atom_ntype,n,3])
        self.f_anomalous_data_sizes = np.zeros([self.atom_ntype,],
            dtype=np.int32)

        for i in range(self.atom_ntype):
            nd = f_anomalous_data[i].shape[0]
            self.f_anomalous_data_sizes[i] = nd
            self.f_anomalous_data[i,:nd,:] = f_anomalous_data[i]

    def CalcXRSF(self, 
        wavelength, 
        w_int):
        """
        the 1E-2 is to convert to A^-2
        since the fitting is done in those units
        """
        fNT = np.zeros([self.atom_ntype,])
        frel = np.zeros([self.atom_ntype,])
        scatfac = np.zeros([self.atom_ntype,11])
        f_anomalous_data = self.f_anomalous_data

        aniU = self.aniU
        occ = self.atom_pos[:,3]
        if aniU:
            betaij = self.betaij
        else:
            betaij = self.U

        self.numat = np.zeros(self.atom_ntype,dtype=np.int32)
        for i in range(0, self.atom_ntype):
            self.numat[i] = self.asym_pos[i].shape[0]
            Z = self.atom_type[i]
            elem = constants.ptableinverse[Z]
            scatfac[i,:] = constants.scatfac[elem]
            frel[i] = constants.frel[elem]
            fNT[i] = constants.fNT[elem]

        self.asym_pos_arr = np.zeros([self.numat.max(),self.atom_ntype, 3])
        for i in range(0, self.atom_ntype):
            nn = self.numat[i]
            self.asym_pos_arr[:nn,i,:] = self.asym_pos[i]

        nref = self.hkls.shape[0]

        sf, sf_raw = _calcxrsf(self.hkls.astype(np.float64),
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
              self.f_anomalous_data_sizes)

        return sf, sf_raw

    def calc_extinction(self,
                        wavelength,
                        tth,
                        f_sqr,
                        shape_factor_K,
                        particle_size_D):

        hkls = self.hkls
        v_unitcell = self.vol

        extinction = _calc_extinction_factor(hkls,
                            tth,
                            v_unitcell*1e3,
                            wavelength,
                            f_sqr,
                            shape_factor_K,
                            particle_size_D)

        return extinction

    def calc_absorption(self,
                        tth,
                        phi,
                        wavelength):
        abs_fact = self.abs_fact
        absorption = _calc_absorption_factor(abs_fact,
                                             tth,
                                             phi,
                                             wavelength)

        return absorption

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
        if(len(val) != 15):
            msg = (f"incorrect shape for shkl. "
                f"shape should be (15, ).")
            raise ValueError(msg)

        self._shkl = val

class Phases_Rietveld:
    """
    ==============================================================================================
    ==============================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       05/20/2020 SS 1.0 original
    >> @DETAILS:    class to handle different phases in the LeBail fit. this is a stripped down
                    version of main Phase class for efficiency. only the components necessary for
                    calculating peak positions are retained. further this will have a slight
                    modification to account for different wavelengths in the same phase name
    ==============================================================================================
     =============================================================================================
    """
    def _kev(x):
        return valWUnit('beamenergy', 'energy', x, 'keV')

    def _nm(x):
        return valWUnit('lp', 'length', x, 'nm')

    def __init__(self, material_file=None,
                 material_keys=None,
                 dmin=_nm(0.05),
                 wavelength={'alpha1': [_nm(0.15406), 1.], 'alpha2': [
                     _nm(0.154443), 0.52]}
                 ):

        self.phase_dict = {}
        self.num_phases = 0

        """
        set wavelength. check if wavelength is supplied in A, if it is
        convert to nm since the rest of the code assumes those units
        """
        wavelength_nm = {}
        for k, v in wavelength.items():
            if(v[0].unit == 'angstrom'):
                wavelength_nm[k] = [
                    valWUnit('lp', 'length', v[0].getVal("nm"), 'nm'), v[1]]
            else:
                wavelength_nm[k] = v
        self.wavelength = wavelength_nm

        self.dmin = dmin

        if(material_file is not None):
            if(material_keys is not None):
                if(type(material_keys) is not list):
                    self.add(material_file, material_keys)
                else:
                    self.add_many(material_file, material_keys)

    def __str__(self):
        resstr = 'Phases in calculation:\n'
        for i, k in enumerate(self.phase_dict.keys()):
            resstr += '\t'+str(i+1)+'. '+k+'\n'
        return resstr

    def __getitem__(self, key):
        if(key in self.phase_dict.keys()):
            return self.phase_dict[key]
        else:
            raise ValueError('phase with name not found')

    def __setitem__(self, key, mat_cls):

        if(key in self.phase_dict.keys()):
            warnings.warn('phase already in parameter list. overwriting ...')
        # if(isinstance(mat_cls, Material_Rietveld)):
        self.phase_dict[key] = mat_cls
        # else:
        # raise ValueError('input not a material class')

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if(self.n < len(self.phase_dict.keys())):
            res = list(self.phase_dict.keys())[self.n]
            self.n += 1
            return res
        else:
            raise StopIteration

    def __len__(self):
        return len(self.phase_dict)

    def add(self, material_file, material_key):
        self[material_key] = {}
        self.num_phases += 1
        for l in self.wavelength:
            lam = self.wavelength[l][0].value * 1e-9
            E = constants.cPlanck * constants.cLight / constants.cCharge / lam
            E *= 1e-3
            kev = valWUnit('beamenergy', 'energy', E, 'keV')
            self[material_key][l] = Material_Rietveld(
                material_file, material_key, dmin=self.dmin, kev=kev)

        for k in self:
            for l in self.wavelength:
                self[k][l].pf = 1.0/self.num_phases

    def add_many(self, material_file, material_keys):

        for k in material_keys:
            self[k] = {}
            self.num_phases += 1
            for l in self.wavelength:
                lam = self.wavelength[l][0].value * 1e-9
                E = constants.cPlanck * constants.cLight / \
                    constants.cCharge / lam
                E *= 1e-3
                kev = valWUnit('beamenergy', 'energy', E, 'keV')
                self[k][l] = Material_Rietveld(
                    material_file, k, dmin=self.dmin, kev=kev)

        for k in self:
            for l in self.wavelength:
                self[k][l].pf = 1.0/self.num_phases

        self.material_file = material_file
        self.material_keys = material_keys

    def load(self, fname):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       06/08/2020 SS 1.0 original
            >> @DETAILS:    load parameters from yaml file
        """
        with open(fname) as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        for mfile in dic.keys():
            mat_keys = list(dic[mfile])
            self.add_many(mfile, mat_keys)

    def dump(self, fname):
        """
            >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
            >> @DATE:       06/08/2020 SS 1.0 original
            >> @DETAILS:    dump parameters to yaml file
        """
        dic = {}
        k = self.material_file
        dic[k] = [m for m in self]

        with open(fname, 'w') as f:
            data = yaml.dump(dic, f, sort_keys=False)

    @property
    def phase_fraction(self):
        pf = []
        for k in self:
            l = list(self.wavelength.keys())[0]
            pf.append(self[k][l].pf)
        return np.array(pf)
    
    @phase_fraction.setter
    def phase_fraction(self, val):
        msg = (f"phase_fraction setter: "
               f"number of phases does not match"
               f"size of input")

        if isinstance(val, list):
            if len(val) != len(self):
                raise ValueError(msg)
        elif isinstance(val, np.ndarray):
            if val.shape[0] != len(self):
                raise ValueError(msg)

        for ii,k in enumerate(self):
            for l in self.wavelength:
                self[k][l].pf = val[ii]