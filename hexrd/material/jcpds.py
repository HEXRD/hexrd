import os
import numpy as np


class JCPDS_extend():
    def __init__(self, filename=None):
        self.a0 = 0
        self.b0 = 0
        self.c0 = 0
        self.alpha0 = 0
        self.beta0 = 0
        self.gamma0 = 0
        self.v0 = 0

        self.k0 = 100
        self.k0p = 5  # k0p at 298K

        self.dk0dt = 0
        self.dk0pdt = 0

        self.symmetry = ''
        self.alpha_t = 0  # alphat at 298K
        self.dalpha_t_dt = 0

        self.file = ' '
        self.name = ' '
        self.version = 0
        self.comments = ''

        if filename:
            self.file = filename
            self.read_file(self.file)
            self.update_v0()

    def read_file(self, file):
        """
        read a jcpds file
        """
        self.file = file
        # Construct base name = file without path and without extension
        name = os.path.splitext(os.path.basename(self.file))[0]
        self.name = name
#       line = '', nd=0
        version = 0.
        self.comments = []
        self.DiffLines = []

        version_status = ''

        inp = open(file, 'r').readlines()
#       my_list = [] # get all the text first and throw into my_list

        if inp[0][0] in ('2', '3', '4'):
            version = int(inp[0])  # JCPDS version number
            self.version = version
            header = inp[1]  # header
            self.comments = header

            item = str.split(inp[2])
            crystal_system = int(item[0])
            if crystal_system == 1:
                self.symmetry = 'cubic'
            elif crystal_system == 2:
                self.symmetry = 'hexagonal'
            elif crystal_system == 3:
                self.symmetry = 'tetragonal'
            elif crystal_system == 4:
                self.symmetry = 'orthorhombic'
            elif crystal_system == 5:
                self.symmetry = 'monoclinic'
            elif crystal_system == 6:
                self.symmetry = 'triclinic'
            elif crystal_system == 7:
                self.symmetry = 'manual'
            # 1 cubic, 2 hexagonal, 3 tetragonal, 4 orthorhombic
            # 5 monoclinic, 6 triclinic, 7 manual P, d-sp input

            k0 = float(item[1])
            k0p = float(item[2])
            self.k0 = k0
            self.k0p = k0p

            item = str.split(inp[3])  # line for unit-cell parameters

            if crystal_system == 1:  # cubic
                a = float(item[0])
                b = a
                c = a
                alpha = 90.
                beta = 90.
                gamma = 90.
            elif crystal_system == 7:  # P, d-sp input
                a = float(item[0])
                b = a
                c = a
                alpha = 90.
                beta = 90.
                gamma = 90.
            elif crystal_system == 2:  # hexagonal
                a = float(item[0])
                c = float(item[1])
                b = a
                alpha = 90.
                beta = 90.
                gamma = 120.
            elif crystal_system == 3:  # tetragonal
                a = float(item[0])
                c = float(item[1])
                b = a
                alpha = 90.
                beta = 90.
                gamma = 90.
            elif crystal_system == 4:  # orthorhombic
                a = float(item[0])
                b = float(item[1])
                c = float(item[2])
                alpha = 90.
                beta = 90.
                gamma = 90.
            elif crystal_system == 5:  # monoclinic
                a = float(item[0])
                b = float(item[1])
                c = float(item[2])
                beta = float(item[3])
                alpha = 90.
                gamma = 90.
            elif crystal_system == 6:  # triclinic
                a = float(item[0])
                b = float(item[1])
                c = float(item[2])
                alpha = float(item[3])
                beta = float(item[4])
                gamma = float(item[5])

            self.a0 = a
            self.b0 = b
            self.c0 = c
            self.alpha0 = alpha
            self.beta0 = beta
            self.gamma0 = gamma

            item = str.split(inp[4])

            if self.version == 3:
                alpha_t = 0.
            else:
                alpha_t = float(item[0])
            self.alpha_t = alpha_t

            version_status = 'new'

        elif 'VERSION' in inp[0]:
            jcpdsfile = open(file, 'r')
            while True:
                jcpdsline = jcpdsfile.readline()
                if jcpdsline == '':
                    break

                jlinespl = jcpdsline.split()

                if jlinespl[0] == 'VERSION:':
                    version = int(jlinespl[1])
                    self.version = version

                if jlinespl[0] == 'COMMENT:':
                    header = ' '.join(jlinespl[1:])
                    self.comments = header

                if jlinespl[0] == 'K0:':
                    k0 = float(jlinespl[1])
                    self.k0 = k0

                if jlinespl[0] == 'K0P:':
                    k0p = float(jlinespl[1])
                    self.k0p = k0p

                if jlinespl[0] == 'DK0DT:':
                    dk0dt = float(jlinespl[1])
                    self.dk0dt = dk0dt

                if jlinespl[0] == 'DK0PDT:':
                    dk0pdt = float(jlinespl[1])
                    self.dk0pdt = dk0pdt

                if jlinespl[0] == 'SYMMETRY:':
                    self.symmetry = jlinespl[1].lower()

                if jlinespl[0] == 'A:':
                    a = float(jlinespl[1])
                    self.a0 = a

                if jlinespl[0] == 'B:':
                    b = float(jlinespl[1])
                    self.b0 = b

                if jlinespl[0] == 'C:':
                    c = float(jlinespl[1])
                    self.c0 = c

                if jlinespl[0] == 'ALPHA:':
                    alpha = float(jlinespl[1])
                    self.alpha0 = alpha

                if jlinespl[0] == 'BETA:':
                    beta = float(jlinespl[1])
                    self.beta0 = beta

                if jlinespl[0] == 'GAMMA:':
                    gamma = float(jlinespl[1])
                    self.gamma0 = gamma

                if jlinespl[0] == 'VOLUME:':
                    v = float(jlinespl[1])
                    self.v0 = v

                if jlinespl[0] == 'ALPHAT:':
                    alphat = float(jlinespl[1])
                    self.alpha_t = alphat

                if jlinespl[0] == 'DALPHATDT:':
                    dalphatdt = float(jlinespl[1])
                    self.dalpha_t_dt = dalphatdt

                if jlinespl[0] == 'DIHKL:':
                    pass

            if self.symmetry == 'cubic':
                self.b0 = self.a0
                self.c0 = self.a0
                self.alpha0 = 90.
                self.beta0 = 90.
                self.gamma0 = 90.
            elif self.symmetry == 'manual':
                self.b0 = self.a0
                self.c0 = self.a0
                self.alpha0 = 90.
                self.beta0 = 90.
                self.gamma0 = 90.
            elif self.symmetry == 'hexagonal' or self.symmetry == 'trigonal':
                self.b0 = self.a0
                self.alpha0 = 90.
                self.beta0 = 90.
                self.gamma0 = 120.
            elif self.symmetry == 'tetragonal':
                self.b0 = self.a0
                self.alpha0 = 90.
                self.beta0 = 90.
                self.gamma0 = 90.
            elif self.symmetry == 'orthorhombic':
                self.alpha0 = 90.
                self.beta0 = 90.
                self.gamma0 = 90.
            elif self.symmetry == 'monoclinic':
                self.alpha0 = 90.
                self.gamma0 = 90.
            # elif self.symmetry == 'triclinic':

            jcpdsfile.close()

            version_status = 'new'

        else:
            version_status = 'old'

        if version_status == 'new':
            self.v = self.calc_volume_unitcell()

    def verify_symmetry_match(self, mat):
        if not self.symmetry_matches(mat):
            msg = (
                f'The JCPDS symmetry "{self.symmetry}" does not match the '
                f'symmetry of the material "{mat.latticeType}"!'
            )
            raise SymmetryMismatch(msg)

    def symmetry_matches(self, mat):
        return mat.latticeType == self.symmetry

    @property
    def _lat_param_names(self):
        return ['a0', 'b0', 'c0', 'alpha0', 'beta0', 'gamma0']

    @property
    def lattice_params(self):
        return [getattr(self, x) for x in self._lat_param_names]

    def matches_material(self, mat):
        self.verify_symmetry_match(mat)
        mat_lp = [x.value for x in mat.latticeParameters]
        for v1, v2 in zip(mat_lp, self.lattice_params):
            if not np.isclose(v1, v2):
                return False

    def write_lattice_params_to_material(self, mat):
        self.verify_symmetry_match(mat)
        mat.latticeParameters0 = self.lattice_params

    def write_pt_params_to_material(self, mat):
        self.verify_symmetry_match(mat)
        mat.k0 = self.k0
        mat.k0p = self.k0p
        mat.dk0dt = self.dk0dt
        mat.dk0pdt = self.dk0pdt
        mat.alpha_t = self.alpha_t
        mat.dalpha_t_dt = self.dalpha_t_dt

    def update_v0(self):
        self.v0 = self.calc_volume_unitcell()

    def calc_volume_unitcell(self):
        '''calculate volume of the unitcell
        Ref: Introduction to conventional TEM
        Marc De Graef, Appendix 1 pg. 662
        '''
        ca = np.cos(np.radians(self.alpha0))
        cb = np.cos(np.radians(self.beta0))
        cg = np.cos(np.radians(self.gamma0))

        v0 = self.a0*self.b0*self.c0
        f = np.sqrt(1 - ca**2 - cb**2 - cg**2
                    + 2 * ca * cb * cg)
        return v0*f


class SymmetryMismatch(Exception):
    pass
