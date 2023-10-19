import os
import numpy as np

class JCPDS_extend():
    def __init__(self, filename=None):
        if filename is None:
            self.file = ' '
            self.name = ' '
            self.version = 0
            self.comments = ''
            self.symmetry = ''
            self.k0 = 0.
            self.k0p = 0.  # k0p at 298K
            self.dk0dt = 0.
            self.dk0pdt = 0.
            self.thermal_expansion = 0.  # alphat at 298K
            self.thermal_expansion_dt = 0.
            self.a0 = 0.
            self.b0 = 0.
            self.c0 = 0.
            self.alpha0 = 0.
            self.beta0 = 0.
            self.gamma0 = 0.
            self.v0 = 0.
        else:
            self.file = filename
            self.dk0dt = 0.
            self.dk0pdt = 0.
            self.thermal_expansion_dt = 0.
            self.b0 = 0.
            self.c0 = 0.
            self.alpha0 = 0.
            self.beta0 = 0.
            self.gamma0 = 0.
            self.v0 = 0.
            self.read_file(self.file)
        self.a = 0.
        self.b = 0.
        self.c = 0.
        self.alpha = 0.
        self.beta = 0.
        self.gamma = 0.

        self.version_status = ''

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
                thermal_expansion = 0.
            else:
                thermal_expansion = float(item[0])
            self.thermal_expansion = thermal_expansion

            self.version_status = 'new'

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
                    self.thermal_expansion = alphat

                if jlinespl[0] == 'DALPHATDT:':
                    dalphatdt = float(jlinespl[1])
                    self.thermal_expansion_dt = dalphatat

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
            #elif self.symmetry == 'triclinic':

            jcpdsfile.close()

            self.version_status = 'new'

        else:
            self.version_status = 'old'

        if self.version_status == 'new':
            self.a = self.a0
            self.b = self.b0
            self.c = self.c0
            self.alpha = self.alpha0
            self.beta = self.beta0
            self.gamma = self.gamma0
            self.v = self.v0

    def calc_pressure(self, volume=None, temperature=None):
        '''calculate the pressure given the volume
           and temperature using the third order 
           birch-murnaghan equation of state.
        '''
        if volume is None:
            return 0
        else:
            v0 = self.v0
            alpha0 = self.thermal_expansion
            alpha1 = self.thermal_expansion_dt

            beta0 = self.dk0dt
            beta1 = self.dk0pdt

            k0 = self.k0
            k0p = self.k0p
            if temperature is None:
                vt = v0
                kt = k0
                ktp = k0p
            else:
                delT = (temperature-298)
                delT2 = (temperature**2-298**2)
                vt = v0*np.exp(alpha0*delT
                        +0.5*alpha1*delT2)
                kt = k0 + beta0*delT
                ktp = k0p + beta1*delT
            f = 0.5*((vt/volume)**(2./3.) - 1)

            return 3.0*kt*f*(1 - 1.5*(4 - ktp)*f)*(1 + 2*f)**2.5
