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
import numpy as np
from scipy import constants as sc

# pi related
pi = np.pi
piby2 = 0.5 * pi
piby3 = pi / 3.
piby4 = 0.25 * pi
piby6 = pi / 6.

# misc radicals
sqrt2 = np.sqrt(2.)
sqrt3 = np.sqrt(3.)
sqrt3by2 = 0.5 * sqrt3

# tolerancing
epsf = np.finfo(float).eps      # ~2.2e-16
ten_epsf = 10 * epsf            # ~2.2e-15
sqrt_epsf = np.sqrt(epsf)       # ~1.5e-8

# for angles
period_dict = {'degrees': 360.0, 'radians': 2*pi}
angular_units = 'radians'  # module-level angle units
d2r = pi / 180.
r2d = 180. / pi

# identity arrays
identity_3x3 = np.eye(3)  # (3, 3) identity
identity_6x1 = np.r_[1., 1., 1., 0., 0., 0.]

# basis vectors
lab_x = np.r_[1., 0., 0.]  # X in the lab frame
lab_y = np.r_[0., 1., 0.]  # Y in the lab frame
lab_z = np.r_[0., 0., 1.]  # Z in the lab frame

zeros_3 = np.zeros(3)
zeros_3x1 = np.zeros((3, 1))
zeros_6x1 = np.zeros((6, 1))

# reference beam direction and eta=0 ref in LAB FRAME for standard geometry
beam_vec = -lab_z
eta_vec = lab_x

# for energy/wavelength conversions
def keVToAngstrom(x):
    return (1e7*sc.c*sc.h/sc.e) / np.array(x, dtype=float)


def _readenv(name, ctor, default):
    try:
        import os
        res = os.environ[name]
        del os
    except KeyError:
        del os
        return default
    else:
        try:
            return ctor(res)
        except:
            import warnings
            warnings.warn("environ %s defined but failed to parse '%s'" %
                          (name, res), RuntimeWarning)
            del warnings
            return default


# 0 = do NOT use numba
# 1 = use numba (default)
USE_NUMBA = _readenv("HEXRD_USE_NUMBA", int, 1)
if USE_NUMBA:
    try:
        import numba
    except ImportError:
        print("*** Numba not available, processing may run slower ***")
        USE_NUMBA = False

del _readenv


# some physical constants
cAvogadro      = 6.02214076E23
cBoltzmann     = 1.380649E-23
cCharge        = 1.602176634E-19
cJ2eV          = 1.602176565E-19
cLight         = 299792458.0
cMoment        = 9.2740100707E-24
cPermea        = 1.2566370616E-6
cPermit        = 8.8541878163E-12
cPlanck        = 6.62607015E-34
cRestmass      = 9.1093837090E-31

'''
scattering factor for all atoms
Doyle-turner scattering factor coefficients
'''
scatfac = np.reshape( np.array([
        0.202,0.244,0.082,0.000,0.30868,0.08544,0.01273,0.00000, 
        0.091,0.181,0.110,0.036,0.18183,0.06212,0.01803,0.00284, 
        1.611,1.246,0.326,0.099,1.07638,0.30480,0.04533,0.00495, 
        1.250,1.334,0.360,0.106,0.60804,0.18591,0.03653,0.00416, 
        0.945,1.312,0.419,0.116,0.46444,0.14178,0.03223,0.00377, 
        0.731,1.195,0.456,0.125,0.36995,0.11297,0.02814,0.00346, 
        0.572,1.043,0.465,0.131,0.28847,0.09054,0.02421,0.00317, 
        0.455,0.917,0.472,0.138,0.23780,0.07622,0.02144,0.00296, 
        0.387,0.811,0.475,0.146,0.20239,0.06609,0.01931,0.00279, 
        0.303,0.720,0.475,0.153,0.17640,0.05860,0.01762,0.00266, 
        2.241,1.333,0.907,0.286,1.08004,0.24505,0.03391,0.00435, 
        2.268,1.803,0.839,0.289,0.73670,0.20175,0.03013,0.00405, 
        2.276,2.428,0.858,0.317,0.72322,0.19773,0.03080,0.00408, 
        2.129,2.533,0.835,0.322,0.57775,0.16476,0.02880,0.00386, 
        1.888,2.469,0.805,0.320,0.44876,0.13538,0.02642,0.00361, 
        1.659,2.386,0.790,0.321,0.36650,0.11488,0.02469,0.00340, 
        1.452,2.292,0.787,0.322,0.30935,0.09980,0.02234,0.00323, 
        1.274,2.190,0.793,0.326,0.26682,0.08813,0.02219,0.00307, 
        3.951,2.545,1.980,0.482,1.37075,0.22402,0.04532,0.00434, 
        4.470,2.971,1.970,0.482,0.99523,0.22696,0.04195,0.00417, 
        3.966,2.917,1.925,0.480,0.88960,0.20606,0.03856,0.00399, 
        3.565,2.818,1.893,0.483,0.81982,0.19049,0.03590,0.00386, 
        3.245,2.698,1.860,0.486,0.76379,0.17726,0.03363,0.00374, 
        2.307,2.334,1.823,0.490,0.78405,0.15785,0.03157,0.00364, 
        2.747,2.456,1.792,0.498,0.67786,0.15674,0.03000,0.00357, 
        2.544,2.343,1.759,0.506,0.64424,0.14880,0.02854,0.00350, 
        2.367,2.236,1.724,0.515,0.61431,0.14180,0.02725,0.00344, 
        2.210,2.134,1.689,0.524,0.58727,0.13553,0.02609,0.00339, 
        1.579,1.820,1.658,0.532,0.62940,0.12453,0.02504,0.00333, 
        1.942,1.950,1.619,0.543,0.54162,0.12518,0.02416,0.00330, 
        2.321,2.486,1.688,0.599,0.65602,0.15458,0.02581,0.00351, 
        2.447,2.702,1.616,0.601,0.55893,0.14393,0.02446,0.00342, 
        2.399,2.790,1.529,0.594,0.45718,0.12817,0.02280,0.00328, 
        2.298,2.854,1.456,0.590,0.38830,0.11536,0.02146,0.00316, 
        2.166,2.904,1.395,0.589,0.33899,0.10497,0.02041,0.00307, 
        2.034,2.927,1.342,0.589,0.29999,0.09598,0.01952,0.00299, 
        4.776,3.859,2.234,0.868,1.40782,0.18991,0.03701,0.00419, 
        5.848,4.003,2.342,0.880,1.04972,0.19367,0.03737,0.00414, 
        4.129,3.012,1.179,0.000,0.27548,0.05088,0.00591,0.000,   
        4.105,3.144,1.229,0.000,0.28492,0.05277,0.00601,0.000,   
        4.237,3.105,1.234,0.000,0.27415,0.05074,0.00593,0.000,   
        3.120,3.906,2.361,0.850,0.72464,0.14642,0.03237,0.00366, 
        4.318,3.270,1.287,0.000,0.28246,0.05148,0.00590,0.000,   
        4.358,3.298,1.323,0.000,0.27881,0.05179,0.00594,0.000,   
        4.431,3.343,1.345,0.000,0.27911,0.05153,0.00592,0.000,   
        4.436,3.454,1.383,0.000,0.28670,0.05269,0.00595,0.000,   
        2.036,3.272,2.511,0.837,0.61497,0.11824,0.02846,0.00327, 
        2.574,3.259,2.547,0.838,0.55675,0.11838,0.02784,0.00322, 
        3.153,3.557,2.818,0.884,0.66649,0.14449,0.02976,0.00335, 
        3.450,3.735,2.118,0.877,0.59104,0.14179,0.02855,0.00327, 
        3.564,3.844,2.687,0.864,0.50487,0.13316,0.02691,0.00316, 
        4.785,3.688,1.500,0.000,0.27999,0.05083,0.00581,0.000,   
        3.473,4.060,2.522,0.840,0.39441,0.11816,0.02415,0.00298, 
        3.366,4.147,2.443,0.829,0.35509,0.11117,0.02294,0.00289, 
        6.062,5.986,3.303,1.096,1.55837,0.19695,0.03335,0.00379, 
        7.821,6.004,3.280,1.103,1.17657,0.18778,0.03263,0.00376, 
        4.940,3.968,1.663,0.000,0.28716,0.05245,0.00594,0.000,   
        5.007,3.980,1.678,0.000,0.28283,0.05183,0.00589,0.000,   
        5.085,4.043,1.684,0.000,0.28588,0.05143,0.00581,0.000,   
        5.151,4.075,1.683,0.000,0.28304,0.05073,0.00571,0.000,   
        5.201,4.094,1.719,0.000,0.28079,0.05081,0.00576,0.000,   
        5.255,4.113,1.743,0.000,0.28016,0.05037,0.00577,0.000,   
        6.267,4.844,3.202,1.200,1.00298,0.16066,0.02980,0.00367, 
        5.225,4.314,1.827,0.000,0.29158,0.05259,0.00586,0.000,   
        5.272,4.347,1.844,0.000,0.29046,0.05226,0.00585,0.000,   
        5.332,4.370,1.863,0.000,0.28888,0.05198,0.00581,0.000,   
        5.376,4.403,1.884,0.000,0.28773,0.05174,0.00582,0.000,   
        5.436,4.437,1.891,0.000,0.28655,0.05117,0.00577,0.000,   
        5.441,4.510,1.956,0.000,0.29149,0.05264,0.00590,0.000,   
        5.529,4.533,1.945,0.000,0.28927,0.05144,0.00578,0.000,   
        5.553,4.580,1.969,0.000,0.28907,0.05160,0.00577,0.000,   
        5.588,4.619,1.997,0.000,0.29001,0.05164,0.00579,0.000,   
        5.659,4.630,2.014,0.000,0.28807,0.05114,0.00578,0.000,   
        5.709,4.677,2.019,0.000,0.28782,0.05084,0.00572,0.000,   
        5.695,4.740,2.064,0.000,0.28968,0.05156,0.00575,0.000,   
        5.750,4.773,2.079,0.000,0.28933,0.05139,0.00573,0.000,   
        5.754,4.851,2.096,0.000,0.29159,0.05152,0.00570,0.000,   
        5.803,4.870,2.127,0.000,0.29016,0.05150,0.00572,0.000,   
        2.388,4.226,2.689,1.255,0.42866,0.09743,0.02264,0.00307, 
        2.682,4.241,2.755,1.270,0.42822,0.09856,0.02295,0.00307, 
        5.932,4.972,2.195,0.000,0.29086,0.05126,0.00572,0.000,   
        3.510,4.552,3.154,1.359,0.52914,0.11884,0.02571,0.00321, 
        3.841,4.679,3.192,1.363,0.50261,0.11999,0.02560,0.00318, 
        6.070,4.997,2.232,0.000,0.28075,0.04999,0.00563,0.000,   
        6.133,5.031,2.239,0.000,0.28047,0.04957,0.00558,0.000,   
        4.078,4.978,3.096,1.326,0.38406,0.11020,0.02355,0.00299, 
        6.201,5.121,2.275,0.000,0.28200,0.04954,0.00556,0.000,   
        6.215,5.170,2.316,0.000,0.28382,0.05002,0.00562,0.000,   
        6.278,5.195,2.321,0.000,0.28323,0.04949,0.00557,0.000,   
        6.264,5.263,2.367,0.000,0.28651,0.05030,0.00563,0.000,   
        6.306,5.303,2.386,0.000,0.28688,0.05026,0.00561,0.000,   
        6.767,6.729,4.014,1.561,0.85951,0.15642,0.02936,0.00335, 
        6.323,5.414,2.453,0.000,0.29142,0.05096,0.00568,0.000,   
        6.415,5.419,2.449,0.000,0.28836,0.05022,0.00561,0.000,   
        6.378,5.495,2.495,0.000,0.29156,0.05102,0.00565,0.000,   
        6.460,5.469,2.471,0.000,0.28396,0.04970,0.00554,0.000,   
        6.502,5.478,2.510,0.000,0.28375,0.04975,0.00561,0.000,   
        6.548,5.526,2.520,0.000,0.28461,0.04965,0.00557,0.000]), [8,98])

'''
atomic weights for things like density computations 
(from NIST elemental data base)
'''
atom_weights = np.array([1.00794, 4.002602, 6.941, 9.012182, 10.811,
12.0107, 14.0067, 15.9994, 18.9984032, 20.1797,
22.98976928, 24.3050, 26.9815386, 28.0855, 30.973762,
32.065, 35.453, 39.948, 39.0983, 40.078,
44.955912, 47.867, 50.9415, 51.9961, 54.938045,
55.845, 58.933195, 58.6934, 63.546, 65.38,
69.723, 72.64, 74.92160, 78.96, 79.904,
83.798, 85.4678, 87.62, 88.90585, 91.224,
92.90638, 95.96, 98.9062, 101.07, 102.90550,
106.42, 107.8682, 112.411, 114.818, 118.710,
121.760, 127.60, 126.90447, 131.293, 132.9054519,
137.327, 138.90547, 140.116, 140.90765, 144.242,
145.0, 150.36, 151.964, 157.25, 158.92535,
162.500, 164.93032, 167.259, 168.93421, 173.054,
174.9668, 178.49, 180.94788, 183.84, 186.207,
190.23, 192.217, 195.084, 196.966569, 200.59,
204.3833, 207.2, 208.98040, 209.0, 210.0,
222.0, 223.0, 226.0, 227.0, 232.03806,
231.03588, 238.02891, 237.0, 244.0, 243.0,
247.0, 251.0, 252.0 ])

''' this variable encodes all the generators (including translations) for all 230 space groups
    will be used to compute the full space group symmetry operators 
'''
SYM_GL= [
"000                                     ","100                                     ","01cOOO0                                 ",\
"01cODO0                                 ","02aDDOcOOO0                             ","01jOOO0                                 ",\
"01jOOD0                                 ","02aDDOjOOO0                             ","02aDDOjOOD0                             ",\
"11cOOO0                                 ","11cODO0                                 ","12aDDOcOOO0                             ",\
"11cOOD0                                 ","11cODD0                                 ","12aDDOcOOD0                             ",\
"02bOOOcOOO0                             ","02bOODcOOD0                             ","02bOOOcDDO0                             ",\
"02bDODcODD0                             ","03aDDObOODcOOD0                         ","03aDDObOOOcOOO0                         ",\
"04aODDaDODbOOOcOOO0                     ","03aDDDbOOOcOOO0                         ","03aDDDbDODcODD0                         ",\
"02bOOOjOOO0                             ","02bOODjOOD0                             ","02bOOOjOOD0                             ",\
"02bOOOjDOO0                             ","02bOODjDOO0                             ","02bOOOjODD0                             ",\
"02bDODjDOD0                             ","02bOOOjDDO0                             ","02bOODjDDO0                             ",\
"02bOOOjDDD0                             ","03aDDObOOOjOOO0                         ","03aDDObOODjOOD0                         ",\
"03aDDObOOOjOOD0                         ","03aODDbOOOjOOO0                         ","03aODDbOOOjODO0                         ",\
"03aODDbOOOjDOO0                         ","03aODDbOOOjDDO0                         ","04aODDaDODbOOOjOOO0                     ",\
"04aODDaDODbOOOjBBB0                     ","03aDDDbOOOjOOO0                         ","03aDDDbOOOjDDO0                         ",\
"03aDDDbOOOjDOO0                         ","12bOOOcOOO0                             ","03bOOOcOOOhDDD1BBB                      ",\
"12bOOOcOOD0                             ","03bOOOcOOOhDDO1BBO                      ","12bDOOcOOO0                             ",\
"12bDOOcDDD0                             ","12bDODcDOD0                             ","12bDOOcOOD0                             ",\
"12bOOOcDDO0                             ","12bDDOcODD0                             ","12bOODcODD0                             ",\
"12bOOOcDDD0                             ","03bOOOcDDOhDDO1BBO                      ","12bDDDcOOD0                             ",\
"12bDODcODD0                             ","12bDODcODO0                             ","13aDDObOODcOOD0                         ",\
"13aDDObODDcODD0                         ","13aDDObOOOcOOO0                         ","13aDDObOOOcOOD0                         ",\
"13aDDObODOcODO0                         ","04aDDObDDOcOOOhODD1OBB                  ","14aODDaDODbOOOcOOO0                     ",\
"05aODDaDODbOOOcOOOhBBB1ZZZ              ","13aDDDbOOOcOOO0                         ","13aDDDbOOOcDDO0                         ",\
"13aDDDbDODcODD0                         ","13aDDDbODOcODO0                         ","02bOOOgOOO0                             ",\
"02bOODgOOB0                             ","02bOOOgOOD0                             ","02bOODgOOF0                             ",\
"03aDDDbOOOgOOO0                         ","03aDDDbDDDgODB0                         ","02bOOOmOOO0                             ",\
"03aDDDbOOOmOOO0                         ","12bOOOgOOO0                             ","12bOOOgOOD0                             ",\
"03bOOOgDDOhDDO1YBO                      ","03bOOOgDDDhDDD1YYY                      ","13aDDDbOOOgOOO0                         ",\
"04aDDDbDDDgODBhODB1OYZ                  ","03bOOOgOOOcOOO0                         ","03bOOOgDDOcDDO0                         ",\
"03bOODgOOBcOOO0                         ","03bOODgDDBcDDB0                         ","03bOOOgOODcOOO0                         ",\
"03bOOOgDDDcDDD0                         ","03bOODgOOFcOOO0                         ","03bOODgDDFcDDF0                         ",\
"04aDDDbOOOgOOOcOOO0                     ","04aDDDbDDDgODBcDOF0                     ","03bOOOgOOOjOOO0                         ",\
"03bOOOgOOOjDDO0                         ","03bOOOgOODjOOD0                         ","03bOOOgDDDjDDD0                         ",\
"03bOOOgOOOjOOD0                         ","03bOOOgOOOjDDD0                         ","03bOOOgOODjOOO0                         ",\
"03bOOOgOODjDDO0                         ","04aDDDbOOOgOOOjOOO0                     ","04aDDDbOOOgOOOjOOD0                     ",\
"04aDDDbDDDgODBjOOO0                     ","04aDDDbDDDgODBjOOD0                     ","03bOOOmOOOcOOO0                         ",\
"03bOOOmOOOcOOD0                         ","03bOOOmOOOcDDO0                         ","03bOOOmOOOcDDD0                         ",\
"03bOOOmOOOjOOO0                         ","03bOOOmOOOjOOD0                         ","03bOOOmOOOjDDO0                         ",\
"03bOOOmOOOjDDD0                         ","04aDDDbOOOmOOOjOOO0                     ","04aDDDbOOOmOOOjOOD0                     ",\
"04aDDDbOOOmOOOcOOO0                     ","04aDDDbOOOmOOOcDOF0                     ","13bOOOgOOOcOOO0                         ",\
"13bOOOgOOOcOOD0                         ","04bOOOgOOOcOOOhDDO1YYO                  ","04bOOOgOOOcOOOhDDD1YYY                  ",\
"13bOOOgOOOcDDO0                         ","13bOOOgOOOcDDD0                         ","04bOOOgDDOcDDOhDDO1YBO                  ",\
"04bOOOgDDOcDDDhDDO1YBO                  ","13bOOOgOODcOOO0                         ","13bOOOgOODcOOD0                         ",\
"04bOOOgDDDcOODhDDD1YBY                  ","04bOOOgDDDcOOOhDDD1YBY                  ","13bOOOgOODcDDO0                         ",\
"13bOOOgDDDcDDD0                         ","04bOOOgDDDcDDDhDDD1YBY                  ","04bOOOgDDDcDDOhDDD1YBY                  ",\
"14aDDDbOOOgOOOcOOO0                     ","14aDDDbOOOgOOOcOOD0                     ","05aDDDbDDDgODBcDOFhODB1OBZ              ",\
"05aDDDbDDDgODBcDOBhODB1OBZ              ","01nOOO0                                 ","01nOOC0                                 ",\
"01nOOE0                                 ","02aECCnOOO0                             ","11nOOO0                                 ",\
"12aECCnOOO0                             ","02nOOOfOOO0                             ","02nOOOeOOO0                             ",\
"02nOOCfOOE0                             ","02nOOCeOOO0                             ","02nOOEfOOC0                             ",\
"02nOOEeOOO0                             ","03aECCnOOOeOOO0                         ","02nOOOkOOO0                             ",\
"02nOOOlOOO0                             ","02nOOOkOOD0                             ","02nOOOlOOD0                             ",\
"03aECCnOOOkOOO0                         ","03aECCnOOOkOOD0                         ","12nOOOfOOO0                             ",\
"12nOOOfOOD0                             ","12nOOOeOOO0                             ","12nOOOeOOD0                             ",\
"13aECCnOOOeOOO0                         ","13aECCnOOOeOOD0                         ","02nOOObOOO0                             ",\
"02nOOCbOOD0                             ","02nOOEbOOD0                             ","02nOOEbOOO0                             ",\
"02nOOCbOOO0                             ","02nOOObOOD0                             ","02nOOOiOOO0                             ",\
"12nOOObOOO0                             ","12nOOObOOD0                             ","03nOOObOOOeOOO0                         ",\
"03nOOCbOODeOOC0                         ","03nOOEbOODeOOE0                         ","03nOOEbOOOeOOE0                         ",\
"03nOOCbOOOeOOC0                         ","03nOOObOODeOOO0                         ","03nOOObOOOkOOO0                         ",\
"03nOOObOOOkOOD0                         ","03nOOObOODkOOD0                         ","03nOOObOODkOOO0                         ",\
"03nOOOiOOOkOOO0                         ","03nOOOiOODkOOD0                         ","03nOOOiOOOeOOO0                         ",\
"03nOOOiOODeOOO0                         ","13nOOObOOOeOOO0                         ","13nOOObOOOeOOD0                         ",\
"13nOOObOODeOOD0                         ","13nOOObOODeOOO0                         ","03bOOOcOOOdOOO0                         ",\
"05aODDaDODbOOOcOOOdOOO0                 ","04aDDDbOOOcOOOdOOO0                     ","03bDODcODDdOOO0                         ",\
"04aDDDbDODcODDdOOO0                     ","13bOOOcOOOdOOO0                         ","04bOOOcOOOdOOOhDDD1YYY                  ",\
"15aODDaDODbOOOcOOOdOOO0                 ","06aODDaDODbOOOcOOOdOOOhBBB1ZZZ          ","14aDDDbOOOcOOOdOOO0                     ",\
"13bDODcODDdOOO0                         ","14aDDDbDODcODDdOOO0                     ","04bOOOcOOOdOOOeOOO0                     ",\
"04bOOOcOOOdOOOeDDD0                     ","06aODDaDODbOOOcOOOdOOOeOOO0             ","06aODDaDODbODDcDDOdOOOeFBF0             ",\
"05aDDDbOOOcOOOdOOOeOOO0                 ","04bDODcODDdOOOeBFF0                     ","04bDODcODDdOOOeFBB0                     ",\
"05aDDDbDODcODDdOOOeFBB0                 ","04bOOOcOOOdOOOlOOO0                     ","06aODDaDODbOOOcOOOdOOOlOOO0             ",\
"05aDDDbOOOcOOOdOOOlOOO0                 ","04bOOOcOOOdOOOlDDD0                     ","06aODDaDODbOOOcOOOdOOOlDDD0             ",\
"05aDDDbDODcODDdOOOlBBB0                 ","14bOOOcOOOdOOOeOOO0                     ","05bOOOcOOOdOOOeOOOhDDD1YYY              ",\
"14bOOOcOOOdOOOeDDD0                     ","05bOOOcOOOdOOOeDDDhDDD1YYY              ","16aODDaDODbOOOcOOOdOOOeOOO0             ",\
"16aODDaDODbOOOcOOOdOOOeDDD0             ","07aODDaDODbODDcDDOdOOOeFBFhBBB1ZZZ      ","07aODDaDODbODDcDDOdOOOeFBFhFFF1XXX      ",\
"15aDDDbOOOcOOOdOOOeOOO0                 ","15aDDDbDODcODDdOOOeFBB0                 ","01dOOO0                                 ",\
"11dOOO0                                 ","02dOOOfOOO0                             ","02dOOOlOOO0                             ",\
"02dOOOlDDD0                             ","12dOOOfOOO0                             ","12dOOOfDDD0                             "]

'''
this dictionary contains the generators encoded in each letter of the generator string
the full symmetry is generated by the repeated action of the generator matrix
'''

''' rotational, inversions, mirrors etc. components
'''

SYM_GENERATORS = {}

# now start to fill them in
# identity 
SYM_GENERATORS['a'] = np.zeros([3,3])
SYM_GENERATORS['a'] = np.eye(3)

# 180@c
SYM_GENERATORS['b'] = np.zeros([3,3])
SYM_GENERATORS['b'][0,0] = -1.; SYM_GENERATORS['b'][1,1] = -1.; SYM_GENERATORS['b'][2,2] = 1.

# 180@b
SYM_GENERATORS['c'] = np.zeros([3,3])
SYM_GENERATORS['c'][0,0] = -1.; SYM_GENERATORS['c'][1,1] = 1.; SYM_GENERATORS['c'][2,2] = -1.

# 120@[111]
SYM_GENERATORS['d'] = np.zeros([3,3]) 
SYM_GENERATORS['d'][0,2] = 1.; SYM_GENERATORS['d'][1,0] = 1.; SYM_GENERATORS['d'][2,1] = 1.

#180@[110]
SYM_GENERATORS['e'] = np.zeros([3,3]) 
SYM_GENERATORS['e'][0,1] = 1.; SYM_GENERATORS['e'][1,0] = 1.; SYM_GENERATORS['e'][2,2] = -1.

#
SYM_GENERATORS['f'] = np.zeros([3,3])
SYM_GENERATORS['f'][0,1] = -1.; SYM_GENERATORS['f'][1,0] = -1.; SYM_GENERATORS['f'][2,2] = -1.

#
SYM_GENERATORS['g'] = np.zeros([3,3])
SYM_GENERATORS['g'][0,1] = -1.; SYM_GENERATORS['g'][1,0] = 1.; SYM_GENERATORS['g'][2,2] = 1.

# inversion
SYM_GENERATORS['h'] = np.zeros([3,3])
SYM_GENERATORS['h'] = -np.eye(3)

# c-mirror
SYM_GENERATORS['i'] = np.zeros([3,3])
SYM_GENERATORS['i'][0,0] = 1.; SYM_GENERATORS['i'][1,1] = 1.; SYM_GENERATORS['i'][2,2] = -1.

# b-mirror
SYM_GENERATORS['j'] = np.zeros([3,3])
SYM_GENERATORS['j'][0,0] = 1.; SYM_GENERATORS['j'][1,1] = -1.; SYM_GENERATORS['j'][2,2] = 1.

# 90@[001]
SYM_GENERATORS['k'] = np.zeros([3,3])
SYM_GENERATORS['k'][0,1] = -1.; SYM_GENERATORS['k'][1,0] = -1.; SYM_GENERATORS['k'][2,2] = 1.

#
SYM_GENERATORS['l'] = np.zeros([3,3])
SYM_GENERATORS['l'][0,1] = 1.; SYM_GENERATORS['l'][1,0] = 1.; SYM_GENERATORS['l'][2,2] = 1.

#
SYM_GENERATORS['m'] = np.zeros([3,3])
SYM_GENERATORS['m'][0,1] = 1.; SYM_GENERATORS['m'][1,0] = -1.; SYM_GENERATORS['m'][2,2] = -1. 

#
SYM_GENERATORS['n'] = np.zeros([3,3])
SYM_GENERATORS['n'][0,1] = -1.; SYM_GENERATORS['n'][1,0] = 1.; SYM_GENERATORS['n'][1,1] = -1.; SYM_GENERATORS['n'][2,2] = 1.

''' translation components
'''
SYM_GENERATORS['A'] = 1./6.
SYM_GENERATORS['B'] = 1./4.
SYM_GENERATORS['C'] = 1./3.
SYM_GENERATORS['D'] = 1./2.
SYM_GENERATORS['E'] = 2./3.
SYM_GENERATORS['F'] = 3./4.
SYM_GENERATORS['G'] = 5./6.
SYM_GENERATORS['O'] = 0.
SYM_GENERATORS['X'] = -3./8.
SYM_GENERATORS['Y'] = -1./4.
SYM_GENERATORS['Z'] = -1./8.