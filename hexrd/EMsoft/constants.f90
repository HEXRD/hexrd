! ###################################################################
! Copyright (c) 2014-2019, Marc De Graef Research Group/Carnegie Mellon University
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without modification, are 
! permitted provided that the following conditions are met:
!
!     - Redistributions of source code must retain the above copyright notice, this list 
!        of conditions and the following disclaimer.
!     - Redistributions in binary form must reproduce the above copyright notice, this 
!        list of conditions and the following disclaimer in the documentation and/or 
!        other materials provided with the distribution.
!     - Neither the names of Marc De Graef, Carnegie Mellon University nor the names 
!        of its contributors may be used to endorse or promote products derived from 
!        this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
! ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
! LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
! USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
! ###################################################################

!--------------------------------------------------------------------------
! EMsoft:constants.f90
!--------------------------------------------------------------------------
!
! MODULE: constants
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief 
!> Definition of constants and constant arrays used by other routines
!
!> @details  
!> physical and mathematical constants used by various programs; periodic
!>  table information; atomic weights; 
!
!> @note mod 5.1: f90wrap does not like arrays of parameters (constants), so we removed the 'parameter' attribute
!>  in all those cases... we should trust that the user will not modify the values in these arrays...
! 
!> @date 1/5/99   MDG 1.0 original
!> @date 5/18/01  MDG 2.0 f90
!> @date 11/27/01 MDG 2.1 added kind support
!> @date 03/19/13 MDG 3.0 added atomic weights 
!> @date 01/10/14 MDG 4.0 new version
!> @date 04/29/14 MDG 4.1 constants updated from NIST physical constants tables
!> @date 07/06/14 MDG 4.2 added omegamax to Lambert constant type
!> @date 08/11/14 MDG 4.3 added infty for handling of +Infinity in rotations module
!> @date 08/11/14 MDG 4.4 added epsijk option to package
!> @date 09/30/14 MDG 4.5 added some additional comments about epsijk
!> @date 10/02/14 MDG 4.6 removed omegamax again, since we now have properly dealt with 180 degree rotations
!> @date 03/11/15 MDG 4.7 added some additional comments about epsijk 
!> @date 04/02/17 MDG 4.8 modified definition of fundamental zones types and orders to accomodate two-phase disorientations
!> @date 10/24/17 MDG 4.9 removed definition of infty and inftyd to be replaced by functions in math.f90 module
!> @date 01/04/18 MDG 5.0 added quasicrystal constant IcoVertices
!> @date 09/01/19 MDG 5.1 remove 'parameter' attribute to allow for f90wrap python wrapping of parameter arrays...
!--------------------------------------------------------------------------

module constants

use local

IMPLICIT NONE

! ****************************************************
! ****************************************************
! ****************************************************
! used to change the sign of the permutation symbol from Adam Morawiec's book to
! the convention used for the EMsoft package.  If you want to use Adam's convention,
! both of these parameters should be set to +1; -1 will change the sign everywhere
! for all representations that involve the unit vector.  The quaternion product is 
! also redefined to include the epsijk parameter.  Doing so guarantees that the 
! quat_Lp operator ALWAYS returns an active result, regardless of the choice of epsijk;
! quat_LPstar ALWAYS returns a passive result.

! Uncomment these for an alternative way of doing things
!real(kind=sgl), parameter :: epsijk = -1.0
!real(kind=dbl), parameter :: epsijkd = -1.D0

! uncomment these for the Morawiec version.
real(kind=sgl), parameter :: epsijk = 1.0
real(kind=dbl), parameter :: epsijkd = 1.D0
!DEC$ ATTRIBUTES DLLEXPORT :: epsijk
!DEC$ ATTRIBUTES DLLEXPORT :: epsijkd

! In the first case, epsijk=-1, the rotation 120@[111] will result in 
! an axis angle pair of [111], 2pi/3.  In the second case, the axis-angle 
! pair will be -[111], 2pi/3.  In all cases, the rotations are interpreted
! in the passive sense.  The case epsijk=+1 corresponds to the mathematically 
! consistent case, using the standard definition for the quaternion product; in
! the other case, epsijk=-1, one must redefine the quaternion product in order
! to produce consistent results.  This takes a lengthy explanation ... see the
! rotations tutorial paper for an in-depth explanation.  These changes propagate
! to a number of files, notably quaternions.f90, and everywhere else that quaternions
! and rotations in general are used.
!
! Reference:  D.J. Rowenhorst, A.D. Rollett, G.S. Roher, M.A. Groeber, M.A. Jackson, 
!  P.J. Konijnenberg, and M. De Graef. "Tutorial: consistent representations of and 
!  conversions between 3D rotations". Modeling and Simulations in Materials Science 
!  and Engineering, 23, 083501 (2015).
!
! ****************************************************
! ****************************************************
! ****************************************************


! various physical constants
!> cPi		    = pi [dimensionless]
!> cLight	    = velocity of light [m/s]
!> cPlanck	    = Planck''s constant [Js]
!> cBoltzmann	= Boltmann constant [J/K]
!> cPermea	    = permeability of vacuum [4pi 10^7 H/m]
!> cPermit	    = permittivity of vacuum [F/m]
!> cCharge	    = electron charge [C]
!> cRestmass	= electron rest mass [kg]
!> cMoment	    = electron magnetic moment [J/T]
!> cJ2eV	    = Joules per eV
!> cAvogadro	= Avogadro's constant [mol^-1]
!
! The values of several of these constants have been updated to the new SI 2019 exact values [MDG, 01/22/19]
! The exact values below are the ones for cLight, cPlanck, cBoltzmann, cCharge; the others are derived using 
! the standard relations in the 2019 SI units document.  In the derivation, we used 0.0072973525664D0 as the 
! value for the hyperfine structure constant alpha. 
!
real(kind=dbl), parameter :: cPi=3.141592653589793238D0, cLight = 299792458.D0, &
                             cPlanck = 6.62607015D-34, cBoltzmann = 1.380649D-23,  &
                             cPermea = 1.2566370616D-6, cPermit = 8.8541878163D-12, &
                             cCharge = 1.602176634D-19, cRestmass = 9.1093837090D-31, &
                             cMoment = 9.2740100707D-24, cJ2eV = 1.602176565D-19, &
                             cAvogadro = 6.02214076D23
!DEC$ ATTRIBUTES DLLEXPORT :: cPi
!DEC$ ATTRIBUTES DLLEXPORT :: cPlanck
!DEC$ ATTRIBUTES DLLEXPORT :: cPermea
!DEC$ ATTRIBUTES DLLEXPORT :: cCharge
!DEC$ ATTRIBUTES DLLEXPORT :: cMoment
!DEC$ ATTRIBUTES DLLEXPORT :: cAvogadro
!DEC$ ATTRIBUTES DLLEXPORT :: cLight
!DEC$ ATTRIBUTES DLLEXPORT :: cBoltzmann
!DEC$ ATTRIBUTES DLLEXPORT :: cPermit
!DEC$ ATTRIBUTES DLLEXPORT :: cRestmass
!DEC$ ATTRIBUTES DLLEXPORT :: cJ2eV

!> element symbols (we'll do 1-98 for all parameter lists)
character(2), dimension(98) :: ATOM_sym=(/' H','He','Li','Be',' B',' C',' N',' O',' F','Ne', &
                                          'Na','Mg','Al','Si',' P',' S','Cl','Ar',' K','Ca', &
                                          'Sc','Ti',' V','Cr','Mn','Fe','Co','Ni','Cu','Zn', &
                                          'Ga','Ge','As','Se','Br','Kr','Rb','Sr',' Y','Zr', &
                                          'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn', &
                                          'Sb','Te',' I','Xe','Cs','Ba','La','Ce','Pr','Nd', &
                                          'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb', &
                                          'Lu','Hf','Ta',' W','Re','Os','Ir','Pt','Au','Hg', &
                                          'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th', &
                                          'Pa',' U','Np','Pu','Am','Cm','Bk','Cf'/)
!DEC$ ATTRIBUTES DLLEXPORT :: ATOM_sym

!> Shannon-Prewitt ionic radii in nanometer
real(kind=sgl), dimension(98) :: ATOM_SPradii=(/0.010,0.010,0.680,0.300,0.160,0.150,0.148,0.146,0.133,0.500, &
                                                0.098,0.065,0.450,0.380,0.340,0.190,0.181,0.500,0.133,0.940, &
                                                0.068,0.060,0.740,0.690,0.670,0.640,0.630,0.620,0.720,0.740, &
                                                0.113,0.073,0.580,0.202,0.196,0.500,0.148,0.110,0.880,0.770, &
                                                0.067,0.068,0.500,0.500,0.500,0.860,0.126,0.970,0.132,0.930, &
                                                0.076,0.222,0.219,0.500,0.167,0.129,0.104,0.111,0.500,0.108, &
                                                0.050,0.104,0.500,0.970,0.500,0.990,0.500,0.960,0.500,0.940, &
                                                0.050,0.050,0.680,0.600,0.520,0.500,0.500,0.500,0.137,0.112, &
                                                0.140,0.132,0.740,0.230,0.227,0.500,0.175,0.137,0.111,0.990, &
                                                0.090,0.083,0.500,0.108,0.500,0.500,0.500,0.500/)
!DEC$ ATTRIBUTES DLLEXPORT :: ATOM_SPradii

!> atomic (metallic) radii in nanometer (0.100 if not known/applicable)
real(kind=sgl), dimension(98) :: ATOM_MTradii=(/0.100,0.100,0.156,0.112,0.100,0.100,0.100,0.100,0.100,0.100, &
                                                0.191,0.160,0.142,0.100,0.100,0.100,0.100,0.100,0.238,0.196, &
                                                0.160,0.146,0.135,0.128,0.136,0.127,0.125,0.124,0.128,0.137, &
                                                0.135,0.139,0.125,0.116,0.100,0.100,0.253,0.215,0.181,0.160, &
                                                0.147,0.140,0.135,0.133,0.134,0.137,0.144,0.152,0.167,0.158, &
                                                0.161,0.143,0.100,0.100,0.270,0.224,0.187,0.182,0.182,0.181, &
                                                0.100,0.100,0.204,0.178,0.177,0.175,0.176,0.173,0.174,0.193, &
                                                0.173,0.158,0.147,0.141,0.137,0.135,0.135,0.138,0.144,0.155, &
                                                0.171,0.174,0.182,0.168,0.100,0.100,0.100,0.100,0.100,0.180, &
                                                0.163,0.154,0.150,0.164,0.100,0.100,0.100,0.100/)
!DEC$ ATTRIBUTES DLLEXPORT :: ATOM_MTradii

!> atom colors for PostScript drawings
character(3), dimension(98) :: ATOM_color=(/'blu','grn','blu','blu','red','bro','blu','red','grn','grn', &
                                            'blu','pnk','grn','blu','pnk','cyn','blu','blu','grn','grn', &
                                            'blu','blu','grn','red','pnk','cyn','blu','blu','grn','grn', &
                                            'blu','blu','grn','red','pnk','cyn','blu','blu','grn','grn', &
                                            'blu','blu','grn','red','pnk','cyn','blu','blu','grn','grn', &
                                            'blu','blu','grn','red','pnk','cyn','blu','blu','grn','grn', &
                                            'blu','blu','grn','red','pnk','cyn','blu','blu','grn','grn', &
                                            'blu','blu','grn','red','pnk','cyn','blu','blu','grn','grn', &
                                            'blu','blu','grn','red','pnk','cyn','blu','blu','grn','grn', &
                                            'blu','blu','grn','red','pnk','cyn','blu','grn'/)

real(kind=sgl), dimension(3,92) :: ATOM_colors = reshape( (/ &
                                            0.90000,0.90000,0.15000, &
                                            0.00000,0.90000,0.15000, &
                                            0.32311,0.53387,0.69078, &
                                            0.61572,0.99997,0.61050, &
                                            0.53341,0.53341,0.71707, &
                                            0.06577,0.02538,0.00287, &
                                            0.50660,0.68658,0.90000, &
                                            0.90000,0.00000,0.00000, &
                                            0.09603,0.80000,0.74127, &
                                            0.90000,0.12345,0.54321, &
                                            0.78946,0.77423,0.00002, &
                                            0.41999,0.44401,0.49998, &
                                            0.09751,0.67741,0.90000, &
                                            0.00000,0.00000,0.90000, &
                                            0.53486,0.51620,0.89000, &
                                            0.90000,0.98070,0.00000, &
                                            0.94043,0.96999,0.37829, &
                                            0.33333,0.00000,0.33333, &
                                            0.65547,0.58650,0.69000, &
                                            0.36245,0.61630,0.77632, &
                                            0.98804,0.41819,0.90000, &
                                            0.31588,0.53976,0.67494, &
                                            0.83998,0.08402,0.67625, &
                                            0.90000,0.00000,0.60000, &
                                            0.90000,0.00000,0.60000, &
                                            0.71051,0.44662,0.00136, &
                                            0.00000,0.00000,0.68666, &
                                            0.22939,0.61999,0.60693, &
                                            0.78996,0.54162,0.14220, &
                                            0.52998,0.49818,0.50561, &
                                            0.74037,0.90000,0.18003, &
                                            0.66998,0.44799,0.19431, &
                                            0.53341,0.53341,0.71707, &
                                            0.92998,0.44387,0.01862, &
                                            0.96999,0.53349,0.24250, &
                                            0.25000,0.75000,0.50000, &
                                            0.90000,0.00000,0.60000, &
                                            0.00000,0.90000,0.15256, &
                                            0.00000,0.00000,0.90000, &
                                            0.00000,0.90000,0.00000, &
                                            0.50660,0.68658,0.90000, &
                                            0.35003,0.52340,0.90000, &
                                            0.80150,0.69171,0.79129, &
                                            0.92998,0.79744,0.04651, &
                                            0.90000,0.06583,0.05002, &
                                            0.00002,0.00005,0.76999, &
                                            0.09751,0.67741,0.90000, &
                                            0.55711,0.50755,0.90000, &
                                            0.22321,0.72000,0.33079, &
                                            0.29000,0.26679,0.28962, &
                                            0.53999,0.42660,0.45495, &
                                            0.90000,0.43444,0.13001, &
                                            0.42605,0.14739,0.66998, &
                                            0.13001,0.90000,0.24593, &
                                            0.90000,0.00000,0.60000, &
                                            0.74342,0.39631,0.45338, &
                                            0.72000,0.24598,0.15122, &
                                            0.00000,0.00000,0.90000, &
                                            0.90000,0.44813,0.23003, &
                                            0.20000,0.90000,0.11111, &
                                            0.90000,0.00000,0.00000, &
                                            0.99042,0.02402,0.49194, &
                                            0.90000,0.00000,0.60000, &
                                            0.66998,0.44799,0.19431, &
                                            0.23165,0.09229,0.57934, &
                                            0.90000,0.87648,0.81001, &
                                            0.00000,0.20037,0.64677, &
                                            0.53332,0.53332,0.53332, &
                                            0.15903,0.79509,0.98584, &
                                            0.15322,0.99164,0.95836, &
                                            0.18293,0.79933,0.59489, &
                                            0.83000,0.09963,0.55012, &
                                            0.34002,0.36210,0.90000, &
                                            0.46000,0.19898,0.03679, &
                                            0.06270,0.56999,0.12186, &
                                            0.18003,0.24845,0.90000, &
                                            0.33753,0.30100,0.43000, &
                                            0.25924,0.25501,0.50999, &
                                            0.90000,0.79663,0.39001, &
                                            0.47999,0.47207,0.46557, &
                                            0.70549,0.83000,0.74490, &
                                            0.38000,0.32427,0.31919, &
                                            0.62942,0.42309,0.73683, &
                                            0.90000,0.00000,0.00000, &
                                            0.50000,0.33333,0.33333, &
                                            0.72727,0.12121,0.50000, &
                                            0.90000,0.00000,0.00000, &
                                            0.46310,0.90950,0.01669, &
                                            0.66667,0.66666,0.00000, &
                                            0.14893,0.99596,0.47105, &
                                            0.53332,0.53332,0.53332, &
                                            0.47773,0.63362,0.66714 /), (/3,92/))
!DEC$ ATTRIBUTES DLLEXPORT :: ATOM_colors

!> Doyle-turner scattering factor coefficients
real(kind=sgl), dimension(8,98)            :: scatfac = reshape( (/ &
        0.202,0.244,0.082,0.000,0.30868,0.08544,0.01273,0.00000, &
        0.091,0.181,0.110,0.036,0.18183,0.06212,0.01803,0.00284, &
        1.611,1.246,0.326,0.099,1.07638,0.30480,0.04533,0.00495, &
        1.250,1.334,0.360,0.106,0.60804,0.18591,0.03653,0.00416, &
        0.945,1.312,0.419,0.116,0.46444,0.14178,0.03223,0.00377, &
        0.731,1.195,0.456,0.125,0.36995,0.11297,0.02814,0.00346, &
        0.572,1.043,0.465,0.131,0.28847,0.09054,0.02421,0.00317, &
        0.455,0.917,0.472,0.138,0.23780,0.07622,0.02144,0.00296, &
        0.387,0.811,0.475,0.146,0.20239,0.06609,0.01931,0.00279, &
        0.303,0.720,0.475,0.153,0.17640,0.05860,0.01762,0.00266, &
        2.241,1.333,0.907,0.286,1.08004,0.24505,0.03391,0.00435, &
        2.268,1.803,0.839,0.289,0.73670,0.20175,0.03013,0.00405, &
        2.276,2.428,0.858,0.317,0.72322,0.19773,0.03080,0.00408, &
        2.129,2.533,0.835,0.322,0.57775,0.16476,0.02880,0.00386, &
        1.888,2.469,0.805,0.320,0.44876,0.13538,0.02642,0.00361, &
        1.659,2.386,0.790,0.321,0.36650,0.11488,0.02469,0.00340, &
        1.452,2.292,0.787,0.322,0.30935,0.09980,0.02234,0.00323, &
        1.274,2.190,0.793,0.326,0.26682,0.08813,0.02219,0.00307, &
        3.951,2.545,1.980,0.482,1.37075,0.22402,0.04532,0.00434, &
        4.470,2.971,1.970,0.482,0.99523,0.22696,0.04195,0.00417, &
        3.966,2.917,1.925,0.480,0.88960,0.20606,0.03856,0.00399, &
        3.565,2.818,1.893,0.483,0.81982,0.19049,0.03590,0.00386, &
        3.245,2.698,1.860,0.486,0.76379,0.17726,0.03363,0.00374, &
        2.307,2.334,1.823,0.490,0.78405,0.15785,0.03157,0.00364, &
        2.747,2.456,1.792,0.498,0.67786,0.15674,0.03000,0.00357, &
        2.544,2.343,1.759,0.506,0.64424,0.14880,0.02854,0.00350, &
        2.367,2.236,1.724,0.515,0.61431,0.14180,0.02725,0.00344, &
        2.210,2.134,1.689,0.524,0.58727,0.13553,0.02609,0.00339, &
        1.579,1.820,1.658,0.532,0.62940,0.12453,0.02504,0.00333, &
        1.942,1.950,1.619,0.543,0.54162,0.12518,0.02416,0.00330, &
        2.321,2.486,1.688,0.599,0.65602,0.15458,0.02581,0.00351, &
        2.447,2.702,1.616,0.601,0.55893,0.14393,0.02446,0.00342, &
        2.399,2.790,1.529,0.594,0.45718,0.12817,0.02280,0.00328, &
        2.298,2.854,1.456,0.590,0.38830,0.11536,0.02146,0.00316, &
        2.166,2.904,1.395,0.589,0.33899,0.10497,0.02041,0.00307, &
        2.034,2.927,1.342,0.589,0.29999,0.09598,0.01952,0.00299, &
        4.776,3.859,2.234,0.868,1.40782,0.18991,0.03701,0.00419, &
        5.848,4.003,2.342,0.880,1.04972,0.19367,0.03737,0.00414, &
        4.129,3.012,1.179,0.000,0.27548,0.05088,0.00591,0.000,   &
        4.105,3.144,1.229,0.000,0.28492,0.05277,0.00601,0.000,   &
        4.237,3.105,1.234,0.000,0.27415,0.05074,0.00593,0.000,   &
        3.120,3.906,2.361,0.850,0.72464,0.14642,0.03237,0.00366, &
        4.318,3.270,1.287,0.000,0.28246,0.05148,0.00590,0.000,   &
        4.358,3.298,1.323,0.000,0.27881,0.05179,0.00594,0.000,   &
        4.431,3.343,1.345,0.000,0.27911,0.05153,0.00592,0.000,   &
        4.436,3.454,1.383,0.000,0.28670,0.05269,0.00595,0.000,   &
        2.036,3.272,2.511,0.837,0.61497,0.11824,0.02846,0.00327, &
        2.574,3.259,2.547,0.838,0.55675,0.11838,0.02784,0.00322, &
        3.153,3.557,2.818,0.884,0.66649,0.14449,0.02976,0.00335, &
        3.450,3.735,2.118,0.877,0.59104,0.14179,0.02855,0.00327, &
        3.564,3.844,2.687,0.864,0.50487,0.13316,0.02691,0.00316, &
        4.785,3.688,1.500,0.000,0.27999,0.05083,0.00581,0.000,   &
        3.473,4.060,2.522,0.840,0.39441,0.11816,0.02415,0.00298, &
        3.366,4.147,2.443,0.829,0.35509,0.11117,0.02294,0.00289, &
        6.062,5.986,3.303,1.096,1.55837,0.19695,0.03335,0.00379, &
        7.821,6.004,3.280,1.103,1.17657,0.18778,0.03263,0.00376, &
        4.940,3.968,1.663,0.000,0.28716,0.05245,0.00594,0.000,   &
        5.007,3.980,1.678,0.000,0.28283,0.05183,0.00589,0.000,   &
        5.085,4.043,1.684,0.000,0.28588,0.05143,0.00581,0.000,   &
        5.151,4.075,1.683,0.000,0.28304,0.05073,0.00571,0.000,   &
        5.201,4.094,1.719,0.000,0.28079,0.05081,0.00576,0.000,   &
        5.255,4.113,1.743,0.000,0.28016,0.05037,0.00577,0.000,   &
        6.267,4.844,3.202,1.200,1.00298,0.16066,0.02980,0.00367, &
        5.225,4.314,1.827,0.000,0.29158,0.05259,0.00586,0.000,   &
        5.272,4.347,1.844,0.000,0.29046,0.05226,0.00585,0.000,   &
        5.332,4.370,1.863,0.000,0.28888,0.05198,0.00581,0.000,   &
        5.376,4.403,1.884,0.000,0.28773,0.05174,0.00582,0.000,   &
        5.436,4.437,1.891,0.000,0.28655,0.05117,0.00577,0.000,   &
        5.441,4.510,1.956,0.000,0.29149,0.05264,0.00590,0.000,   &
        5.529,4.533,1.945,0.000,0.28927,0.05144,0.00578,0.000,   &
        5.553,4.580,1.969,0.000,0.28907,0.05160,0.00577,0.000,   &
        5.588,4.619,1.997,0.000,0.29001,0.05164,0.00579,0.000,   &
        5.659,4.630,2.014,0.000,0.28807,0.05114,0.00578,0.000,   &
        5.709,4.677,2.019,0.000,0.28782,0.05084,0.00572,0.000,   &
        5.695,4.740,2.064,0.000,0.28968,0.05156,0.00575,0.000,   &
        5.750,4.773,2.079,0.000,0.28933,0.05139,0.00573,0.000,   &
        5.754,4.851,2.096,0.000,0.29159,0.05152,0.00570,0.000,   &
        5.803,4.870,2.127,0.000,0.29016,0.05150,0.00572,0.000,   &
        2.388,4.226,2.689,1.255,0.42866,0.09743,0.02264,0.00307, &
        2.682,4.241,2.755,1.270,0.42822,0.09856,0.02295,0.00307, &
        5.932,4.972,2.195,0.000,0.29086,0.05126,0.00572,0.000,   &
        3.510,4.552,3.154,1.359,0.52914,0.11884,0.02571,0.00321, &
        3.841,4.679,3.192,1.363,0.50261,0.11999,0.02560,0.00318, &
        6.070,4.997,2.232,0.000,0.28075,0.04999,0.00563,0.000,   &
        6.133,5.031,2.239,0.000,0.28047,0.04957,0.00558,0.000,   &
        4.078,4.978,3.096,1.326,0.38406,0.11020,0.02355,0.00299, &
        6.201,5.121,2.275,0.000,0.28200,0.04954,0.00556,0.000,   &
        6.215,5.170,2.316,0.000,0.28382,0.05002,0.00562,0.000,   &
        6.278,5.195,2.321,0.000,0.28323,0.04949,0.00557,0.000,   &
        6.264,5.263,2.367,0.000,0.28651,0.05030,0.00563,0.000,   &
        6.306,5.303,2.386,0.000,0.28688,0.05026,0.00561,0.000,   &
        6.767,6.729,4.014,1.561,0.85951,0.15642,0.02936,0.00335, &
        6.323,5.414,2.453,0.000,0.29142,0.05096,0.00568,0.000,   &
        6.415,5.419,2.449,0.000,0.28836,0.05022,0.00561,0.000,   &
        6.378,5.495,2.495,0.000,0.29156,0.05102,0.00565,0.000,   &
        6.460,5.469,2.471,0.000,0.28396,0.04970,0.00554,0.000,   &
        6.502,5.478,2.510,0.000,0.28375,0.04975,0.00561,0.000,   &
        6.548,5.526,2.520,0.000,0.28461,0.04965,0.00557,0.000/), (/8,98/))

!> atomic weights for things like density computations (from NIST elemental data base)
real(kind=sgl),dimension(98)    :: ATOM_weights(98) = (/1.00794, 4.002602, 6.941, 9.012182, 10.811, &
                                                        12.0107, 14.0067, 15.9994, 18.9984032, 20.1797, &
                                                        22.98976928, 24.3050, 26.9815386, 28.0855, 30.973762, &
                                                        32.065, 35.453, 39.948, 39.0983, 40.078, &
                                                        44.955912, 47.867, 50.9415, 51.9961, 54.938045, &
                                                        55.845, 58.933195, 58.6934, 63.546, 65.38, &
                                                        69.723, 72.64, 74.92160, 78.96, 79.904, &
                                                        83.798, 85.4678, 87.62, 88.90585, 91.224, &
                                                        92.90638, 95.96, 98.9062, 101.07, 102.90550, &
                                                        106.42, 107.8682, 112.411, 114.818, 118.710, &
                                                        121.760, 127.60, 126.90447, 131.293, 132.9054519, &
                                                        137.327, 138.90547, 140.116, 140.90765, 144.242, &
                                                        145.0, 150.36, 151.964, 157.25, 158.92535, &
                                                        162.500, 164.93032, 167.259, 168.93421, 173.054, &
                                                        174.9668, 178.49, 180.94788, 183.84, 186.207, &
                                                        190.23, 192.217, 195.084, 196.966569, 200.59, &
                                                        204.3833, 207.2, 208.98040, 209.0, 210.0, &
                                                        222.0, 223.0, 226.0, 227.0, 232.03806, &
                                                        231.03588, 238.02891, 237.0, 244.0, 243.0, &
                                                        247.0, 251.0, 252.0 /)
!DEC$ ATTRIBUTES DLLEXPORT :: ATOM_weights


! these are a bunch of constants used for Lambert and related projections; they are all in double precision
type LambertParametersType
        real(kind=dbl)          :: Pi=3.141592653589793D0       !  pi
        real(kind=dbl)          :: iPi=0.318309886183791D0      !  1/pi
        real(kind=dbl)          :: sPi=1.772453850905516D0      !  sqrt(pi)
        real(kind=dbl)          :: sPio2=1.253314137315500D0    !  sqrt(pi/2)
        real(kind=dbl)          :: sPi2=0.886226925452758D0     !  sqrt(pi)/2
        real(kind=dbl)          :: srt=0.86602540378D0      !  sqrt(3)/2
        real(kind=dbl)          :: isrt=0.57735026919D0    !  1/sqrt(3)
        real(kind=dbl)          :: alpha=1.346773687088598D0   !  sqrt(pi)/3^(1/4)
        real(kind=dbl)          :: rtt=1.732050807568877D0      !  sqrt(3)
        real(kind=dbl)          :: prea=0.525037567904332D0    !  3^(1/4)/sqrt(2pi)
        real(kind=dbl)          :: preb=1.050075135808664D0     !  3^(1/4)sqrt(2/pi)
        real(kind=dbl)          :: prec=0.906899682117109D0    !  pi/2sqrt(3)
        real(kind=dbl)          :: pred=2.094395102393195D0     !  2pi/3
        real(kind=dbl)          :: pree=0.759835685651593D0     !  3^(-1/4)
        real(kind=dbl)          :: pref=1.381976597885342D0     !  sqrt(6/pi)
        real(kind=dbl)          :: preg=1.5551203015562141D0    ! 2sqrt(pi)/3^(3/4)
! the following constants are used for the cube to quaternion hemisphere mapping
        real(kind=dbl)          :: a=1.925749019958253D0        ! pi^(5/6)/6^(1/6)
        real(kind=dbl)          :: ap=2.145029397111025D0       ! pi^(2/3)
        real(kind=dbl)          :: sc=0.897772786961286D0       ! a/ap
        real(kind=dbl)          :: beta=0.962874509979126D0     ! pi^(5/6)/6^(1/6)/2
        real(kind=dbl)          :: R1=1.330670039491469D0       ! (3pi/4)^(1/3)
        real(kind=dbl)          :: r2=1.414213562373095D0       ! sqrt(2)
        real(kind=dbl)          :: r22=0.707106781186547D0      ! 1/sqrt(2)
        real(kind=dbl)          :: pi12=0.261799387799149D0     ! pi/12
        real(kind=dbl)          :: pi8=0.392699081698724D0      ! pi/8
        real(kind=dbl)          :: prek=1.643456402972504D0     ! R1 2^(1/4)/beta
        real(kind=dbl)          :: r24=4.898979485566356D0      ! sqrt(24)
!       real(kind=dbl)          :: tfit(16) = (/1.0000000000018852D0, -0.5000000002194847D0, & 
!                                            -0.024999992127593126D0, - 0.003928701544781374D0, & 
!                                            -0.0008152701535450438D0, - 0.0002009500426119712D0, & 
!                                            -0.00002397986776071756D0, - 0.00008202868926605841D0, & 
!                                            +0.00012448715042090092D0, - 0.0001749114214822577D0, & 
!                                            +0.0001703481934140054D0, - 0.00012062065004116828D0, & 
!                                            +0.000059719705868660826D0, - 0.00001980756723965647D0, & 
!                                            +0.000003953714684212874D0, - 0.00000036555001439719544D0 /)
! a more accurate fit, up to order 40  [MDG, 03/28/16]
        real(kind=dbl)          :: tfit(21) = (/ 0.9999999999999968D0, -0.49999999999986866D0,  &
                                                -0.025000000000632055D0, - 0.003928571496460683D0, &
                                                -0.0008164666077062752D0, - 0.00019411896443261646D0, &
                                                -0.00004985822229871769D0, - 0.000014164962366386031D0, &
                                                -1.9000248160936107D-6, - 5.72184549898506D-6, &
                                                7.772149920658778D-6, - 0.00001053483452909705D0, &
                                                9.528014229335313D-6, - 5.660288876265125D-6, &
                                                1.2844901692764126D-6, 1.1255185726258763D-6, &
                                                -1.3834391419956455D-6, 7.513691751164847D-7, &
                                                -2.401996891720091D-7, 4.386887017466388D-8, &
                                                -3.5917775353564864D-9 /)

	real(kind=dbl)          :: BP(12)= (/ 0.D0, 1.D0, 0.577350269189626D0, 0.414213562373095D0, 0.D0,  &
                                             0.267949192431123D0, 0.D0, 0.198912367379658D0, 0.D0, &
                                             0.158384440324536D0, 0.D0, 0.131652497587396D0/)       ! used for Fundamental Zone determination in so3 module
end type LambertParametersType

type(LambertParametersType)        :: LPs
!DEC$ ATTRIBUTES DLLEXPORT :: LPs






! The following two arrays are used to determine the FZtype (FZtarray) and primary rotation axis order (FZoarray)
! for each of the 32 crystallographic point group symmetries (in the order of the International Tables)
!
!                                       '    1','   -1','    2','    m','  2/m','  222', &
!                                       '  mm2','  mmm','    4','   -4','  4/m','  422', &
!                                       '  4mm',' -42m','4/mmm','    3','   -3','   32', &
!                                       '   3m','  -3m','    6','   -6','  6/m','  622', &
!                                       '  6mm',' -6m2','6/mmm','   23','   m3','  432', &
!                                       ' -43m',' m-3m'/
!
! 1 (C1), -1 (Ci), [triclinic]
! 2 (C2), m (Cs), 2/m (C2h), [monoclinic]
! 222 (D2), mm2 (C2v), mmm (D2h), [orthorhombic]
! 4 (C4), -4 (S4), 4/m (C4h), 422 (D4), 4mm (C4v), -42m (D2d), 4/mmm (D4h), [tetragonal]
! 3 (C3), -3 (C3i), 32 (D3), 3m (C3v), -3m (D3d), [trigonal]
! 6 (C6), -6 (C3h), 6/m (C6h), 622 (D6), 6mm (C6v), -6m2 (D3h), 6/mmm (D6h), [hexagonal]
! 23 (T), m3 (Th), 432 (O), -43m (Td), m-3m (Oh) [cubic]
!
! FZtype
! 0        no symmetry at all
! 1        cyclic symmetry
! 2        dihedral symmetry
! 3        tetrahedral symmetry
! 4        octahedral symmetry
!
! these parameters are used in the so3 module
!
integer(kind=irg),dimension(36)     :: FZtarray = (/ 0,0,1,1,1,2,2,2,1,1,1,2,2,2,2,1,1,2, &
                                                     2,2,1,1,1,2,2,2,2,3,3,4,3,4,5,2,2,2 /)
!DEC$ ATTRIBUTES DLLEXPORT :: FZtarray

integer(kind=irg),dimension(36)     :: FZoarray = (/ 0,0,2,2,2,2,2,2,4,4,4,4,4,4,4,3,3,3, &
                                                     3,3,6,6,6,6,6,6,6,0,0,0,0,0,0,8,10,12 /)
!DEC$ ATTRIBUTES DLLEXPORT :: FZoarray

! vertex coordinates of the icosahedron (normalized)
real(kind=dbl),dimension(3,12)      :: IcoVertices = reshape( (/ 0D0,0.D0,1.D0, &
                                     0.89442719099991587856D0,0.D0,0.44721359549995793928D0, &
                                     0.27639320225002103036D0,0.85065080835203993218D0,0.44721359549995793928D0, &
                                    -0.72360679774997896964D0,0.52573111211913360603D0,0.44721359549995793928D0, &
                                    -0.72360679774997896964D0,-0.52573111211913360603D0,0.44721359549995793928D0, &
                                     0.27639320225002103036D0,-0.85065080835203993218D0,0.44721359549995793928D0, &
                                    -0.89442719099991587856D0,0.D0,-0.44721359549995793928D0, &
                                    -0.27639320225002103036D0,-0.85065080835203993218D0,-0.44721359549995793928D0, &
                                     0.72360679774997896964D0,-0.52573111211913360603D0,-0.44721359549995793928D0, &
                                     0.72360679774997896964D0,0.52573111211913360603D0,-0.44721359549995793928D0, &
                                    -0.27639320225002103036D0,0.85065080835203993218D0,-0.44721359549995793928D0, &
                                     0.D0,0.D0,-1.D0 /), (/3,12/))
!DEC$ ATTRIBUTES DLLEXPORT :: IcoVertices

end module
