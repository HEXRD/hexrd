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
! EMsoft:diffraction.f90
!--------------------------------------------------------------------------
!
! MODULE: diffraction
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief Anything related to dynamical diffraction
!
!> @todo general cleanup; merge with dynamical module?; add Private/Public 
! 
!> @date   10/13/98 MDG 1.0 original
!> @date    5/22/01 MDG 2.0 f90
!> @date   11/27/01 MDG 2.1 added kind support
!> @date    3/14/02 MDG 2.2 added CalcDynMat routine
!> @date   01/10/14 MDG 3.0 update with new cell type etc...
!> @date   12/01/14 MDG 3.1 removal of all global variables, including mLambda etc...
!--------------------------------------------------------------------------
module diffraction

use local
use typedefs

! atomic scattering factor parametrization (Doyle-Turner, Smith-Burge)
! used only if absorption is not taken into account;  otherwise
! the Weickenmeier-Kohl routine is used.
real(kind=sgl),parameter,private   :: scatfac(8,98) = reshape( (/ &
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


! the following variables used to be globals, but they are now entries in the cell pointer structure...
!
! mLambda       = electron wavelength [nm]
! mRelcor       = relativistic correction factor gamma [dimensionless]
! mSigma        = interaction constant [ ]
! mPsihat       = relativistic acceleration potential

! interface statements
interface Calcsg
        module procedure CalcsgSingle
        module procedure CalcsgDouble
end interface

contains

!--------------------------------------------------------------------------
!
! SUBROUTINE: CalcWaveLength
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief compute the electron wavelength and select scattering data set
! 
!> @details computes the relativistic electron wavelength
!>  These quantities are computed in double precision because of the 
!>  wide range of magnitudes.  If a crystal structure has been defined
!>  then the gamma*V_0 term is added to correct for refraction.
!
!> @param cell unit cell pointer
!> @param skip scattering set identifier (optional)
!
!> @date   10/20/98 MDG 1.0 original
!> @date    5/22/01 MDG 2.0 f90
!> @date   11/27/01 MDG 2.1 added kind support
!> @date   03/26/13 MDG 3.0 updated IO
!> @date   12/02/14 MDG 3.1 removed mAccvol as global variable
!--------------------------------------------------------------------------
recursive subroutine CalcWaveLength(cell,rlp,skip,verbose)
!DEC$ ATTRIBUTES DLLEXPORT :: CalcWaveLength

use constants
use symmetry
use io

IMPLICIT NONE

type(unitcell)                          :: cell
type(gnode),INTENT(INOUT)               :: rlp
!f2py intent(in,out) ::  rlp
integer(kind=irg),INTENT(IN),OPTIONAL   :: skip                 !< scattering set identifier
logical,INTENT(IN),OPTIONAL             :: verbose

real(kind=dbl)                          :: temp1,temp2, oi_real(1)
integer(kind=irg)                       :: hkl(3), io_int(1)

  temp1 = 1.0D+9*cPlanck/dsqrt(2.D0*cRestmass*cCharge)
  temp2 = cCharge*0.5D0*cell%voltage*1000.D0/cRestmass/(cLight**2)

! relativistic correction factor (known as gamma)      
  cell%mRelcor = 1.0D0+2.0D0*temp2

! relativistic acceleration voltage
  cell%mPsihat = cell%voltage*(1.D0+temp2)*1000.D0

! compute the electron wavelength in nm
! compute V_0 and add it to mPsihat (corrected by mRelcor)
  call CalcPositions(cell,'v')

! which scattering factors should be used ?
  if (present(skip)) then
   select case (skip) 
    case(1); rlp%method='DT'; 
    case(2); rlp%method='WK'; 
    case(3); rlp%method='WK'; rlp%absorption=.TRUE.
   end select
  else
   call Message(' The following scattering factor sets are available :', frm = "(/A/)")
   call Message('  [1] Doyle-Turner/Smith-Burge (no absorption) ', frm = "(A)")
   call Message('  [2] Weickenmeier-Kohl (no absorption) ', frm = "(A)")
   call Message('  [3] Weickenmeier-Kohl (with absorption) ', frm = "(A/)")
   call ReadValue(' Which set do you want to use [1/2/3] ? ', io_int,1)
   rlp%absorption = .FALSE.
   select case (io_int(1)) 
    case(1); rlp%method='DT'; 
    case(2); rlp%method='WK'; 
    case(3); rlp%method='WK'; rlp%absorption=.TRUE.
   end select
  end if

 hkl=(/0,0,0/)
 call CalcUcg(cell,rlp,hkl) 
 cell%mPsihat = cell%mPsihat + dble(rlp%Vmod)
 cell%mLambda = temp1/dsqrt(cell%mPsihat)
! interaction constant sigma
 cell%mSigma = 2.D0*cPi*cRestmass*cell%mRelcor*cCharge*cell%mLambda
 cell%mSigma = 1.0D-18*cell%mSigma/cPlanck**2

 if (present(verbose)) then
  if (verbose) then
    oi_real(1) = rlp%Vmod
    call WriteValue(' Mean inner potential [V] ', oi_real, 1,"(' ',E10.4)")
    call Message(' Wavelength corrected for refraction', frm = "(A)")
    oi_real(1) = cell%mRelcor
    call WriteValue(' Relativistic correction factor [gamma]  ', oi_real, 1,"(' ',E10.4)")
    oi_real(1) = cell%mPsihat
    call WriteValue(' Relativistic Accelerating Potential [V] ', oi_real, 1,"(' ',E10.4)")
    oi_real(1) = cell%mLambda
    call WriteValue(' Electron Wavelength [nm]                ', oi_real, 1,"(' ',E10.4)")
    oi_real(1) = cell%mSigma
    call WriteValue(' Interaction constant [V nm]^(-1)        ', oi_real, 1,"(' ',E10.4)")
  end if
 end if

 
end subroutine

!--------------------------------------------------------------------------
!
! FUNCTION: CalcDiffAngle
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief compute the diffraction angle 2theta in radians
!
!> @param h  Miller index
!> @param k  Miller index
!> @param l  Miller index
!
!> @date   10/20/98 MDG 1.0 original
!> @date    5/22/01 MDG 2.0 f90
!> @date   11/27/01 MDG 2.1 added kind support
!> @date   03/26/13 MDG 3.0 updated IO
!--------------------------------------------------------------------------
recursive function CalcDiffAngle(cell,h,k,l) result(tt)
!DEC$ ATTRIBUTES DLLEXPORT :: CalcDiffAngle

use crystal

IMPLICIT NONE

type(unitcell)                  :: cell
integer(kind=irg),INTENT(IN)    :: h,k,l                !< Miller indices

real(kind=sgl)                  :: tt

tt = 2.0*asin(0.50*sngl(cell%mLambda)*CalcLength( cell, float( (/h,k,l/) ), 'r') )

end function

!--------------------------------------------------------------------------
!
! FUNCTION: LorentzPF
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief compute the Lorentz Polarization Factor Lp
!
!> @param theta scattering angle  
!> @param HEDM optional string to indicate HEDM mode
!
!> @date  03/26/13  MDG  1.0 added for HEDM project
!--------------------------------------------------------------------------
recursive function LorentzPF(theta,HEDM) result(tt)
!DEC$ ATTRIBUTES DLLEXPORT :: LorentzPF

use crystal

IMPLICIT NONE

real(kind=sgl),INTENT(IN)                       :: theta                !< scattering angle
character(*),INTENT(IN),OPTIONAL                :: HEDM         !< for HEDM we have a different polarization factor
real(kind=sgl)                                  :: tt

if (present(HEDM)) then
  tt = (1.0+cos(2.0*theta)**2) / sin(theta)**2 / cos(theta)
else
  tt = (1.0+cos(2.0*theta)**2) / sin(theta)**2 / cos(theta)
end if

end function

!--------------------------------------------------------------------------
!
! SUBROUTINE: CalcUcg
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief compute the complex Structure Factor for a given g
!
!> @details includes computation of extinction distance, absorption length, etc...
!> This routine is probably the most important one in all the dynamical routines,
!> because it computes all possible relevant parameters and stores them in the rlp variable.
!> We've added the XR rlp%method parameter so that the same routine can be used for the
!> computation of kinematical x-ray scattering; this was needed for the HEDM package.
!
!> @param cell unit cell pointer
!> @param rlp reciprocal lattice point
!> @param hkl  Miller indices
!> @param applyqgshift [optional] multiply qg by exp[i theta_g] if present and .TRUE.
!
!> @note CalcPositions must be called before calling this routine
!
!> @date  10/20/98 MDG 1.0 original
!> @date   5/22/01 MDG 2.0 f90
!> @date  11/27/01 MDG 2.1 added kind support
!> @date  03/26/13 MDG 3.0 updated IO
!> @date  03/26/13 MDG 3.1 added XRD support
!> @date  01/10/14 MDG 4.0 new cell type
!> @date  06/09/14 MDG 4.1 added cell as argument
!> @date  12/02/14 MDG 4.2 added voltage as argument
!> @date  09/11/15 MDG 4.3 added optional argument
!> @date  08/09/18 MDG 5.0 added option to use precomputed FSCATT values stored in cell structure
!--------------------------------------------------------------------------
recursive subroutine CalcUcg(cell,rlp,hkl,applyqgshift,interpolate)
!DEC$ ATTRIBUTES DLLEXPORT :: CalcUcg

use crystal
use symmetry
use constants
use others

IMPLICIT NONE

type(unitcell)                  :: cell
type(gnode),INTENT(INOUT)       :: rlp
!f2py intent(in,out) ::  rlp
integer(kind=irg),INTENT(IN)    :: hkl(3)               !< Miller indices
logical,OPTIONAL,INTENT(IN)     :: applyqgshift
logical,OPTIONAL,INTENT(IN)     :: interpolate          ! requires rlp%mode = 'IP'

integer(kind=irg)               :: j,absflg,m,ii
real(kind=sgl)                  :: s,twopi,arg,swk,dwwk,pref,ul,pre,sct,fs,fsp
real(kind=sgl),parameter        :: preg = 0.664840340614319 ! = 2.0 * sngl(cRestmass*cCharge/cPlanck**2)*1.0E-18
complex(kind=sgl)               :: ff,gg,sf,p1
complex(kind=sgl)               :: czero
complex(kind=sgl),allocatable   :: sfarray(:)
logical                         :: accflg, dwflg, interp
character(2)                    :: smb

! interpolatiobn is only used for the WK mode to pre-compute the scattering factor array
interp = .FALSE.
if (present(interpolate)) then
  if (interpolate.eqv..TRUE.) interp=.TRUE.
end if

twopi = sngl(2.D0*cPi)
czero = cmplx(0.0,0.0)
rlp%hkl = hkl

! compute the scattering parameter s^2=(g/2)^2
 if (sum(hkl**2).eq.0) then 
  s = 0.0
  rlp%g = 0.0
 else
  rlp%g = sngl(CalcLength(cell,dble(hkl),'r'))
  s = (0.50*rlp%g)**2
 end if

!----------------------------------
! first the simplest case: kinematical X-ray scattering factors
! this option was added to accomodate the HEDM forward projector needs
if (rlp%method.eq.'XR') then 

! set the prefactor for the Doyle-Turner summation
  pref = 0.4178214

! initialize the real and imaginary parts of the structure factor
  sf = czero
  
! loop over all atoms in the asymmetric unit
 do m=1,cell % ATOM_ntype

! get the atomic scattering factor for this atom
  sct=0.0
  j=cell % ATOM_type(m)
  do ii=1,4
   sct=sct+scatfac(ii,j)*exp(-scatfac(ii+4,j)*s)
  end do

! scale and include Debye-Waller factor and site occupation parameter
  fsp = pref * s * sct 
  fs = (float(cell % ATOM_type(m)) - fsp ) * cell % ATOM_pos(m,4) * exp(-cell % ATOM_pos(m,5)*s)
  ! write (*,*) 'atom type ',cell % ATOM_type(m),'; s = ',sqrt(s), &
  !    '; DW ',cell % ATOM_pos(m,5),-cell % ATOM_pos(m,5)*s,exp(-cell % ATOM_pos(m,5)*s)

! loop over all atoms in the orbit
  do j=1,cell%numat(m)
   arg=twopi * sum(hkl(1:3)*cell%apos(m,j,1:3))
   sf = sf + fs * cmplx(cos(arg),-sin(arg))
  end do

 end do ! m loop

! and fill in just two entries of the rlp variable
 rlp%hkl = hkl
 rlp%Ucg = sf
 
end if

!----------------------------------
! for Doyle-Turner scattering factors
if (rlp%method.eq.'DT') then 
! initialize the real and imaginary parts of the structure factor
 sf = czero

! compute the prefactor (this also scales from Angstrom to nm)
 pref = 0.04787801/cell % vol

! loop over all atoms in the asymmetric unit
 do m=1,cell % ATOM_ntype

! get the atomic scattering factor for this atom
  sct=0.0
  j=cell % ATOM_type(m)
  do ii=1,4
   sct=sct+scatfac(ii,j)*exp(-scatfac(ii+4,j)*s)
  end do

! scale and include Debye-Waller factor
! and site occupation parameter
  fs=pref*sct*exp(-cell % ATOM_pos(m,5)*s)*cell % ATOM_pos(m,4)

! loop over all atoms in the orbit
  do j=1,cell%numat(m)
   arg=twopi*sum(hkl(1:3)*cell%apos(m,j,1:3))
   sf = sf + fs*exp(cmplx(0.0,-arg))
  end do
 end do ! m loop

! and fill in the entries of the rlp variable
 pre = 2.0*sngl(cRestmass*cCharge/cPlanck**2)*1.0E-18
 rlp%hkl = hkl
 rlp%Vmod = cabs(sf)*cell%mRelcor
 rlp%Vphase = atan2(aimag(sf),real(sf))
 rlp%Vpmod = 0.0
 rlp%Vpphase = 0.0
 if (rlp%Vmod.gt.0.0) then
  rlp%xg = 1.0/(pre*rlp%Vmod*cell%mLambda)
 else
  rlp%xg = 1.0E+8
 end if
 rlp%xgp = 0.0
 rlp%ar = 0.0
 rlp%Vg = rlp%Vmod * exp(cmplx(0.0,rlp%Vphase))
 rlp%Ucg = pre*rlp%Vg
 rlp%qg = cmplx(1.0/rlp%xg,0.0)
end if


!----------------------------------
if (rlp%method.eq.'WK') then 
! The Weickenmeier-Kohl (WK) subroutine works in Angstrom, and also 
! scales reciprocal space by a factor of 2*pi;  this scaling
! is accomplished by changing the g-value in nm^{-1} by a 
! scaling factor swk = 2*pi/10, to go from book units to WK units.
!
! A similar scaling must be performed on the Debye Waller factor;
! the book defines it as exp(-Bs^2), with s in [nm^2]; WK define
! it in A^2, and with a scaled reciprocal space.  The conversion
! factor dwwk = 100.0/8*pi^2
!
! To go from the standard B factor in [nm^2] to ul^2 in A^2,
! ul = sqrt(B*dwwk)
 swk = 0.1*twopi
 dwwk = 100.0/(8.0*cPi**2)
  
! properly scale the scattering parameter
 s = rlp%g*swk

! let fscatt perform the relativistic corrections for f_g and fprime_g
 accflg = .TRUE.

! include absorption ?
 absflg = 0
 if (rlp%absorption.eqv..TRUE.) absflg = 3  ! include phonon and core contributions

! always include Debye-Waller factor
 dwflg  = .TRUE.

! compute the scaling prefactors
! pref contains A to nm conversion, and divides by 4pi
 pref = 0.04787801/cell % vol/(4.0*cPi) 

! preg is used to go from V to U, remembering that gamma is already
! included in the output from fscatt
 pre = pref * preg

! initialize the real and imaginary parts of the structure factor
 ff=czero
 gg=czero

! loop over all atoms in the asymmetric unit
 do m=1,cell % ATOM_ntype
! get the atomic scattering factor for this atom
! scale and include Debye-Waller factor and site occupation parameter
  ul = sqrt(cell % ATOM_pos(m,5)*dwwk)
  j = cell % ATOM_type(m)
  sf = FSCATT(s,ul,j,smb,sngl(cell%voltage),absflg,accflg,dwflg)*cmplx(cell%ATOM_pos(m,4),0.0)

! loop over all atoms in the orbit
  p1 = czero
  do j=1,cell%numat(m)
   arg=twopi*sum(float(hkl(1:3))*cell%apos(m,j,1:3))
   p1 = p1 + exp(cmplx(0.0,-arg))
  end do

  ff = ff + p1*real(sf)
  gg = gg + p1*aimag(sf)

 end do
!
! fill in the entries of the rlp variable
 rlp%hkl = hkl

! these are the modulus and phase of the real part of Vg
 rlp%Vmod = pref * cabs(ff)
 rlp%Vphase = atan2(aimag(ff),real(ff))

! modulus of U_g
 rlp%Umod = preg*rlp%Vmod

! if absorption is included, also compute the imaginary part of Vg, i.e., Vprime_g
 if (rlp%absorption.eqv..TRUE.) then 
  rlp%Vpmod = pref * cabs(gg)
  rlp%Vpphase = atan2(aimag(gg),real(gg))

! modulus of Uprime_g
  rlp%Upmod = preg*rlp%Vpmod

! complex Ucg = U_g + i Uprime_g = U_g,r-Uprime_g,i + i(U_g,i+Uprime_g,r)
  rlp%Ucg = pre * cmplx(real(ff)-aimag(gg),aimag(ff)+real(gg))
 else ! set absorption parameters to zero
  rlp%Vpmod = 0.0
  rlp%Vpphase = 0.0
! Ucg = U_g (complex number)
  rlp%Ucg = pre * ff
 end if

! complex Vg 
 rlp%Vg = rlp%Ucg/preg
 if (abs(rlp%Umod).gt.0.0) then 
  rlp%xg = 1.0/abs(rlp%Umod)/cell%mLambda
 else
  rlp%xg = 1.0E+8
 end if 

 if (abs(rlp%Upmod).gt.0.0) then 
  rlp%xgp = 1.0/abs(rlp%Upmod)/cell%mLambda
 else
  rlp%xgp = 1.0E+8
 end if 

 if (rlp%absorption.eqv..TRUE.) then 
  rlp%ar = rlp%xgp/rlp%xg
  if (present(applyqgshift)) then
    if (applyqgshift.eqv..TRUE.) then
      rlp%qg = cmplx(cos(rlp%Vphase)/rlp%xg-sin(rlp%Vpphase)/rlp%xgp,cos(rlp%Vpphase)/rlp%xgp+sin(rlp%Vphase)/rlp%xg)
    end if
  else
    arg = rlp%Vpphase-rlp%Vphase
    rlp%qg = cmplx(1.0/rlp%xg-sin(arg)/rlp%xgp,cos(arg)/rlp%xgp)
  end if
 else
  rlp%ar = 0.0
  rlp%qg = cmplx(1.0/rlp%xg,0.0)
 end if

end if

!----------------------------------
if (rlp%method.eq.'IP') then 
 allocate(sfarray(cell%ATOM_ntype))

! The Weickenmeier-Kohl (WK) scattering parameters have been pre-calculated 
! so all we need to do is linear interpolation to get the correct value
 swk = 0.1*twopi
  
! properly scale the scattering parameter
 s = rlp%g*swk

! get the atomic scattering factors for all atom types by linear interpolation
 call getScatfac(cell, s, sfarray, cell%ATOM_ntype)

! compute the scaling prefactors
! pref contains A to nm conversion, and divides by 4pi
 pref = 0.04787801/cell % vol/(4.0*cPi) 

! preg is used to go from V to U, remembering that gamma is already
! included in the output from fscatt
 pre = pref * preg

! initialize the real and imaginary parts of the structure factor
 ff=czero
 gg=czero

! loop over all atoms in the asymmetric unit
 do m=1,cell % ATOM_ntype
  sf = sfarray(m)

! loop over all atoms in the orbit
  p1 = czero
  do j=1,cell%numat(m)
   arg=twopi*sum(float(hkl(1:3))*cell%apos(m,j,1:3))
   p1 = p1 + exp(cmplx(0.0,-arg))
  end do

  ff = ff + p1*real(sf)
  gg = gg + p1*aimag(sf)

 end do
!
! fill in the entries of the rlp variable
 rlp%hkl = hkl

! these are the modulus and phase of the real part of Vg
 rlp%Vmod = pref * cabs(ff)
 rlp%Vphase = atan2(aimag(ff),real(ff))

! modulus of U_g
 rlp%Umod = preg*rlp%Vmod

! if absorption is included, also compute the imaginary part of Vg, i.e., Vprime_g
 if (rlp%absorption.eqv..TRUE.) then 
  rlp%Vpmod = pref * cabs(gg)
  rlp%Vpphase = atan2(aimag(gg),real(gg))

! modulus of Uprime_g
  rlp%Upmod = preg*rlp%Vpmod

! complex Ucg = U_g + i Uprime_g = U_g,r-Uprime_g,i + i(U_g,i+Uprime_g,r)
  rlp%Ucg = pre * cmplx(real(ff)-aimag(gg),aimag(ff)+real(gg))
 else ! set absorption parameters to zero
  rlp%Vpmod = 0.0
  rlp%Vpphase = 0.0
! Ucg = U_g (complex number)
  rlp%Ucg = pre * ff
 end if

! complex Vg 
 rlp%Vg = rlp%Ucg/preg
 if (abs(rlp%Umod).gt.0.0) then 
  rlp%xg = 1.0/abs(rlp%Umod)/cell%mLambda
 else
  rlp%xg = 1.0E+8
 end if 

 if (abs(rlp%Upmod).gt.0.0) then 
  rlp%xgp = 1.0/abs(rlp%Upmod)/cell%mLambda
 else
  rlp%xgp = 1.0E+8
 end if 

 if (rlp%absorption.eqv..TRUE.) then 
  rlp%ar = rlp%xgp/rlp%xg
  if (present(applyqgshift)) then
    if (applyqgshift.eqv..TRUE.) then
      rlp%qg = cmplx(cos(rlp%Vphase)/rlp%xg-sin(rlp%Vpphase)/rlp%xgp,cos(rlp%Vpphase)/rlp%xgp+sin(rlp%Vphase)/rlp%xg)
    end if
  else
    arg = rlp%Vpphase-rlp%Vphase
    rlp%qg = cmplx(1.0/rlp%xg-sin(arg)/rlp%xgp,cos(arg)/rlp%xgp)
  end if
 else
  rlp%ar = 0.0
  rlp%qg = cmplx(1.0/rlp%xg,0.0)
 end if

 deallocate(sfarray)
end if


end subroutine CalcUcg


!--------------------------------------------------------------------------
!
! SUBROUTINE: PreCalcFSCATT
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief precompute the FSCATT values for interpolation purposes, to speed up STEM-DCI and other codes
!
!> @param cell unit cell pointer
!> @param dmin smallest d-spacing to consider
!> @param gstep step size in the cell%scatfacg array
!
!> @date   08/09/18 MDG 1.0 original
!--------------------------------------------------------------------------
recursive subroutine PreCalcFSCATT(cell, dmin, gstep)
!DEC$ ATTRIBUTES DLLEXPORT :: PreCalcFSCATT

use crystal
!use symmetry
use constants
use others

IMPLICIT NONE

type(unitcell)                  :: cell
real(kind=sgl),INTENT(IN)       :: dmin
real(kind=sgl),INTENT(IN)       :: gstep

integer(kind=irg)               :: j,m,ii,i
real(kind=sgl)                  :: s,ul
real(kind=sgl),parameter        :: swk = 0.628318530717959
real(kind=sgl),parameter        :: dwwk = 1.26651479552922
integer(kind=irg),parameter     :: absflg = 3
logical                         :: accflg=.TRUE., dwflg=.TRUE.
character(2)                    :: smb

! first generate the array of s-values for which the scattering factors need to be computed
s = 2.0/dmin   ! maximum range in reciprocal space
cell%numscatfac = nint(s/gstep) + 2
allocate(cell%scatfacg(cell%numscatfac))
cell%scatfacg = (/ (gstep * float(i-1),i=1,cell%numscatfac) /)
cell%scatfacg = cell%scatfacg * swk

! allocate the scattering factor interpolation array
allocate( cell%scatfac(cell%numscatfac, cell % ATOM_ntype) )

! The Weickenmeier-Kohl (WK) subroutine works in Angstrom, and also 
! scales reciprocal space by a factor of 2*pi;  this scaling
! is accomplished by changing the g-value in nm^{-1} by a 
! scaling factor swk = 2*pi/10, to go from book units to WK units.
!
! A similar scaling must be performed on the Debye Waller factor;
! the book defines it as exp(-Bs^2), with s in [nm^2]; WK define
! it in A^2, and with a scaled reciprocal space.  The conversion
! factor dwwk = 100.0/8*pi^2
!
! To go from the standard B factor in [nm^2] to ul^2 in A^2,
! ul = sqrt(B*dwwk)
  
do i=1,cell%numscatfac
! properly scale the scattering parameter
 s = cell%scatfacg(i)

! loop over all atoms in the asymmetric unit
 do m=1,cell % ATOM_ntype
! get the atomic scattering factor for this atom
! scale and include Debye-Waller factor and site occupation parameter
  ul = sqrt(cell % ATOM_pos(m,5)*dwwk)
  j = cell % ATOM_type(m)
  cell%scatfac(i,m) = FSCATT(s,ul,j,smb,sngl(cell%voltage),absflg,accflg,dwflg)*cell%ATOM_pos(m,4)
 end do 
end do 

end subroutine PreCalcFSCATT

!--------------------------------------------------------------------------
!
! SUBROUTINE: getScatfac
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief interpolate the precomputed FSCATT values 
!
!> @param cell unit cell pointer
!> @param s reciprocal distance value
!> @param sfarray returned scattering factor values
!
!> @date   08/09/18 MDG 1.0 original
!--------------------------------------------------------------------------
recursive subroutine getScatfac(cell, s, sfarray, ntypes)
!DEC$ ATTRIBUTES DLLEXPORT :: getScatfac

use crystal
!use symmetry
use constants
use others

IMPLICIT NONE

type(unitcell)                  :: cell
real(kind=sgl),INTENT(IN)       :: s
integer(kind=irg),INTENT(IN)    :: ntypes
complex(kind=sgl),INTENT(OUT)   :: sfarray(ntypes)

integer(kind=irg)               :: jj
real(kind=sgl)                  :: dx

if (s.eq.0.0) then 
    sfarray(1:ntypes) = cell%scatfac(1,1:ntypes)
else
    jj = ifix(s/cell%scatfacg(2))
    if (jj.ge.cell%numscatfac) then
        sfarray(1:ntypes) = cell%scatfac(cell%numscatfac,1:ntypes)
    else
        dx = s/cell%scatfacg(2) - float(jj)
        sfarray(1:ntypes) = cell%scatfac(jj,1:ntypes)*(1.0-dx) + &
                                     cell%scatfac(jj+1,1:ntypes)*dx
    end if
end if

end subroutine getScatfac


!--------------------------------------------------------------------------
!
! FUNCTION: CalcsgSingle
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief compute the excitation error for a given reflection
!
!> @param cell unit cell pointer
!> @param gg reciprocal lattice point indices
!> @param kk wave vector components
!> @param FN foil normal
!
!> @date   10/20/98 MDG 1.0 original
!> @date    5/22/01 MDG 2.0 f90
!> @date   11/27/01 MDG 2.1 added kind support
!> @date   03/26/13 MDG 3.0 updated IO
!> @date   06/09/14 MDG 4.0 added cell as argument 
!--------------------------------------------------------------------------
recursive function CalcsgSingle(cell,gg,kk,FN) result(sg)
!DEC$ ATTRIBUTES DLLEXPORT :: CalcsgSingle

use crystal

IMPLICIT NONE

type(unitcell)                  :: cell
real(kind=sgl),INTENT(IN)       :: gg(3)                !< reciprocal lattice point
real(kind=sgl),INTENT(IN)       :: kk(3)                !< wave vector
real(kind=sgl),INTENT(IN)       :: FN(3)                !< foil normal

real(kind=sgl)                  :: kpg(3),tkpg(3),xnom,xden,q1,q2,sg


 kpg=kk+gg
 tkpg=2.0*kk+gg

! use equation of Ewald sphere
 xnom = -CalcDot(cell,gg,tkpg,'r')

! 2|k0+g|cos(alpha) = 2(k0+g).Foilnormal
 q1 = CalcLength(cell,kpg,'r')
 q2 = CalcAngle(cell,kpg,FN,'r')
 xden = 2.0*q1*cos(q2)
 sg = xnom/xden

end function CalcsgSingle

!--------------------------------------------------------------------------
!
! FUNCTION: CalcsgDouble
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief compute the excitation error for a given reflection (double precision)
!
!> @param cell unit cell pointer
!> @param gg reciprocal lattice point indices
!> @param kk wave vector components
!> @param FN foil normal
!
!> @date   10/20/98 MDG 1.0 original
!> @date    5/22/01 MDG 2.0 f90
!> @date   11/27/01 MDG 2.1 added kind support
!> @date   03/26/13 MDG 3.0 updated IO
!> @date   06/09/14 MDG 4.0 added cell as argument
!--------------------------------------------------------------------------
recursive function CalcsgDouble(cell,gg,kk,FN) result(sg)
!DEC$ ATTRIBUTES DLLEXPORT :: CalcsgDouble

use crystal

IMPLICIT NONE

type(unitcell)                  :: cell
real(kind=dbl),INTENT(IN)       :: gg(3)                !< reciprocal lattice point
real(kind=dbl),INTENT(IN)       :: kk(3)                !< wave vector
real(kind=dbl),INTENT(IN)       :: FN(3)                !< foil normal

real(kind=dbl)                  :: kpg(3),tkpg(3),xnom,xden,q1,q2,sg

 kpg=kk+gg
 tkpg=2.D0*kk+gg

! use equation of Ewald sphere
 xnom = -CalcDot(cell,gg,tkpg,'r')

! 2|k0+g|cos(alpha) = 2(k0+g).Foilnormal
 q1 = CalcLength(cell,kpg,'r')
 q2 = CalcAngle(cell,kpg,FN,'r')
 xden = 2.D0*q1*dcos(q2)
 sg = xnom/xden

end function CalcsgDouble

end module diffraction
