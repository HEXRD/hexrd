! ###################################################################
! Copyright (c) 2013-2019, Marc De Graef Research Group/Carnegie Mellon University
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
! EMsoft:math.f90
!--------------------------------------------------------------------------
!
! MODULE: math
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief collection of mathematical/numerical routines that don't fit anywhere else
! 
!> @date 10/13/98 MDG 1.0 original
!> @date 05/19/01 MDG 2.0 f90
!> @date 11/27/01 MDG 2.1 added kind support
!> @date 03/19/13 MDG 3.0 updated all routines
!> @date 11/13/13 MDG 4.0 added MatrixExponential routine
!> @date 11/23/15 MDG 4.1 moved several routines from other mods into this one
!> @date 10/24/17 MDG 4.2 added infty()/inftyd() functions to return the IEEE infinity value
!> @date 08/23/19 MDG 4.3 removed spaces around "kind" statements to facilitate f90wrap python wrapper generation
!> @date 10/04/19 MDG 4.4 adds vecnorm to replace non-standard NORM2 calls  (F2003 compliance)
!> @date 10/04/19 MDG 4.5 adds nan() function, returning a single or double precision IEEE NaN value
!--------------------------------------------------------------------------
! ###################################################################
!  

module math

use local

public :: mInvert, cross3, infty, inftyd

interface mInvert
        module procedure mInvert
        module procedure mInvert_d
end interface

interface cross3
        module procedure cross3
        module procedure cross3_d
end interface 

interface vecnorm
        module procedure vecnorm 
        module procedure vecnorm_d
        module procedure vecnorm2 
        module procedure vecnorm2_d
end interface

contains


!--------------------------------------------------------------------------
!
! FUNCTION: vecnorm
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief return the single precision length of a 1D vector
!
!> @date  10/04/19 MDG 1.0 original
!--------------------------------------------------------------------------
recursive function vecnorm(vec) result(veclen)
!DEC$ ATTRIBUTES DLLEXPORT :: vecnorm

real(kind=sgl),INTENT(IN)        :: vec(:)
real(kind=sgl)                   :: veclen

integer(kind=irg)                :: sz(1)

sz = size(vec)

veclen = sqrt(sum(vec(1:sz(1))*vec(1:sz(1))))

end function vecnorm

!--------------------------------------------------------------------------
!
! FUNCTION: vecnorm_d
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief return the double precision length of a 1D vector
!
!> @date  10/04/19 MDG 1.0 original
!--------------------------------------------------------------------------
recursive function vecnorm_d(vec) result(veclen)
!DEC$ ATTRIBUTES DLLEXPORT :: vecnorm_d

real(kind=dbl),INTENT(IN)       :: vec(:)
real(kind=dbl)                  :: veclen

integer(kind=irg)               :: sz(1)

sz = size(vec)

veclen = sqrt(sum(vec(1:sz(1))*vec(1:sz(1))))

end function vecnorm_d

!--------------------------------------------------------------------------
!
! FUNCTION: vecnorm2
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief return the single precision length of a 2D array
!
!> @date  10/04/19 MDG 1.0 original
!--------------------------------------------------------------------------
recursive function vecnorm2(vec) result(veclen)
!DEC$ ATTRIBUTES DLLEXPORT :: vecnorm2

real(kind=sgl),INTENT(IN)        :: vec(:,:)
real(kind=sgl)                   :: veclen

integer(kind=irg)                :: sz(2)

sz = size(vec)

veclen = sqrt(sum(vec(1:sz(1),1:sz(2))*vec(1:sz(1),1:sz(2))))

end function vecnorm2

!--------------------------------------------------------------------------
!
! FUNCTION: vecnorm2_d
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief return the double precision length of a 2D array
!
!> @date  10/04/19 MDG 1.0 original
!--------------------------------------------------------------------------
recursive function vecnorm2_d(vec) result(veclen)
!DEC$ ATTRIBUTES DLLEXPORT :: vecnorm2_d

real(kind=dbl),INTENT(IN)        :: vec(:,:)
real(kind=dbl)                   :: veclen

integer(kind=irg)                :: sz(2)

sz = size(vec)

veclen = sqrt(sum(vec(1:sz(1),1:sz(2))*vec(1:sz(1),1:sz(2))))

end function vecnorm2_d


!--------------------------------------------------------------------------
!
! FUNCTION: infty
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief return the single precision IEEE value for infinity
!
!> @date  10/24/17 MDG 1.0 original
!--------------------------------------------------------------------------
recursive function infty() result(infinity)
!DEC$ ATTRIBUTES DLLEXPORT :: infty

real(kind=sgl)      :: infinity
real(kind=sgl)      :: big 

big = HUGE(1.0)
infinity = big + HUGE(1.0)

end function infty


!--------------------------------------------------------------------------
!
! FUNCTION: inftyd
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief return the double precision IEEE value for infinity
!
!> @date  10/24/17 MDG 1.0 original
!--------------------------------------------------------------------------
recursive function inftyd() result(infinity)
!DEC$ ATTRIBUTES DLLEXPORT :: inftyd

real(kind=dbl)      :: infinity
real(kind=dbl)      :: big 

big = HUGE(1.D0)
infinity = big + HUGE(1.D0)

end function inftyd

!--------------------------------------------------------------------------
!
! SUBROUTINE: getPolarDecomposition
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief Use LAPACK routines to compute the polar decomposition of a real 3x3 matrix
!
!> @param F input matrix
!> @param Rmatrix (output) unitary matrix
!> @param Smatrix (output) symmetric stretch matrix
!
!> @date  09/29/17 MDG 1.0 original
!--------------------------------------------------------------------------
recursive subroutine getPolarDecomposition(F, Rmatrix, Smatrix)
!DEC$ ATTRIBUTES DLLEXPORT :: getPolarDecomposition

use local
use error
use io

IMPLICIT NONE

real(kind=dbl),INTENT(IN)     :: F(3,3)           !< input matrix
real(kind=dbl),INTENT(OUT)    :: Rmatrix(3,3)     !< output unitary matrix
real(kind=dbl),INTENT(OUT)    :: Smatrix(3,3)     !< output symmetric stretch matrix

! various parameters needed by the LAPACK routine
integer(kind=irg)               :: INFO, LDA, LDU, LDVT, LWORK, M, N, i
integer(kind=irg),parameter     :: LWMAX = 100 
real(kind=dbl)                  :: A(3,3), WORK(LWMAX), S(3), U(3,3), VT(3,3), Sm(3,3)
character                       :: JOBU, JOBVT

! set initial LAPACK variables
JOBU = 'A'
JOBVT = 'A'
M = 3
N = 3
LDA = 3
A = F
LDU = 3
LDVT = 3

S = 0.D0
U = 0.D0
VT = 0.D0

LWORK = LWMAX

call dgesvd(JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO)
if (INFO.ne.0) call FatalError('Error in getPolarDecomposition: ','DGESVD return not zero')
Sm = 0.D0
Sm(1,1) = S(1)
Sm(2,2) = S(2)
Sm(3,3) = S(3)

! next use these matrices to compute the polar decomposition
Rmatrix = matmul(U, VT)
Smatrix = matmul(transpose(VT),matmul(Sm, VT))

end subroutine getPolarDecomposition


!--------------------------------------------------------------------------
!
! SUBROUTINE:mInvert_d
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief Invert a 3x3 matrix
!
!> @details  Invert a 3x3 matrix; if unitary, simply transpose
!
!> @param a input matrix
!> @param b output matrix
!> @param uni .TRUE. if unitary matrix, .FALSE. otherwise
!
!> @todo this should really be replaced by a BLAS call
! 
!> @date   10/13/98 MDG 1.0 original
!> @date    4/ 5/00 MDG 1.1 added inverse of unitary matrix
!> @date    5/19/01 MDG 2.0 f90
!> @date   11/27/01 MDG 2.1 added kind support
!
!--------------------------------------------------------------------------
recursive subroutine mInvert_d(a,b,uni)
!DEC$ ATTRIBUTES DLLEXPORT :: mInvert_d

use error

IMPLICIT NONE

real(kind=dbl),INTENT(IN)               :: a(3,3)               !< input matrix
real(kind=dbl),INTENT(OUT)              :: b(3,3)               !< output matrix
logical,INTENT(IN)                      :: uni                  !< unitary logical
real(kind=dbl)                          :: d                    !< auxiliary variable

! it is a regular (non-unitary) matrix
 if (.not.uni) then 
  d = a(1,1)*a(2,2)*a(3,3)+a(1,2)*a(2,3)*a(3,1)+ &
         a(1,3)*a(2,1)*a(3,2)-a(1,3)*a(2,2)*a(3,1)- &
         a(1,2)*a(2,1)*a(3,3)-a(1,1)*a(2,3)*a(3,2)
  if (d.ne.0.D0) then
   b(1,1)=a(2,2)*a(3,3)-a(2,3)*a(3,2)
   b(1,2)=a(1,3)*a(3,2)-a(1,2)*a(3,3)
   b(1,3)=a(1,2)*a(2,3)-a(1,3)*a(2,2)
   b(2,1)=a(2,3)*a(3,1)-a(2,1)*a(3,3)
   b(2,2)=a(1,1)*a(3,3)-a(1,3)*a(3,1)
   b(2,3)=a(1,3)*a(2,1)-a(1,1)*a(2,3)
   b(3,1)=a(2,1)*a(3,2)-a(2,2)*a(3,1)
   b(3,2)=a(1,2)*a(3,1)-a(1,1)*a(3,2)
   b(3,3)=a(1,1)*a(2,2)-a(1,2)*a(2,1)
   b = b/d
  else
   call FatalError('mInvert','matrix has zero determinant')
  end if
 else
! it is a unitary matrix, so simply get the transpose
  b = transpose(a)
 endif

end subroutine mInvert_d
      
!--------------------------------------------------------------------------
!
! SUBROUTINE:mInvert
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief Invert a 3x3 matrix
!
!> @details  Invert a 3x3 matrix; if unitary, simply transpose
!
!> @param a input matrix
!> @param b output matrix
!> @param uni .TRUE. if unitary matrix, .FALSE. otherwise
!
!> @todo this should really be replaced by a BLAS call
! 
!> @date   10/13/98 MDG 1.0 original
!> @date    4/ 5/00 MDG 1.1 added inverse of unitary matrix
!> @date    5/19/01 MDG 2.0 f90
!> @date   11/27/01 MDG 2.1 added kind support
!
!--------------------------------------------------------------------------
recursive subroutine mInvert(a,b,uni)
!DEC$ ATTRIBUTES DLLEXPORT :: mInvert

use error

IMPLICIT NONE

real(kind=sgl),INTENT(IN)               :: a(3,3)               !< input matrix
real(kind=sgl),INTENT(OUT)              :: b(3,3)               !< output matrix
logical,INTENT(IN)                      :: uni                  !< unitary logical
real(kind=sgl)                          :: d                    !< auxiliary variable

! it is a regular (non-unitary) matrix
 if (.not.uni) then 
  d = a(1,1)*a(2,2)*a(3,3)+a(1,2)*a(2,3)*a(3,1)+ &
         a(1,3)*a(2,1)*a(3,2)-a(1,3)*a(2,2)*a(3,1)- &
         a(1,2)*a(2,1)*a(3,3)-a(1,1)*a(2,3)*a(3,2)
  if (d.ne.0.0) then
   b(1,1)=a(2,2)*a(3,3)-a(2,3)*a(3,2)
   b(1,2)=a(1,3)*a(3,2)-a(1,2)*a(3,3)
   b(1,3)=a(1,2)*a(2,3)-a(1,3)*a(2,2)
   b(2,1)=a(2,3)*a(3,1)-a(2,1)*a(3,3)
   b(2,2)=a(1,1)*a(3,3)-a(1,3)*a(3,1)
   b(2,3)=a(1,3)*a(2,1)-a(1,1)*a(2,3)
   b(3,1)=a(2,1)*a(3,2)-a(2,2)*a(3,1)
   b(3,2)=a(1,2)*a(3,1)-a(1,1)*a(3,2)
   b(3,3)=a(1,1)*a(2,2)-a(1,2)*a(2,1)
   b = b/d
  else
   call FatalError('mInvert','matrix has zero determinant')
  end if
 else
! it is a unitary matrix, so simply get the transpose
  b = transpose(a)
 endif

end subroutine mInvert

!--------------------------------------------------------------------------
!
! SUBROUTINE:cInvert
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief Invert a 3x3 complex matrix
!
!> @param a input matrix
!> @param b output matrix
!
!> @todo this should really be replaced by a BLAS call
! 
!> @date   10/13/98 MDG 1.0 original
!> @date    4/ 5/00 MDG 1.1 added inverse of unitary matrix
!> @date    5/19/01 MDG 2.0 f90
!> @date   11/27/01 MDG 2.1 added kind support
!
!--------------------------------------------------------------------------
recursive subroutine cInvert(a,b)
!DEC$ ATTRIBUTES DLLEXPORT :: cInvert

use error

IMPLICIT NONE

complex(kind=dbl),INTENT(IN)            :: a(3,3)               !< input matrix
complex(kind=dbl),INTENT(OUT)           :: b(3,3)               !< output matrix
complex(kind=dbl)                                       :: d                    !< auxiliary variable

  d = a(1,1)*a(2,2)*a(3,3)+a(1,2)*a(2,3)*a(3,1)+ &
      a(1,3)*a(2,1)*a(3,2)-a(1,3)*a(2,2)*a(3,1)- &
      a(1,2)*a(2,1)*a(3,3)-a(1,1)*a(2,3)*a(3,2)
  if (abs(d).ne.0.D0) then
   b(1,1)=a(2,2)*a(3,3)-a(2,3)*a(3,2)
   b(1,2)=a(1,3)*a(3,2)-a(1,2)*a(3,3)
   b(1,3)=a(1,2)*a(2,3)-a(1,3)*a(2,2)
   b(2,1)=a(2,3)*a(3,1)-a(2,1)*a(3,3)
   b(2,2)=a(1,1)*a(3,3)-a(1,3)*a(3,1)
   b(2,3)=a(1,3)*a(2,1)-a(1,1)*a(2,3)
   b(3,1)=a(2,1)*a(3,2)-a(2,2)*a(3,1)
   b(3,2)=a(1,2)*a(3,1)-a(1,1)*a(3,2)
   b(3,3)=a(1,1)*a(2,2)-a(1,2)*a(2,1)
   b = b/d
  else
   call FatalError('cInvert','Matrix has complex zero determinant')
  end if

end subroutine cInvert


!--------------------------------------------------------------------------
!
! SUBROUTINE: MatrixExponential
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief compute the exponential of a dynamical matrix
!
!> @details This routine uses two different methods, one based on the Taylor
!> expansion, the other on Pade approximants, both with "scaling & squaring".
!> Currently, the routine targets an accuracy level of 10^{-9}.
!> This routine uses table 1 in "Nineteen dubious ways to compute the exponential of a matrix,
!> twenty-five years later", C. Moler, C. Van Loan, SIAM Review, 45, 1 (2003)
!
!> @param A input matrix
!> @param E output matrix
!> @param z0 slice thickness in [nm]
!> @param TP 'Tayl' or 'Pade', to select method
!> @param nn number of row/column entries in A
!
!> @date 09/16/13 MDG 1.0 original, tested against analytical version for small array
!> @date 06/05/14 MDG 1.1 updated IO
!--------------------------------------------------------------------------
recursive subroutine MatrixExponential(A,E,z0,TP,nn)
!DEC$ ATTRIBUTES DLLEXPORT :: MatrixExponential

use io
use error

IMPLICIT NONE

integer(kind=irg),INTENT(IN)            :: nn
complex(kind=dbl),INTENT(IN)            :: A(nn,nn)
complex(kind=dbl),INTENT(OUT)           :: E(nn,nn)
real(kind=dbl),INTENT(IN)               :: z0
character(4),INTENT(IN)         :: TP

real(kind=dbl)                          :: modA, pref, sgn
complex(kind=dbl),allocatable           :: B(:,:), add(:,:), Nqq(:,:), Dqq(:,:), C(:,:)

integer(kind=irg)                       :: i, k, j, icnt, q, istat, ilev

integer(kind=irg)                       :: INFO, LDA, MILWORK
integer(kind=irg),allocatable           :: JPIV(:)
complex(kind=dbl),allocatable           :: MIWORK(:)

integer(kind=irg),parameter             :: kTaylor(6) = (/ 3, 4, 6, 8, 7, 6 /)          ! from table 1 in reference above
integer(kind=irg),parameter             :: jTaylor(6) = (/ 1, 2, 3, 5, 9, 13 /)
integer(kind=irg),parameter             :: qPade(6) = (/ 2, 3, 4, 4, 4, 4 /)
integer(kind=irg),parameter             :: jPade(6) = (/ 0, 0, 1, 5, 8, 11 /)

! set output array to zero
E = cmplx(0.D0,0.D0)

!get the max( ||A|| ) value and determine index into (k,j) or (q,j) pairs
modA = maxval(abs(A)*z0)
ilev = nint(log10(modA))+3
if (ilev.le.0) ilev = 1         ! can not be smaller than 1

! if modA gets to be too large, abort with a message
if (modA.gt.10000.D0) call FatalError('MatrixExponential','  routine can not deal with ||A|| > 10000.0')

if (TP.eq.'Tayl') then ! use scaling and squaring for the Taylor expansion
        k = kTaylor(ilev)
        j = jTaylor(ilev)

        ! allocate an auxiliary array
        allocate( B(nn,nn), add(nn,nn), stat=istat )
        if (istat.ne.0) call FatalError('MatrixExponential',' Error allocating arrays for Taylor approximation')
        
        ! perform the scaling step
        B = (A * z0) / 2.0D0**j ! cmplx(2.0**j,0.0)
        
        ! initialize the diagonal of E
        forall (i=1:nn) E(i,i) = cmplx(1.0D0,0.D0)
        
        ! loop over the Taylor series
        add = B
        E = E + add
        do icnt=2,k
          add = matmul( add, B/cmplx(icnt,0) )
          E = E + add
        end do
        
        ! and deallocate the auxiliary arrays
        deallocate(add, B)

else ! Pade approximation for target accuracy 10^(-9)
        q = qPade(ilev)
        j = jPade(ilev)

        ! allocate auxiliary arrays
        allocate(B(nn,nn),C(nn,nn), Nqq(nn,nn), Dqq(nn,nn), stat=istat )
        if (istat.ne.0) call FatalError('MatrixExponential',' Error allocating arrays for Pade approximation')
        
        ! perform the scaling step
        B = (A * z0) / 2.D0**j  ! cmplx(2.0**j,0.0)
        C = B
                
        ! initialize the diagonal of both arrays
        Nqq = cmplx(0.D0,0.D0)
        forall (i=1:nn) Nqq(i,i) = cmplx(1.0D0,0.D0)
        Dqq = Nqq

        ! init some constants
        pref = 1.D0
        sgn = -1.D0
        
        ! and loop
        do icnt=1,q
          pref = pref * dble(q-icnt+1) / dble(icnt) / dble(2*q-icnt+1)
          Nqq = Nqq + pref * C
          Dqq = Dqq + sgn * pref * C
          sgn = -sgn
          C = matmul( C, B )
        end do
        
        ! get the inverse of Dqq using the LAPACK routines zgetrf and zgetri
        LDA = nn
        allocate( JPIV(nn) )
        call zgetrf(nn,nn,Dqq,LDA,JPIV,INFO)
        if (INFO.ne.0) call FatalError('Error in MatrixExponential: ',' ZGETRF return not zero')

        MILWORK = 64*nn 
        allocate(MIWORK(MILWORK))

        MIWORK = cmplx(0.0_dbl,0.0_dbl)
        call zgetri(nn,Dqq,LDA,JPIV,MIWORK,MILWORK,INFO)
        if (INFO.ne.0) call FatalError('Error in MatrixExponential: ',' ZGETRI return not zero')

        ! and compute E
        E = matmul( Dqq, Nqq )
        
        ! clean up
        deallocate(Nqq, Dqq, C, B, JPIV, MIWORK)
end if

! and finally compute the power 2^j of the matrix E (i.e. the squaring step)
do icnt = 1,j
  E = matmul( E, E )
end do

end subroutine MatrixExponential

!--------------------------------------------------------------------------
!
! SUBROUTINE: TransFourthRankTensor 
!
!> @author Marc De Graef, Carnegie Mellon University
!
!> @brief  transform a fourth rank tensor using a given transformation matrix
! 
!> @note This is one of the very few places in this package where we still use the 
!> good old-fashioned rotation matrix instead of quaternions... Note also that we
!> use the 6x6 notation for the tensors, so we need to convert them to real tensor
!> notation before carrying out the rotations.
!
!> @param al rotation matrix
!> @param cin unrotated tensor
!> @param cout rotated tensor
! 
!> @date 1/5/99   MDG 1.0 original
!> @date    5/19/01 MDG 2.0 f90 version
!> @date   11/27/01 MDG 2.1 added kind support
!> @date   06/04/13 MDG 3.0 rewrite
!--------------------------------------------------------------------------
recursive subroutine TransFourthRankTensor(al,cin,cout)
!DEC$ ATTRIBUTES DLLEXPORT :: TransFourthRankTensor

IMPLICIT NONE

real(kind=dbl),INTENT(IN)       :: al(3,3)
real(kind=sgl),INTENT(IN)       :: cin(6,6)
real(kind=sgl),INTENT(OUT)      :: cout(6,6) 

real(kind=sgl)                          :: cold(3,3,3,3), cnew(3,3,3,3)
integer(kind=irg)                       :: i,j,k,l,p,q,r,s,delta(3,3),gamma(6,2)

! initalize a bunch of variables
cold = 0.0
cnew = 0.0
cout = 0.0
delta(1,1) = 1; delta(1,2) = 6; delta(1,3) = 5
delta(2,1) = 6; delta(2,2) = 2; delta(2,3) = 4
delta(3,1) = 5; delta(3,2) = 4; delta(3,3) = 3
gamma(1,1) = 1; gamma(1,2) = 1
gamma(2,1) = 2; gamma(2,2) = 2
gamma(3,1) = 3; gamma(3,2) = 3
gamma(4,1) = 2; gamma(4,2) = 3
gamma(5,1) = 1; gamma(5,2) = 3
gamma(6,1) = 1; gamma(6,2) = 2

! convert to real tensor indices
do i=1,3
 do j=1,3
  do k=1,3
   do l=1,3
    cold(i,j,k,l) = cin(delta(i,j),delta(k,l))
   end do
  end do
 end do
end do

! and transform
do i=1,3
 do j=1,3
  do k=1,3
   do l=1,3
    do p=1,3
     do q=1,3
      do r=1,3
       do s=1,3
        cnew(i,j,k,l) = cnew(i,j,k,l) + al(i,p)*al(j,q)*al(k,r)*al(l,s)*cold(p,q,r,s)
       end do
      end do
     end do
    end do
   end do
  end do
 end do
end do

! and convert to 6-index notation again
do i=1,6
 do j=1,6
  cout(i,j) = cnew(gamma(i,1),gamma(i,2),gamma(j,1),gamma(j,2))
 end do
end do
! That's it.

end subroutine TransFourthRankTensor

!--------------------------------------------------------------------------
!
! FUNCTION: cross3
!
!> @author Saransh Singh, Carnegie Mellon University
!
!> @brief  cross product of two 3D vector in the order of input
!
!
!> @param u input vector 1
!> @param v input vector 2
! 
!> @date 03/03/16   SS 1.0 original
!> @date 12/01/16  MDG 1.1 split in single and double precision versions
!--------------------------------------------------------------------------
recursive function cross3(u, v) result(res)
!DEC$ ATTRIBUTES DLLEXPORT :: cross3

IMPLICIT NONE

real(kind=sgl),INTENT(IN)      :: u(3)
real(kind=sgl),INTENT(IN)      :: v(3)
real(kind=sgl)                 :: res(3)

res(1) = u(2)*v(3) - u(3)*v(2)
res(2) = u(3)*v(1) - u(1)*v(3)
res(3) = u(1)*v(2) - u(2)*v(1)

end function cross3

!--------------------------------------------------------------------------
!
! FUNCTION: cross3_d
!
!> @author Saransh Singh, Carnegie Mellon University
!
!> @brief  cross product of two 3D vector in the order of input (double precision)
!
!
!> @param u input vector 1
!> @param v input vector 2
! 
!> @date 03/03/16   SS 1.0 original
!--------------------------------------------------------------------------
recursive function cross3_d(u, v) result(res)
!DEC$ ATTRIBUTES DLLEXPORT :: cross3_d

IMPLICIT NONE

real(kind=dbl),INTENT(IN)      :: u(3)
real(kind=dbl),INTENT(IN)      :: v(3)
real(kind=dbl)                 :: res(3)

res(1) = u(2)*v(3) - u(3)*v(2)
res(2) = u(3)*v(1) - u(1)*v(3)
res(3) = u(1)*v(2) - u(2)*v(1)

end function cross3_d

!--------------------------------------------------------------------------
!
! function: CalcDeterminant
!
!> @author Saransh Singh, Carnegie Mellon University
!
!> @brief calculate determinant of mxn real matrix
!
!> @details a generic subroutine to calculate the determinant of any mxn matrix.
!> we will be using the dgetrf subroutine of BLAS to calculate the decomposition of
!> A as A = P * L * U, where L is the unitriangular matrix i.e. det(L) = 1; P is the
!> permutation matrix so its determinant is either +/- 1. using the property det(A) = 
!> det(P) * det(L) * det(U), we can calculate determinant as simply the product of diagonal
!> elements of U.
!
!> @param A nxn real matrix
!> @param n size of A
! 
!> @date  06/01/18 SS 1.0 original
!--------------------------------------------------------------------------
recursive function CalcDeterminant(A, m, n) result(determinant)
!DEC$ ATTRIBUTES DLLEXPORT :: CalcDeterminant

use local
use error

IMPLICIT NONE

real(kind=dbl),INTENT(IN)             :: A(m,n)
integer(kind=irg),INTENT(IN)          :: m, n

real(kind=dbl),allocatable            :: Ap(:,:)
integer(kind=irg),allocatable         :: IPIV(:)
integer(kind=irg)                     :: LDA, INFO, mm, nn, ii
real(kind=dbl)                        :: determinant

LDA = maxval((/m,1/))
nn = minval((/m,n/))

allocate(IPIV(nn), Ap(LDA,n))
Ap = A

call dgetrf(m, n, Ap, LDA, IPIV, INFO)

if(INFO .ne. 0) call FatalError('CalcDeterminant:','BLAS subroutine did not return successfully.')

determinant = 1.D0
do ii = 1,nn
  determinant = determinant * Ap(ii,ii)
  if(IPIV(ii) .ne. ii) determinant = -determinant
end do


end function CalcDeterminant

end module math
