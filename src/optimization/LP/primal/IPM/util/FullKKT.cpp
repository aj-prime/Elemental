/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {
namespace lp {
namespace primal {

// The full KKT system is of the form
// |  X  Z 0   | | z | = | X Z e + tau e |
// | -I  0 A^T | | x |   |      -c       |
// |  0  A 0   | | y |   |       b       |

// TODO
// (Pivot first row to the end)
// | -I  0 A^T | | z |   |      -c       |
// |  0  A 0   | | x |   |       b       |
// |  X  Z 0   | | y | = | X Z e + tau e |

// TODO
// (Negate the third equation, divide by Z, and swap x and z)
//   |  0 A^T    -I     | | x |   |        -c           |
//   |  A 0       0     | | y | = |         b           |
//   | -I 0   -inv(Z) X | | z |   | (X Z e + tau e) / Z |

template<typename Real>
void KKT
( const Matrix<Real>& A, const Matrix<Real>& x, const Matrix<Real>& z,
  Matrix<Real>& J )
{
    DEBUG_ONLY(CallStackEntry cse("lp::primal::KKT"))
    const Int m = A.Height();
    const Int n = A.Width();

    Zeros( J, 2*n+m, 2*n+m );
    const IR zInd(0,n), xInd(n,2*n), yInd(2*n,2*n+m);
    auto Jzz = J(zInd,zInd); auto Jzx = J(zInd,xInd); auto Jzy = J(zInd,yInd);
    auto Jxz = J(xInd,zInd); auto Jxx = J(xInd,xInd); auto Jxy = J(xInd,yInd);
    auto Jyz = J(yInd,zInd); auto Jyx = J(yInd,xInd); auto Jyy = J(yInd,yInd);

    // Jxz := X
    // ========
    Diagonal( Jzz, x );

    // Jzx := Z
    // ========
    Diagonal( Jzx, z );

    // Jxz := -I
    // =========
    Identity( Jxz, n, n );
    Scale( Real(-1), Jxz );

    // Jxy := A^T
    // ==========
    Transpose( A, Jxy ); 

    // Jyx := A
    // ========
    Jyx = A;
}

template<typename Real>
void KKT
( const AbstractDistMatrix<Real>& A, 
  const AbstractDistMatrix<Real>& xPre, const AbstractDistMatrix<Real>& zPre,
  AbstractDistMatrix<Real>& JPre )
{
    DEBUG_ONLY(CallStackEntry cse("lp::primal::KKT"))
    const Int m = A.Height();
    const Int n = A.Width();

    auto xPtr = ReadProxy<Real,STAR,STAR>(&xPre); auto& x = *xPtr;
    auto zPtr = ReadProxy<Real,STAR,STAR>(&zPre); auto& z = *zPtr;
    auto JPtr = WriteProxy<Real,MC,MR>(&JPre); auto& J = *JPtr;

    Zeros( J, 2*n+m, 2*n+m );
    const IR zInd(0,n), xInd(n,2*n), yInd(2*n,2*n+m);
    auto Jzz = J(zInd,zInd); auto Jzx = J(zInd,xInd); auto Jzy = J(zInd,yInd);
    auto Jxz = J(xInd,zInd); auto Jxx = J(xInd,xInd); auto Jxy = J(xInd,yInd);
    auto Jyz = J(yInd,zInd); auto Jyx = J(yInd,xInd); auto Jyy = J(yInd,yInd);

    // Jxz := X
    // ========
    Diagonal( Jzz, x.LockedMatrix() );

    // Jzx := Z
    // ========
    Diagonal( Jzx, z.LockedMatrix() );

    // Jxz := -I
    // =========
    Identity( Jxz, n, n );
    Scale( Real(-1), Jxz );

    // Jxy := A^T
    // ==========
    Transpose( A, Jxy );

    // Jyx := A
    // ========
    Jyx = A;
}

template<typename Real>
void KKTRHS
( const Matrix<Real>& rmu, const Matrix<Real>& rc, const Matrix<Real>& rb,
  Matrix<Real>& rhs )
{
    DEBUG_ONLY(CallStackEntry cse("lp::primal::KKTRHS"))
    const Int m = rb.Height();
    const Int n = rc.Height();
    const IR zInd(0,n), xInd(n,2*n), yInd(2*n,2*n+m);
    Zeros( rhs, 2*n+m, 1 );

    auto rhs_z = rhs(zInd,IR(0,1));
    rhs_z = rmu;
    Scale( Real(-1), rhs_z );

    auto rhs_x = rhs(xInd,IR(0,1));
    rhs_x = rc;
    Scale( Real(-1), rhs_x );

    auto rhs_y = rhs(yInd,IR(0,1));
    rhs_y = rb;
    Scale( Real(-1), rhs_y );
}

template<typename Real>
void KKTRHS
( const AbstractDistMatrix<Real>& rmu, const AbstractDistMatrix<Real>& rc, 
  const AbstractDistMatrix<Real>& rb, AbstractDistMatrix<Real>& rhsPre )
{
    DEBUG_ONLY(CallStackEntry cse("lp::primal::KKTRHS"))

    auto rhsPtr = WriteProxy<Real,MC,MR>(&rhsPre); 
    auto& rhs = *rhsPtr;

    const Int m = rb.Height();
    const Int n = rc.Height();
    const IR zInd(0,n), xInd(n,2*n), yInd(2*n,2*n+m);
    Zeros( rhs, 2*n+m, 1 );

    auto rhs_z = rhs(zInd,IR(0,1));
    Copy( rmu, rhs_z );
    Scale( Real(-1), rhs_z );

    auto rhs_x = rhs(xInd,IR(0,1));
    Copy( rc, rhs_x );
    Scale( Real(-1), rhs_x );

    auto rhs_y = rhs(yInd,IR(0,1));
    Copy( rb, rhs_y );
    Scale( Real(-1), rhs_y );
}

template<typename Real>
void ExpandKKTSolution
( Int m, Int n, const Matrix<Real>& rhs, 
  Matrix<Real>& dx, Matrix<Real>& dy, Matrix<Real>& dz )
{
    DEBUG_ONLY(CallStackEntry cse("lp::primal::ExpandKKTSolution"))
    if( rhs.Height() != 2*n+m || rhs.Width() != 1 )
        LogicError("Right-hand side was the wrong size");

    const IR zInd(0,n), xInd(n,2*n), yInd(2*n,2*n+m);    
    dz = rhs(zInd,IR(0,1));
    dx = rhs(xInd,IR(0,1));
    dy = rhs(yInd,IR(0,1));
}

template<typename Real>
void ExpandKKTSolution
( Int m, Int n, const AbstractDistMatrix<Real>& rhsPre, 
  AbstractDistMatrix<Real>& dx, AbstractDistMatrix<Real>& dy, 
  AbstractDistMatrix<Real>& dz )
{
    DEBUG_ONLY(CallStackEntry cse("lp::primal::ExpandKKTSolution"))
    
    auto rhsPtr = ReadProxy<Real,MC,MR>(&rhsPre);    
    auto& rhs = *rhsPtr;

    if( rhs.Height() != 2*n+m || rhs.Width() != 1 )
        LogicError("Right-hand side was the wrong size");

    const IR zInd(0,n), xInd(n,2*n), yInd(2*n,2*n+m);    
    Copy( rhs(zInd,IR(0,1)), dz );
    Copy( rhs(xInd,IR(0,1)), dx );
    Copy( rhs(yInd,IR(0,1)), dy );
}

#define PROTO(Real) \
  template void KKT \
  ( const Matrix<Real>& A, \
    const Matrix<Real>& x, const Matrix<Real>& z, \
    Matrix<Real>& J ); \
  template void KKT \
  ( const AbstractDistMatrix<Real>& A, \
    const AbstractDistMatrix<Real>& x, const AbstractDistMatrix<Real>& z, \
    AbstractDistMatrix<Real>& J ); \
  template void KKTRHS \
  ( const Matrix<Real>& rmu, const Matrix<Real>& rc, \
    const Matrix<Real>& rb, Matrix<Real>& rhs ); \
  template void KKTRHS \
  ( const AbstractDistMatrix<Real>& rmu, const AbstractDistMatrix<Real>& rc, \
    const AbstractDistMatrix<Real>& rb, AbstractDistMatrix<Real>& rhs ); \
  template void ExpandKKTSolution \
  ( Int m, Int n, const Matrix<Real>& rhs, \
    Matrix<Real>& dx, Matrix<Real>& dy, Matrix<Real>& dz ); \
  template void ExpandKKTSolution \
  ( Int m, Int n, const AbstractDistMatrix<Real>& rhs, \
    AbstractDistMatrix<Real>& dx, AbstractDistMatrix<Real>& dy, \
    AbstractDistMatrix<Real>& dz );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#include "El/macros/Instantiate.h"

} // namespace primal
} // namespace lp
} // namespace El
