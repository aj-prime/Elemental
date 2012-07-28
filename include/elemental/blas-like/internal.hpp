/*
   Copyright (c) 2009-2012, Jack Poulson
   All rights reserved.

   This file is part of Elemental.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

namespace elem {
namespace internal {

//----------------------------------------------------------------------------//
// Local BLAS-like: Level 2                                                   //
//----------------------------------------------------------------------------//

template<typename T,Distribution AColDist,Distribution ARowDist,
                    Distribution xColDist,Distribution xRowDist,
                    Distribution yColDist,Distribution yRowDist>
void LocalGemv
( Orientation orientation, 
  T alpha, const DistMatrix<T,AColDist,ARowDist>& A, 
           const DistMatrix<T,xColDist,xRowDist>& x,
  T beta,        DistMatrix<T,yColDist,yRowDist>& y );

template<typename T,Distribution xColDist,Distribution xRowDist,
                    Distribution yColDist,Distribution yRowDist,
                    Distribution AColDist,Distribution ARowDist>
inline void LocalGer
( T alpha, const DistMatrix<T,xColDist,xRowDist>& x, 
           const DistMatrix<T,yColDist,yRowDist>& y,
                 DistMatrix<T,AColDist,ARowDist>& A );


//----------------------------------------------------------------------------//
// Local BLAS-like: Level 3                                                   //
//----------------------------------------------------------------------------//

template<typename T,Distribution AColDist,Distribution ARowDist,
                    Distribution BColDist,Distribution BRowDist,
                    Distribution CColDist,Distribution CRowDist>
void LocalGemm
( Orientation orientationOfA, Orientation orientationOfB,
  T alpha, const DistMatrix<T,AColDist,ARowDist>& A, 
           const DistMatrix<T,BColDist,BRowDist>& B,
  T beta,        DistMatrix<T,CColDist,CRowDist>& C );

template<typename T>
void LocalTrtrmm
( Orientation orientation, UpperOrLower uplo, DistMatrix<T,STAR,STAR>& A );

template<typename T>
void LocalTrdtrmm
( Orientation orientation, UpperOrLower uplo, DistMatrix<T,STAR,STAR>& A );

template<typename T,Distribution BColDist,Distribution BRowDist>
void LocalTrmm
( LeftOrRight side, UpperOrLower uplo, 
  Orientation orientation, UnitOrNonUnit diag,
  T alpha, const DistMatrix<T,STAR,STAR>& A,
                 DistMatrix<T,BColDist,BRowDist>& B );

template<typename F,Distribution XColDist,Distribution XRowDist>
void LocalTrsm
( LeftOrRight side, UpperOrLower uplo, 
  Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F,STAR,STAR>& A, 
                 DistMatrix<F,XColDist,XRowDist>& X,
  bool checkIfSingular=false );

// TODO: Finish adding wrappers for Local BLAS-like routines

//----------------------------------------------------------------------------//
// Distributed BLAS-like helpers: Level 2                                     //
//----------------------------------------------------------------------------//
            
// This is for the case where x is a column vector and A is lower.
//
// Returns the unreduced components z[MC,* ] and z[MR,* ]:
//     z[MC,* ] := alpha tril(A)[MC,MR] x[MR,* ]
//     z[MR,* ] := alpha (trils(A)[MC,MR])^H x[MC,* ]
template<typename T>
void LocalHemvColAccumulateL
( T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,MC,STAR>& x_MC_STAR,
  const DistMatrix<T,MR,STAR>& x_MR_STAR,
        DistMatrix<T,MC,STAR>& z_MC_STAR,
        DistMatrix<T,MR,STAR>& z_MR_STAR
);

// This is for the case where x is a column vector and A is upper.
//
// Returns the unreduced components z[MC,* ] and z[MR,* ]:
//     z[MC,* ] := alpha triu(A)[MC,MR] x[MR,* ]
//     z[MR,* ] := alpha (trius(A)[MC,MR])^H x[MC,* ]
template<typename T>
void LocalHemvColAccumulateU
( T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,MC,STAR>& x_MC_STAR,
  const DistMatrix<T,MR,STAR>& x_MR_STAR,
        DistMatrix<T,MC,STAR>& z_MC_STAR,
        DistMatrix<T,MR,STAR>& z_MR_STAR
);

// This is for the case where x is a row vector and A is lower.
//
// Returns the unreduced components z[MC,* ] and z[MR,* ]:
//     z[MC,* ] := alpha tril(A)[MC,MR] (x[* ,MR])^H
//     z[MR,* ] := alpha (trils(A)[MC,MR])^H (x[* ,MC])^H
template<typename T>
void LocalHemvRowAccumulateL
( T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,STAR,MC>& x_STAR_MC,
  const DistMatrix<T,STAR,MR>& x_STAR_MR,
        DistMatrix<T,STAR,MC>& z_STAR_MC,
        DistMatrix<T,STAR,MR>& z_STAR_MR
);

// This is for the case where x is a row vector and A is upper.
//
// Returns the unreduced components z[MC,* ] and z[MR,* ]:
//     z[MC,* ] := alpha triu(A)[MC,MR] (x[* ,MR])^H
//     z[MR,* ] := alpha (trius(A)[MC,MR])^H (x[* ,MC])^H
template<typename T>
void LocalHemvRowAccumulateU
( T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,STAR,MC>& x_STAR_MC,
  const DistMatrix<T,STAR,MR>& x_STAR_MR,
        DistMatrix<T,STAR,MC>& z_STAR_MC,
        DistMatrix<T,STAR,MR>& z_STAR_MR
);

// This is for the case where x is a column vector and A is lower.
//
// Returns the unreduced components z[MC,* ] and z[MR,* ]:
//     z[MC,* ] := alpha tril(A)[MC,MR] x[MR,* ]
//     z[MR,* ] := alpha (trils(A)[MC,MR])^T x[MC,* ]
template<typename T>
void LocalSymvColAccumulateL
( T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,MC,STAR>& x_MC_STAR,
  const DistMatrix<T,MR,STAR>& x_MR_STAR,
        DistMatrix<T,MC,STAR>& z_MC_STAR,
        DistMatrix<T,MR,STAR>& z_MR_STAR
);

// This is for the case where x is a column vector and A is upper.
//
// Returns the unreduced components z[MC,* ] and z[MR,* ]:
//     z[MC,* ] := alpha triu(A)[MC,MR] x[MR,* ]
//     z[MR,* ] := alpha (trius(A)[MC,MR])^T x[MC,* ]
template<typename T>
void LocalSymvColAccumulateU
( T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,MC,STAR>& x_MC_STAR,
  const DistMatrix<T,MR,STAR>& x_MR_STAR,
        DistMatrix<T,MC,STAR>& z_MC_STAR,
        DistMatrix<T,MR,STAR>& z_MR_STAR
);

// This is for the case where x is a row vector and A is lower.
//
// Returns the unreduced components z[MC,* ] and z[MR,* ]:
//     z[MC,* ] := alpha tril(A)[MC,MR] (x[* ,MR])^T
//     z[MR,* ] := alpha (trils(A)[MC,MR])^T (x[* ,MC])^T
template<typename T>
void LocalSymvRowAccumulateL
( T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,STAR,MC>& x_STAR_MC,
  const DistMatrix<T,STAR,MR>& x_STAR_MR,
        DistMatrix<T,STAR,MC>& z_STAR_MC,
        DistMatrix<T,STAR,MR>& z_STAR_MR
);

// This is for the case where x is a row vector and A is upper.
//
// Returns the unreduced components z[MC,* ] and z[MR,* ]:
//     z[MC,* ] := alpha triu(A)[MC,MR] (x[* ,MR])^T
//     z[MR,* ] := alpha (trius(A)[MC,MR])^T (x[* ,MC])^T
template<typename T>
void LocalSymvRowAccumulateU
( T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,STAR,MC>& x_STAR_MC,
  const DistMatrix<T,STAR,MR>& x_STAR_MR,
        DistMatrix<T,STAR,MC>& z_STAR_MC,
        DistMatrix<T,STAR,MR>& z_STAR_MR
);

//----------------------------------------------------------------------------//
// Distributed BLAS-like helpers: Level 3                                     //
//----------------------------------------------------------------------------//

template<typename T>
void LocalSymmetricAccumulateLL
( Orientation orientation, T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,MC,  STAR>& B_MC_STAR,
  const DistMatrix<T,STAR,MR  >& BHermOrTrans_STAR_MR,
        DistMatrix<T,MC,  STAR>& Z_MC_STAR,
        DistMatrix<T,MR,  STAR>& Z_MR_STAR
);

template<typename T>
void LocalSymmetricAccumulateLU
( Orientation orientation, T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,MC,  STAR>& B_MC_STAR,
  const DistMatrix<T,STAR,MR  >& BHermOrTrans_STAR_MR,
        DistMatrix<T,MC,  STAR>& Z_MC_STAR,
        DistMatrix<T,MR,  STAR>& Z_MR_STAR
);

template<typename T>
void LocalSymmetricAccumulateRL
( Orientation orientation, T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,STAR,MC  >& B_STAR_MC,
  const DistMatrix<T,MR,  STAR>& BHermOrTrans_MR_STAR,
        DistMatrix<T,MC,  STAR>& ZHermOrTrans_MC_STAR,
        DistMatrix<T,MR,  STAR>& ZHermOrTrans_MR_STAR
);

template<typename T>
void LocalSymmetricAccumulateRU
( Orientation orientation, T alpha, 
  const DistMatrix<T>& A,
  const DistMatrix<T,STAR,MC  >& B_STAR_MC,
  const DistMatrix<T,MR,  STAR>& BHermOrTrans_MR_STAR,
        DistMatrix<T,MC,  STAR>& ZHermOrTrans_MC_STAR,
        DistMatrix<T,MR,  STAR>& ZHermOrTrans_MR_STAR
);

template<typename T>
void LocalTrmmAccumulateLLN
( Orientation orientation, UnitOrNonUnit diag, T alpha, 
  const DistMatrix<T>& L,
  const DistMatrix<T,STAR,MR  >& XHermOrTrans_STAR_MR,
        DistMatrix<T,MC,  STAR>& Z_MC_STAR );

template<typename T>
void LocalTrmmAccumulateLLT
( Orientation orientation, UnitOrNonUnit diag, T alpha, 
  const DistMatrix<T>& L,
  const DistMatrix<T,MC,  STAR>& X_MC_STAR,
        DistMatrix<T,MR,  STAR>& Z_MR_STAR );

template<typename T>
void LocalTrmmAccumulateLUN
( Orientation orientation, UnitOrNonUnit diag, T alpha, 
  const DistMatrix<T>& U,
  const DistMatrix<T,STAR,MR  >& XHermOrTrans_STAR_MR,
        DistMatrix<T,MC,  STAR>& Z_MC_STAR );

template<typename T>
void LocalTrmmAccumulateLUT
( Orientation orientation, UnitOrNonUnit diag, T alpha, 
  const DistMatrix<T>& U,
  const DistMatrix<T,MC,  STAR>& X_MC_STAR,
        DistMatrix<T,MR,  STAR>& Z_MR_STAR );

template<typename T>
void LocalTrmmAccumulateRLN
( Orientation orientation, UnitOrNonUnit diag, T alpha, 
  const DistMatrix<T>& L,
  const DistMatrix<T,STAR,MC  >& X_STAR_MC,
        DistMatrix<T,MR,  STAR>& ZHermOrTrans_MR_STAR );

template<typename T>
void LocalTrmmAccumulateRLT
( Orientation orientation, UnitOrNonUnit diag, T alpha, 
  const DistMatrix<T>& L,
  const DistMatrix<T,MR,  STAR>& XHermOrTrans_MR_STAR,
        DistMatrix<T,MC,  STAR>& ZHermOrTrans_MC_STAR );

template<typename T>
void LocalTrmmAccumulateRUN
( Orientation orientation, UnitOrNonUnit diag, T alpha, 
  const DistMatrix<T>& U,
  const DistMatrix<T,STAR,MC  >& X_STAR_MC,
        DistMatrix<T,MR,  STAR>& ZHermOrTrans_MR_STAR );

template<typename T>
void LocalTrmmAccumulateRUT
( Orientation orientation, UnitOrNonUnit diag, T alpha, 
  const DistMatrix<T>& U,
  const DistMatrix<T,MR,  STAR>& XHermOrTrans_MR_STAR,
        DistMatrix<T,MC,  STAR>& ZHermOrTrans_MC_STAR );

// Triangular rank-k Update:
// tril(C) := alpha tril( A B ) + beta tril(C)
//   or 
// triu(C) := alpha triu( A B ) + beta triu(C)

template<typename T>
void LocalTrrk
( UpperOrLower uplo,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,STAR,MR>& B,
  T beta,        DistMatrix<T>& C );

// Triangular rank-k Update:
// tril(C) := alpha tril( A B^{T/H} ) + beta tril(C)
//   or 
// triu(C) := alpha triu( A B^{T/H} ) + beta triu(C)

template<typename T>
void LocalTrrk
( UpperOrLower uplo,
  Orientation orientationOfB,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,MR,STAR>& B,
  T beta,        DistMatrix<T>& C );

// Triangular rank-k Update:
// tril(C) := alpha tril( A^{T/H} B ) + beta tril(C)
//   or 
// triu(C) := alpha triu( A^{T/H} B ) + beta triu(C)

template<typename T>
void LocalTrrk
( UpperOrLower uplo,
  Orientation orientationOfA,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,STAR,MR>& B,
  T beta,        DistMatrix<T>& C );

// Triangular rank-k Update:
// tril(C) := alpha tril( A^{T/H} B^{T/H} ) + beta tril(C)
//   or 
// triu(C) := alpha triu( A^{T/H} B^{T/H} ) + beta triu(C)

template<typename T>
void LocalTrrk
( UpperOrLower uplo,
  Orientation orientationOfA, Orientation orientationOfB,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,MR,STAR>& B,
  T beta,        DistMatrix<T>& C );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A B + C D ) + beta tril(E)
//   or
// triu(E) := alpha triu( A B + C D ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,STAR,MR>& B, 
           const DistMatrix<T,MC,STAR>& C, const DistMatrix<T,STAR,MR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A B + C D^{T/H} ) + beta tril(E)
//   or
// triu(E) := alpha triu( A B + C D^{T/H} ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfD,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,STAR,MR>& B,
           const DistMatrix<T,MC,STAR>& C, const DistMatrix<T,MR,STAR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A B + C^{T/H} D ) + beta tril(E)
//   or
// triu(E) := alpha triu( A B + C^{T/H} D ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfC,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,STAR,MR>& B,
           const DistMatrix<T,STAR,MC>& C, const DistMatrix<T,STAR,MR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A B + C^{T/H} D^{T/H} ) + beta tril(E)
//   or
// triu(E) := alpha triu( A B + C^{T/H} D^{T/H} ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfC,
  Orientation orientationOfD,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,STAR,MR>& B,
           const DistMatrix<T,STAR,MC>& C, const DistMatrix<T,MR,STAR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A B^{T/H} + C D ) + beta tril(E)
//   or
// triu(E) := alpha triu( A B^{T/H} + C D ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfB,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,MR,STAR>& B,
           const DistMatrix<T,MC,STAR>& C, const DistMatrix<T,STAR,MR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A B^{T/H} + C D^{T/H} ) + beta tril(E)
//   or
// triu(E) := alpha triu( A B^{T/H} + C D^{T/H} ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfB,
  Orientation orientationOfD,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,MR,STAR>& B,
           const DistMatrix<T,MC,STAR>& C, const DistMatrix<T,MR,STAR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A B^{T/H} + C^{T/H} D ) + beta tril(E)
//   or
// triu(E) := alpha triu( A B^{T/H} + C^{T/H} D ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfB,
  Orientation orientationOfC,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,MR,STAR>& B,
           const DistMatrix<T,STAR,MC>& C, const DistMatrix<T,STAR,MR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A B^{T/H} + C^{T/H} D^{T/H} ) + beta tril(E)
//   or
// triu(E) := alpha triu( A B^{T/H} + C^{T/H} D^{T/H} ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfB,
  Orientation orientationOfC,
  Orientation orientationOfD,
  T alpha, const DistMatrix<T,MC,STAR>& A, const DistMatrix<T,MR,STAR>& B,
           const DistMatrix<T,STAR,MC>& C, const DistMatrix<T,MR,STAR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A^{T/H} B + C D ) + beta tril(E)
//   or
// triu(E) := alpha triu( A^{T/H} B + C D ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfA,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,STAR,MR>& B,
           const DistMatrix<T,MC,STAR>& C, const DistMatrix<T,STAR,MR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A^{T/H} B + C D^{T/H} ) + beta tril(E)
//   or
// triu(E) := alpha triu( A^{T/H} B + C D^{T/H} ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfD,
  T alpha, const DistMatrix<T,STAR,MC  >& A, const DistMatrix<T,STAR,MR  >& B,
           const DistMatrix<T,MC,  STAR>& C, const DistMatrix<T,MR,  STAR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A^{T/H} B + C^{T/H} D ) + beta tril(E)
//   or
// triu(E) := alpha triu( A^{T/H} B + C^{T/H} D ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfC,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,STAR,MR>& B,
           const DistMatrix<T,STAR,MC>& C, const DistMatrix<T,STAR,MR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A^{T/H} B + C^{T/H} D^{T/H} ) + beta tril(E)
//   or
// triu(E) := alpha triu( A^{T/H} B + C^{T/H} D^{T/H} ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfC,
  Orientation orientationOfD,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,STAR,MR>& B,
           const DistMatrix<T,STAR,MC>& C, const DistMatrix<T,MR,STAR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A^{T/H} B^{T/H} + C D ) + beta tril(E)
//   or
// triu(E) := alpha triu( A^{T/H} B^{T/H} + C D ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfB,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,MR,STAR>& B,
           const DistMatrix<T,MC,STAR>& C, const DistMatrix<T,STAR,MR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A^{T/H} B^{T/H} + C D^{T/H} ) + beta tril(E)
//   or
// triu(E) := alpha triu( A^{T/H} B^{T/H} + C D^{T/H} ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfB,
  Orientation orientationOfD,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,MR,STAR>& B,
           const DistMatrix<T,MC,STAR>& C, const DistMatrix<T,MR,STAR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A^{T/H} B^{T/H} + C^{T/H} D ) + beta tril(E)
//   or
// triu(E) := alpha triu( A^{T/H} B^{T/H} + C^{T/H} D ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfB,
  Orientation orientationOfC,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,MR,STAR>& B,
           const DistMatrix<T,STAR,MC>& C, const DistMatrix<T,STAR,MR>& D,
  T beta,        DistMatrix<T>& E );

// Triangular rank-2k Update:
// tril(E) := alpha tril( A^{T/H} B^{T/H} + C^{T/H} D^{T/H} ) + beta tril(E)
//   or
// triu(E) := alpha triu( A^{T/H} B^{T/H} + C^{T/H} D^{T/H} ) + beta triu(E)

template<typename T>
void LocalTrr2k
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfB,
  Orientation orientationOfC,
  Orientation orientationOfD,
  T alpha, const DistMatrix<T,STAR,MC>& A, const DistMatrix<T,MR,STAR>& B, 
           const DistMatrix<T,STAR,MC>& C, const DistMatrix<T,MR,STAR>& D,
  T beta,        DistMatrix<T>& E );

// Left, Lower, Normal Trsm
template<typename F>
void TrsmLLNLarge
( UnitOrNonUnit diag,
  F alpha, const DistMatrix<F>& L, DistMatrix<F>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLLNMedium
( UnitOrNonUnit diag,
  F alpha, const DistMatrix<F>& L, DistMatrix<F>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLLNSmall
( UnitOrNonUnit diag,
  F alpha, const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X,
  bool checkIfSingular=false );

// Left, Lower, (Conjugate)Transpose Trsm
template<typename F>
void TrsmLLTLarge
( Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F>& L, DistMatrix<F>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLLTMedium
( Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F>& L, DistMatrix<F>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLLTSmall
( Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F,VC,STAR>& L, DistMatrix<F,VC,STAR>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLLTSmall
( Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F,STAR,VR>& L, DistMatrix<F,VR,STAR>& X,
  bool checkIfSingular=false );

// Left, Upper, Normal Trsm
template<typename F>
void TrsmLUNLarge
( UnitOrNonUnit diag,
  F alpha, const DistMatrix<F>& U, DistMatrix<F>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLUNMedium
( UnitOrNonUnit diag,
  F alpha, const DistMatrix<F>& U, DistMatrix<F>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLUNSmall
( UnitOrNonUnit diag,
  F alpha, const DistMatrix<F,VC,STAR>& U, DistMatrix<F,VC,STAR>& X,
  bool checkIfSingular=false );

// Left, Upper, (Conjugate)Transpose Trsm
template<typename F>
void TrsmLUTLarge
( Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F>& U, DistMatrix<F>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLUTMedium
( Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F>& U, DistMatrix<F>& X,
  bool checkIfSingular=false );
template<typename F>
void TrsmLUTSmall
( Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F,STAR,VR>& U, DistMatrix<F,VR,STAR>& X,
  bool checkIfSingular=false );

//----------------------------------------------------------------------------//
// Level 2 BLAS-like Utility Functions                                        //
//----------------------------------------------------------------------------//
template<typename T>
double
SymvGFlops
( int m, double seconds );
 
//----------------------------------------------------------------------------//
// Level 3 BLAS-like Utility Functions                                        //
//----------------------------------------------------------------------------//
template<typename T>
double 
GemmGFlops
( int m, int n, int k, double seconds );

template<typename T>
double
HemmGFlops
( LeftOrRight side, int m, int n, double seconds );

template<typename T>
double
Her2kGFlops
( int m, int k, double seconds );

template<typename T>
double
HerkGFlops
( int m, int k, double seconds );

template<typename T>
double
SymmGFlops
( LeftOrRight side, int m, int n, double seconds );
            
template<typename T>
double
Syr2kGFlops
( int m, int k, double seconds );

template<typename T>
double
SyrkGFlops
( int m, int k, double seconds );
 
template<typename T>
double
TrmmGFlops
( LeftOrRight side, int m, int n, double seconds );
            
template<typename F>
double
TrsmGFlops
( LeftOrRight side, int m, int n, double seconds );

} // internal
} // elem

//----------------------------------------------------------------------------//
// Implementations begin here                                                 //
//----------------------------------------------------------------------------//

namespace elem {
namespace internal {

//
// Level 2 Local BLAS-like routines
//

template<typename T,Distribution AColDist,Distribution ARowDist,
                    Distribution xColDist,Distribution xRowDist,
                    Distribution yColDist,Distribution yRowDist>
inline void LocalGemv
( Orientation orientation, 
  T alpha, const DistMatrix<T,AColDist,ARowDist>& A, 
           const DistMatrix<T,xColDist,xRowDist>& x,
  T beta,        DistMatrix<T,yColDist,yRowDist>& y )
{
#ifndef RELEASE
    PushCallStack("internal::LocalGemv");
    // TODO: Add error checking here
#endif
    Gemv
    ( orientation , 
      alpha, A.LockedLocalMatrix(), x.LockedLocalMatrix(),
      beta,                         y.LocalMatrix() );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T,Distribution xColDist,Distribution xRowDist,
                    Distribution yColDist,Distribution yRowDist,
                    Distribution AColDist,Distribution ARowDist>
inline void LocalGer
( T alpha, const DistMatrix<T,xColDist,xRowDist>& x, 
           const DistMatrix<T,yColDist,yRowDist>& y,
                 DistMatrix<T,AColDist,ARowDist>& A )
{
#ifndef RELEASE
    PushCallStack("internal::LocalGer");
    // TODO: Add error checking here
#endif
    Ger( alpha, x.LockedLocalMatrix(), y.LockedLocalMatrix(), A.LocalMatrix() );
#ifndef RELEASE
    PopCallStack();
#endif
}

//
// Level 3 Local BLAS-like routines
//

template<typename T,Distribution AColDist,Distribution ARowDist,
                    Distribution BColDist,Distribution BRowDist,
                    Distribution CColDist,Distribution CRowDist>
inline void LocalGemm
( Orientation orientationOfA, Orientation orientationOfB,
  T alpha, const DistMatrix<T,AColDist,ARowDist>& A, 
           const DistMatrix<T,BColDist,BRowDist>& B,
  T beta,        DistMatrix<T,CColDist,CRowDist>& C )
{
#ifndef RELEASE
    PushCallStack("internal::LocalGemm");
    if( orientationOfA == NORMAL && orientationOfB == NORMAL )
    {
        if( AColDist != CColDist || 
            ARowDist != BColDist || 
            BRowDist != CRowDist )
            throw std::logic_error("C[X,Y] = A[X,Z] B[Z,Y]");
        if( A.ColAlignment() != C.ColAlignment() )
            throw std::logic_error("A's cols must align with C's rows");
        if( A.RowAlignment() != B.ColAlignment() )
            throw std::logic_error("A's rows must align with B's cols");
        if( B.RowAlignment() != C.RowAlignment() )
            throw std::logic_error("B's rows must align with C's rows");
        if( A.Height() != C.Height() || 
            A.Width() != B.Height() || 
            B.Width() != C.Width() )
        {
            std::ostringstream msg;
            msg << "Nonconformal LocalGemmNN:\n"
                << "  A ~ " << A.Height() << " x " << A.Width() << "\n"
                << "  B ~ " << B.Height() << " x " << B.Width() << "\n"
                << "  C ~ " << C.Height() << " x " << C.Width();
            throw std::logic_error( msg.str().c_str() );
        }
    }
    else if( orientationOfA == NORMAL )
    {
        if( AColDist != CColDist ||
            ARowDist != BRowDist ||
            BColDist != CRowDist )
            throw std::logic_error("C[X,Y] = A[X,Z] (B[Y,Z])^(T/H)");
        if( A.ColAlignment() != C.ColAlignment() )
            throw std::logic_error("A's cols must align with C's rows");
        if( A.RowAlignment() != B.RowAlignment() )
            throw std::logic_error("A's rows must align with B's rows");
        if( B.ColAlignment() != C.RowAlignment() )
            throw std::logic_error("B's cols must align with C's rows");
        if( A.Height() != C.Height() || 
            A.Width() != B.Width() || 
            B.Height() != C.Width() )
        {
            std::ostringstream msg;
            msg << "Nonconformal LocalGemmNT:\n"
                << "  A ~ " << A.Height() << " x " << A.Width() << "\n"
                << "  B ~ " << B.Height() << " x " << B.Width() << "\n"
                << "  C ~ " << C.Height() << " x " << C.Width();
            throw std::logic_error( msg.str().c_str() );
        }
    }
    else if( orientationOfB == NORMAL )
    {
        if( ARowDist != CColDist ||
            AColDist != BColDist ||
            BRowDist != CRowDist )
            throw std::logic_error("C[X,Y] = (A[Z,X])^(T/H) B[Z,Y]");
        if( A.RowAlignment() != C.ColAlignment() )
            throw std::logic_error("A's rows must align with C's cols");
        if( A.ColAlignment() != B.ColAlignment() )
            throw std::logic_error("A's cols must align with B's cols");
        if( B.RowAlignment() != C.RowAlignment() )
            throw std::logic_error("B's rows must align with C's rows");
        if( A.Width() != C.Height() || 
            A.Height() != B.Height() || 
            B.Width() != C.Width() )
        {
            std::ostringstream msg;
            msg << "Nonconformal LocalGemmTN:\n"
                << "  A ~ " << A.Height() << " x " << A.Width() << "\n"
                << "  B ~ " << B.Height() << " x " << B.Width() << "\n"
                << "  C ~ " << C.Height() << " x " << C.Width();
            throw std::logic_error( msg.str().c_str() );
        }
    }
    else
    {
        if( ARowDist != CColDist ||
            AColDist != BRowDist ||
            BColDist != CRowDist )
            throw std::logic_error("C[X,Y] = (A[Z,X])^(T/H) (B[Y,Z])^(T/H)");
        if( A.RowAlignment() != C.ColAlignment() )
            throw std::logic_error("A's rows must align with C's cols");
        if( A.ColAlignment() != B.RowAlignment() )
            throw std::logic_error("A's cols must align with B's rows");
        if( B.ColAlignment() != C.RowAlignment() )
            throw std::logic_error("B's cols must align with C's rows");
        if( A.Width() != C.Height() || 
            A.Height() != B.Width() || 
            B.Height() != C.Width() )
        {
            std::ostringstream msg;
            msg << "Nonconformal LocalGemmTT:\n"
                << "  A ~ " << A.Height() << " x " << A.Width() << "\n"
                << "  B ~ " << B.Height() << " x " << B.Width() << "\n"
                << "  C ~ " << C.Height() << " x " << C.Width();
            throw std::logic_error( msg.str().c_str() );
        }
    }
#endif
    Gemm
    ( orientationOfA , orientationOfB, 
      alpha, A.LockedLocalMatrix(), B.LockedLocalMatrix(),
      beta, C.LocalMatrix() );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
inline void 
LocalTrtrmm
( Orientation orientation, UpperOrLower uplo, DistMatrix<T,STAR,STAR>& A )
{
#ifndef RELEASE
    PushCallStack("internal::LocalTrtrmm");
#endif
    Trtrmm( orientation, uplo, A.LocalMatrix() );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
inline void 
LocalTrdtrmm
( Orientation orientation, UpperOrLower uplo, DistMatrix<T,STAR,STAR>& A )
{
#ifndef RELEASE
    PushCallStack("internal::LocalTrdtrmm");
#endif
    Trdtrmm( orientation, uplo, A.LocalMatrix() );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T,Distribution BColDist,Distribution BRowDist>
inline void
LocalTrmm
( LeftOrRight side, UpperOrLower uplo, 
  Orientation orientation, UnitOrNonUnit diag,
  T alpha, const DistMatrix<T,STAR,STAR>& A,
                 DistMatrix<T,BColDist,BRowDist>& B )
{
#ifndef RELEASE
    PushCallStack("internal::LocalTrmm");
    if( (side == LEFT && BColDist != STAR) || 
        (side == RIGHT && BRowDist != STAR) )
        throw std::logic_error
        ("Distribution of RHS must conform with that of triangle");
#endif
    Trmm
    ( side, uplo, orientation, diag, 
      alpha, A.LockedLocalMatrix(), B.LocalMatrix() );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F,Distribution XColDist,Distribution XRowDist>
inline void
LocalTrsm
( LeftOrRight side, UpperOrLower uplo, 
  Orientation orientation, UnitOrNonUnit diag,
  F alpha, const DistMatrix<F,STAR,STAR>& A, 
                 DistMatrix<F,XColDist,XRowDist>& X,
  bool checkIfSingular )
{
#ifndef RELEASE
    PushCallStack("internal::LocalTrsm");
    if( (side == LEFT && XColDist != STAR) || 
        (side == RIGHT && XRowDist != STAR) )
        throw std::logic_error
        ("Distribution of RHS must conform with that of triangle");
#endif
    Trsm
    ( side, uplo, orientation, diag,
      alpha, A.LockedLocalMatrix(), X.LocalMatrix(), checkIfSingular );
#ifndef RELEASE
    PopCallStack();
#endif
}

//
// Level 2 Utility functions
//

template<>
inline double
SymvGFlops<float>
( int m, double seconds )
{ return (1.*m*m)/(1.e9*seconds); }

template<>
inline double
SymvGFlops<double>
( int m, double seconds )
{ return SymvGFlops<float>(m,seconds); }

template<>
inline double
SymvGFlops<scomplex>
( int m, double seconds )
{ return 4.*SymvGFlops<float>(m,seconds); }

template<>
inline double
SymvGFlops<dcomplex>
( int m, double seconds )
{ return 4.*SymvGFlops<float>(m,seconds); }

//
// Level 3 Utility functions
//

template<>
inline double
GemmGFlops<float>
( int m, int n, int k, double seconds )
{ return (2.*m*n*k)/(1.e9*seconds); }

template<>
inline double
GemmGFlops<double>
( int m, int n, int k, double seconds )
{ return GemmGFlops<float>(m,n,k,seconds); }

template<>
inline double
GemmGFlops<scomplex>
( int m, int n, int k, double seconds )
{ return 4.*GemmGFlops<float>(m,n,k,seconds); }

template<>
inline double
GemmGFlops<dcomplex>
( int m, int n, int k, double seconds )
{ return 4.*GemmGFlops<float>(m,n,k,seconds); }

template<>
inline double
HemmGFlops<float>
( LeftOrRight side, int m, int n, double seconds )
{
    if( side == LEFT )
        return (2.*m*m*n)/(1.e9*seconds);
    else
        return (2.*m*n*n)/(1.e9*seconds);
}

template<>
inline double
HemmGFlops<double>
( LeftOrRight side, int m, int n, double seconds )
{ return HemmGFlops<float>(side,m,n,seconds); }

template<>
inline double
HemmGFlops<scomplex>
( LeftOrRight side, int m, int n, double seconds )
{ return 4.*HemmGFlops<float>(side,m,n,seconds); }

template<>
inline double
HemmGFlops<dcomplex>
( LeftOrRight side, int m, int n, double seconds )
{ return 4.*HemmGFlops<float>(side,m,n,seconds); }

template<>
inline double
Her2kGFlops<float>
( int m, int k, double seconds )
{ return (2.*m*m*k)/(1.e9*seconds); }

template<>
inline double
Her2kGFlops<double>
( int m, int k, double seconds )
{ return Her2kGFlops<float>(m,k,seconds); }

template<>
inline double
Her2kGFlops<scomplex>
( int m, int k, double seconds )
{ return 4.*Her2kGFlops<float>(m,k,seconds); }

template<>
inline double
Her2kGFlops<dcomplex>
( int m, int k, double seconds )
{ return 4.*Her2kGFlops<float>(m,k,seconds); }

template<>
inline double
HerkGFlops<float>
( int m, int k, double seconds )
{ return (1.*m*m*k)/(1.e9*seconds); }

template<>
inline double
HerkGFlops<double>
( int m, int k, double seconds )
{ return HerkGFlops<float>(m,k,seconds); }

template<>
inline double
HerkGFlops<scomplex>
( int m, int k, double seconds )
{ return 4.*HerkGFlops<float>(m,k,seconds); }

template<>
inline double
HerkGFlops<dcomplex>
( int m, int k, double seconds )
{ return 4.*HerkGFlops<float>(m,k,seconds); }

template<>
inline double
SymmGFlops<float>
( LeftOrRight side, int m, int n, double seconds )
{
    if( side == LEFT )
        return (2.*m*m*n)/(1.e9*seconds);
    else
        return (2.*m*n*n)/(1.e9*seconds);
}

template<>
inline double
SymmGFlops<double>
( LeftOrRight side, int m, int n, double seconds ) 
{ return SymmGFlops<float>(side,m,n,seconds); }
            
template<>
inline double
SymmGFlops<scomplex>
( LeftOrRight side, int m, int n, double seconds )
{ return 4.*SymmGFlops<float>(side,m,n,seconds); }

template<>
inline double
SymmGFlops<dcomplex>
( LeftOrRight side, int m, int n, double seconds )
{ return 4.*SymmGFlops<float>(side,m,n,seconds); }
            
template<>
inline double
Syr2kGFlops<float>
( int m, int k, double seconds )
{ return (2.*m*m*k)/(1.e9*seconds); }

template<>
inline double
Syr2kGFlops<double>
( int m, int k, double seconds )
{ return Syr2kGFlops<float>(m,k,seconds); }
            
template<>
inline double
Syr2kGFlops<scomplex>
( int m, int k, double seconds )
{ return 4.*Syr2kGFlops<float>(m,k,seconds); }

template<>
inline double
Syr2kGFlops<dcomplex>
( int m, int k, double seconds )
{ return 4.*Syr2kGFlops<float>(m,k,seconds); }
            
template<>
inline double
SyrkGFlops<float>
( int m, int k, double seconds )
{ return (1.*m*m*k)/(1.e9*seconds); }

template<>
inline double
SyrkGFlops<double>
( int m, int k, double seconds )
{ return SyrkGFlops<float>(m,k,seconds); }
            
template<>
inline double
SyrkGFlops<scomplex>
( int m, int k, double seconds )
{ return 4.*SyrkGFlops<float>(m,k,seconds); }
            
template<>
inline double
SyrkGFlops<dcomplex>
( int m, int k, double seconds )
{ return 4.*SyrkGFlops<float>(m,k,seconds); }
            
template<>
inline double
TrmmGFlops<float>
( LeftOrRight side, int m, int n, double seconds )
{
    if( side == LEFT )
        return (1.*m*m*n)/(1.e9*seconds);
    else
        return (1.*m*n*n)/(1.e9*seconds);
}

template<>
inline double
TrmmGFlops<double>
( LeftOrRight side, int m, int n, double seconds )
{ return TrmmGFlops<float>(side,m,n,seconds); }

template<>
inline double
TrmmGFlops<scomplex>
( LeftOrRight side, int m, int n, double seconds )
{ return 4.*TrmmGFlops<float>(side,m,n,seconds); }

template<>
inline double
TrmmGFlops<dcomplex>
( LeftOrRight side, int m, int n, double seconds )
{ return 4.*TrmmGFlops<float>(side,m,n,seconds); }
            
template<>
inline double
TrsmGFlops<float>
( LeftOrRight side, int m, int n, double seconds )
{
    if( side == LEFT )
        return (1.*m*m*n)/(1.e9*seconds);
    else
        return (1.*m*n*n)/(1.e9*seconds);
}

template<>
inline double
TrsmGFlops<double>
( LeftOrRight side, int m, int n, double seconds )
{ return TrsmGFlops<float>(side,m,n,seconds); }
            
template<>
inline double
TrsmGFlops<scomplex>
( LeftOrRight side, int m, int n, double seconds )
{ return 4.*TrsmGFlops<float>(side,m,n,seconds); }

template<>
inline double
TrsmGFlops<dcomplex>
( LeftOrRight side, int m, int n, double seconds )
{ return 4.*TrsmGFlops<float>(side,m,n,seconds); }
            
} // internal
} // elem
