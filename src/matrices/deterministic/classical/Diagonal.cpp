/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/matrices.hpp>

namespace El {

template<typename S,typename T>
void Diagonal( Matrix<S>& D, const vector<T>& d )
{
    EL_DEBUG_CSE
    const Int n = d.size();
    Zeros( D, n, n );

    for( Int j=0; j<n; ++j )
        D(j,j) = d[j];
}

template<typename S,typename T>
void Diagonal( Matrix<S>& D, const Matrix<T>& d )
{
    EL_DEBUG_CSE
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int n = d.Height();
    Zeros( D, n, n );

    for( Int j=0; j<n; ++j )
        D(j,j) = d(j);
}

template<typename S,typename T>
void Diagonal( AbstractDistMatrix<S>& D, const vector<T>& d )
{
    EL_DEBUG_CSE
    const Int n = d.size();
    Zeros( D, n, n );

    const Int localWidth = D.LocalWidth();
    for( Int jLoc=0; jLoc<localWidth; ++jLoc )
    {
        const Int j = D.GlobalCol(jLoc);
        D.Set( j, j, S(d[j]) );
    }
}

template<typename S,typename T>
void Diagonal( AbstractDistMatrix<S>& D, const Matrix<T>& d )
{
    EL_DEBUG_CSE
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int n = d.Height();
    Zeros( D, n, n );

    const Int localWidth = D.LocalWidth();
    for( Int jLoc=0; jLoc<localWidth; ++jLoc )
    {
        const Int j = D.GlobalCol(jLoc);
        D.Set( j, j, S(d(j)) );
    }
}

template<typename S,typename T>
void Diagonal( AbstractDistMatrix<S>& D, const AbstractDistMatrix<T>& d )
{
    EL_DEBUG_CSE
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int n = d.Height();
    Zeros( D, n, n );
    if( d.RedundantRank() == 0 && d.IsLocalCol(0) )
    {
        D.Reserve( d.LocalHeight() );
        const Int localHeight = d.LocalHeight();
        for( Int iLoc=0; iLoc<localHeight; ++iLoc )
        {
            const Int i = d.GlobalRow(iLoc);
            D.QueueUpdate( i, i, S(d.GetLocal(iLoc,0)) );
        }
    }
    D.ProcessQueues();
}


#define PROTO_TYPES(S,T) \
  template void Diagonal( Matrix<S>& D, const vector<T>& d ); \
  template void Diagonal( Matrix<S>& D, const Matrix<T>& d ); \
  template void Diagonal( AbstractDistMatrix<S>& D, const vector<T>& d ); \
  template void Diagonal( AbstractDistMatrix<S>& D, const Matrix<T>& d ); \
  template void Diagonal \
  ( AbstractDistMatrix<S>& D, const AbstractDistMatrix<T>& d );

#define PROTO_INT(S) PROTO_TYPES(S,S)

#define PROTO_REAL(S) \
  PROTO_TYPES(S,Int) \
  PROTO_TYPES(S,S)

#define PROTO_COMPLEX(S) \
  PROTO_TYPES(S,Int) \
  PROTO_TYPES(S,Base<S>) \
  PROTO_TYPES(S,S)

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
