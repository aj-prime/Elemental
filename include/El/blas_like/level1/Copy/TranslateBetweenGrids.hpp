/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_TRANSLATEBETWEENGRIDS_HPP
#define EL_BLAS_COPY_TRANSLATEBETWEENGRIDS_HPP

namespace El
{
namespace copy
{

template<typename T,Dist U,Dist V,Device D1,Device D2>
void TranslateBetweenGrids
(DistMatrix<T,U,V,ELEMENT,D1> const& A,
  DistMatrix<T,U,V,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE

    //if (D1 != Device::CPU)
        //LogicError("TranslateBetweenGrids: Device not implemented.");

    if (D1 != D2)
        LogicError("TranslateBetweenGrids: ",
                   "Mixed-device implementation not implemented.");

    GeneralPurpose(A, B);
}

// TODO(poulson): Compare against copy::GeneralPurpose
// FIXME (trb 03/06/18) -- Need to do the GPU impl
template<typename T, Device D1, Device D2>
void TranslateBetweenGrids
(DistMatrix<T,MC,MR,ELEMENT,D1> const& A,
  DistMatrix<T,MC,MR,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE

    //if (D1 != Device::CPU)
        //LogicError("TranslateBetweenGrids<MC,MR,ELEMENT>: "
                   //"Device not implemented.");

    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();
    B.Resize(m, n);
    mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
    mpi::Group owningGroupA = A.Grid().OwningGroup();

    // Just need to ensure that each viewing comm contains the other team's
    // owning comm. Congruence is too strong.

    // Compute the number of process rows and columns that each process
    // needs to send to.
    const Int colStride = B.ColStride();
    const Int rowStride = B.RowStride();
    const Int colShiftB = B.ColShift();
    const Int rowShiftB = B.RowShift();
    const Int colRank = B.ColRank();
    const Int rowRank = B.RowRank();
    const Int colRankA = A.ColRank();
    const Int rowRankA = A.RowRank();
    const Int colStrideA = A.ColStride();
    const Int rowStrideA = A.RowStride();
    const Int colGCD = GCD(colStride, colStrideA);
    const Int rowGCD = GCD(rowStride, rowStrideA);
    const Int colLCM = colStride*colStrideA / colGCD;
    const Int rowLCM = rowStride*rowStrideA / rowGCD;
    const Int numColSends = colStride / colGCD;
    const Int numRowSends = rowStride / rowGCD;

    const Int colAlignA = A.ColAlign();
    const Int rowAlignA = A.RowAlign();
    const Int colAlignB = B.ColAlign();
    const Int rowAlignB = B.RowAlign();

    const bool inBGrid = B.Participating();
    const bool inAGrid = A.Participating();

    // std::printf("MPI_RANK:%d colStrideB:%d rowStrideB:%d colShiftB:%d rowShiftB:%d colRankB:%d rowRankB:%d colAlignB:%d rowAlignB:%d colStrideA:%d rowStrideA:%d  colRankA:%d rowRankA:%d colAlignA:%d rowAlignA:%d\n"
    //     ,mpi::Rank(viewingCommB),colStride,rowStride,colShiftB,rowShiftB,colRank,rowRank, colAlignB,rowAlignB,    colStrideA,rowStrideA,colRankA,rowRankA,colAlignA,rowAlignA);
    if(!inBGrid && !inAGrid)
        return;

    const Int maxSendSize =
      (m/(colStrideA*numColSends)+1) * (n/(rowStrideA*numRowSends)+1);

    //std::printf("colStrideA %d rowStrideA %d\n",colStrideA,rowStrideA);

    // Translate the ranks from A's VC communicator to B's viewing so that
    // we can match send/recv communicators. Since A's VC communicator is not
    // necessarily defined on every process, we instead work with A's owning
    // group and account for row-major ordering if necessary.
    const int sizeA = A.Grid().Size();
    vector<int> rankMap(sizeA), ranks(sizeA);
    if(A.Grid().Order() == COLUMN_MAJOR)
    {
        for(int j=0; j<sizeA; ++j)
            ranks[j] = j;
    }
    else
    {
        // The (i,j) = i + j*colStrideA rank in the column-major ordering is
        // equal to the j + i*rowStrideA rank in a row-major ordering.
        // Since we desire rankMap[i+j*colStrideA] to correspond to process
        // (i,j) in A's grid's rank in this viewing group, ranks[i+j*colStrideA]
        // should correspond to process (i,j) in A's owning group. Since the
        // owning group is ordered row-major in this case, its rank is
        // j+i*rowStrideA. Note that setting
        // ranks[j+i*rowStrideA] = i+j*colStrideA is *NOT* valid.
        for(int i=0; i<colStrideA; ++i)
            for(int j=0; j<rowStrideA; ++j)
                ranks[i+j*colStrideA] = j+i*rowStrideA;
    }
    mpi::Translate(
        owningGroupA, sizeA, ranks.data(), viewingCommB, rankMap.data());

    // Have each member of A's grid individually send to all numRow x numCol
    // processes in order, while the members of this grid receive from all
    // necessary processes at each step.
    Int requiredMemory = 0;
    if(inAGrid)
        requiredMemory += maxSendSize;
    if(inBGrid)
        requiredMemory += maxSendSize;

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B.LockedMatrix());

    simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);

    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();

    Int recvRow = 0; // avoid compiler warnings...
    if(inAGrid)
        recvRow = Mod(Mod(colRankA-colAlignA,colStrideA)+colAlignB,colStride);

    //std::printf("numColSends: %d numRowSends: %d\n", numColSends,numRowSends);
    for(Int colSend=0; colSend<numColSends; ++colSend)
    {
        Int recvCol = 0; // avoid compiler warnings...
        if(inAGrid)
            recvCol=Mod(Mod(rowRankA-rowAlignA,rowStrideA)+rowAlignB,rowStride);
        for(Int rowSend=0; rowSend<numRowSends; ++rowSend)
        {
            mpi::Request<T> sendRequest;
            // Fire off this round of non-blocking sends
            if(inAGrid)
            {
                // Pack the data
                Int sendHeight = Length(mLocA,colSend,numColSends);
                Int sendWidth = Length(nLocA,rowSend,numRowSends);
                //std::printf("sendHeight: %d sendWidth: %d\n", sendHeight,sendWidth);
                copy::util::InterleaveMatrix(
                    sendHeight, sendWidth,
                    A.LockedBuffer(colSend,rowSend),
                    numColSends, numRowSends*A.LDim(),
                    sendBuf, 1, sendHeight, syncInfoA);

                Synchronize(syncInfoA);

                // Send data
                const Int recvVCRank = recvRow + recvCol*colStride;
                const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);
                mpi::ISend
                (sendBuf, sendHeight*sendWidth, recvViewingRank,
                  viewingCommB, sendRequest);
            }
            // Perform this round of recv's
            if(inBGrid)
            {
                const Int sendColOffset = colAlignA;
                const Int recvColOffset =
                  Mod(colSend*colStrideA+colAlignB,colStride);
                const Int sendRowOffset = rowAlignA;
                const Int recvRowOffset =
                  Mod(rowSend*rowStrideA+rowAlignB,rowStride);

                const Int colShift = Mod(colRank-recvColOffset, colStride);
                const Int rowShift = Mod(rowRank-recvRowOffset, rowStride);

                const Int firstSendRow = Mod(colShift+sendColOffset,colStrideA);
                const Int firstSendCol = Mod(rowShift+sendRowOffset,rowStrideA);

                const Int numColRecvs = Length(colStrideA,colShift,colStride);
                const Int numRowRecvs = Length(rowStrideA,rowShift,rowStride);

                // Recv data
                // For now, simply receive sequentially. Until we switch to
                // nonblocking recv's, we won't be using much of the
                // recvBuf
                Int sendRow = firstSendRow;
                for(Int colRecv=0; colRecv<numColRecvs; ++colRecv)
                {
                    const Int sendColShift =
                      Shift(sendRow, colAlignA, colStrideA) +
                      colSend*colStrideA;
                    const Int sendHeight = Length(m, sendColShift, colLCM);
                    const Int localColOffset =
                      (sendColShift-colShiftB) / colStride;

                    Int sendCol = firstSendCol;
                    for(Int rowRecv=0; rowRecv<numRowRecvs; ++rowRecv)
                    {
                        const Int sendRowShift =
                          Shift(sendCol, rowAlignA, rowStrideA) +
                          rowSend*rowStrideA;
                        const Int sendWidth = Length(n, sendRowShift, rowLCM);
                        const Int localRowOffset =
                          (sendRowShift-rowShiftB) / rowStride;

                        const Int sendVCRank = sendRow+sendCol*colStrideA;
                        mpi::Recv(
                            recvBuf, sendHeight*sendWidth, rankMap[sendVCRank],
                            viewingCommB, syncInfoB);

                        // Unpack the data
                        copy::util::InterleaveMatrix(
                            sendHeight, sendWidth,
                            recvBuf, 1, sendHeight,
                            B.Buffer(localColOffset,localRowOffset),
                            colLCM/colStride, (rowLCM/rowStride)*B.LDim(),
                            syncInfoB);

                        // Set up the next send col
                        sendCol = Mod(sendCol+rowStride,rowStrideA);
                    }
                    // Set up the next send row
                    sendRow = Mod(sendRow+colStride,colStrideA);
                }
            }
            // Ensure that this round of non-blocking sends completes
            if(inAGrid)
            {
                mpi::Wait(sendRequest);
                recvCol = Mod(recvCol+rowStrideA,rowStride);
            }
        }
        if(inAGrid)
            recvRow = Mod(recvRow+colStrideA,colStride);
    }
}




template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllreduce
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
    std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    This function is specific to the LBANN with implementation for specific cases
    Subgrids in B_vector are assumed to be subset of resources in A grid 
    */
    EL_DEBUG_CSE

    Int indexB = -1;
    const Int numSubGrids = int(B_Vector.size());

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if((*B_Vector[i]).Participating())
        {
            if(indexB!=-1)
            {
                std::printf("Error: rank is in multiple subgrids\n");
            }
            indexB = i;
        }
    }

    //DistMatrix<T,STAR,VC,ELEMENT,D2>& B(*B_Vector[indexB]) ;
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>(&(*B_Vector[indexB]));
    //const Int colStrideB = B->ColStride();
    const Int rowStrideB = B->RowStride();
    //const Int colShiftB = B->ColShift();
    //const Int rowShiftB = B->RowShift();
    //const Int colAlignB = B->ColAlign();
    //const Int rowAlignB = B->RowAlign();
    const Int sizeB = B->Grid().VCSize();


    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();

    const Int posInSubGrid = B->Grid().VCRank();
    const Int myLocalRankB = posInSubGrid;
    const Int posInGrid = A.Grid().VCRank();
    A.Resize(m,n); 
    

    
    

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    //mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();
    const Int sizeA = A.Grid().VCSize();

    //const Int myRankViewing = mpi::Rank(viewingCommA);


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    //const Int myLocalRankA = A.Grid().VCRank();


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3 
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    std::vector<bool> require_data(sizeB,false);
    std::vector<int> index_to_put(sizeB,-1);
    std::vector<int> index_from(sizeB,-1);
    Int temp_require_data = Mod(posInGrid, sizeB) ;
    double total_iter = posInGrid;

    for(Int i=0; i< int(rowLCM/sizeA); ++i)
    {

        require_data[temp_require_data] = true;
        index_to_put[temp_require_data] = i;
        index_from[temp_require_data] = int(std::floor(total_iter/sizeB));

        total_iter += sizeA;
        temp_require_data = Mod(temp_require_data+sizeA, sizeB );
    }

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());
    SyncInfo<D1> syncGeneral = SyncInfo<D1>();

    //

    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocB * nLocB;
    simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);
    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();

    for(Int localRankB = 0; localRankB < sizeB; localRankB++)
    {
        if(myLocalRankB==localRankB && inBGrid)
        {
            copy::util::InterleaveMatrix(
                mLocB, nLocB,
                B->LockedBuffer(0,0),
                1, B->LDim(),
                sendBuf, 1, mLocB, syncInfoB);

        }
        else
        {
            T val = 0;
            hydrogen::details::setBufferToValue(sendBuf, maxSendSize, val,syncGeneral);
        }

        Synchronize(syncGeneral);
        mpi::AllReduce( sendBuf, recvBuf, maxSendSize, mpi::SUM, viewingCommA,syncInfoB);

        if(require_data[localRankB])
        {
            int sendWidth = int(n / rowLCM);
            copy::util::InterleaveMatrix(
                            m, sendWidth,
                            recvBuf + (index_from[localRankB] * m ) , 1, m*(rowLCM/sizeB),
                            A.Buffer(0,index_to_put[localRankB]),
                            1, (rowLCM/sizeA)*A.LDim(),
                            syncInfoA);

        }
        Synchronize(syncInfoA);

    }
    Synchronize(syncInfoA);
    Synchronize(syncInfoB);


}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllreduceOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, mpi::Comm const& allreduceComm, SyncInfo<D1> & syncGeneral)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    This function is specific to the LBANN with implementation for specific cases
    Subgrids in B_vector are assumed to be subset of resources in A grid 
    */
    EL_DEBUG_CSE

    Int indexB = -1;
    const Int numSubGrids = int(B_Vector.size());

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if((*B_Vector[i]).Participating())
        {
            if(indexB!=-1)
            {
                std::printf("Error: rank is in multiple subgrids\n");
            }
            indexB = i;
        }
    }
    //DistMatrix<T,STAR,VC,ELEMENT,D2>& B(*B_Vector[indexB]) ;
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>(&(*B_Vector[indexB]));

    //const Int colStrideB = B->ColStride();
    const Int rowStrideB = B->RowStride();
    //const Int colShiftB = B->ColShift();
    //const Int rowShiftB = B->RowShift();
    //const Int colAlignB = B->ColAlign();
    //const Int rowAlignB = B->RowAlign();
    const Int sizeB = B->Grid().VCSize();


    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();

    const Int posInSubGrid = B->Grid().VCRank();
    //const Int myLocalRankB = posInSubGrid;
    const Int posInGrid = A.Grid().VCRank();
    A.Resize(m,n); 
    

    
    

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    //mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();
    const Int sizeA = A.Grid().VCSize();

    //const Int myRankViewing = mpi::Rank(viewingCommA);


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    //const Int myLocalRankA = A.Grid().VCRank();


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3 
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    

    const Int index_from = int(std::floor(posInGrid/sizeB));

    

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());
    //SyncInfo<D1> syncGeneral = SyncInfo<D1>();

    //

    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocB * nLocB;
    simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);
    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();

    //mpi::Comm allreduceComm;

    //mpi::Split(viewingCommA, posInSubGrid, posInGrid, allreduceComm);

    if(inBGrid)
    {
        copy::util::InterleaveMatrix(
                mLocB, nLocB,
                B->LockedBuffer(0,0),
                1, B->LDim(),
                sendBuf, 1, mLocB, syncGeneral);

    }
    //Synchronize(syncGeneral);
    mpi::AllReduce( sendBuf, recvBuf, maxSendSize, mpi::SUM, allreduceComm,syncGeneral);

    if(inAGrid)
    {
        int sendWidth = int(n / rowLCM);
        copy::util::InterleaveMatrix(
                m, sendWidth,
                recvBuf + (index_from * m ) , 1, m*(rowLCM/sizeB),
                A.Buffer(0,0),
                1, (rowLCM/sizeA)*A.LDim(),
                syncGeneral);

    }

}



template<typename T, Device D1, Device D2>
void TranslateBetweenGridsAllreduceOpt
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    This function is specific to the LBANN with implementation for specific cases
    Subgrids in B_vector are assumed to be subset of resources in A grid 
    */
    EL_DEBUG_CSE

    Int indexB = -1;
    const Int numSubGrids = int(B_Vector.size());

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if((*B_Vector[i]).Participating())
        {
            if(indexB!=-1)
            {
                std::printf("Error: rank is in multiple subgrids\n");
            }
            indexB = i;
        }
    }
    //DistMatrix<T,STAR,VC,ELEMENT,D2>& B(*B_Vector[indexB]) ;
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    //const Int colStrideB = B->ColStride();
    const Int rowStrideB = B->RowStride();
    //const Int colShiftB = B->ColShift();
    //const Int rowShiftB = B->RowShift();
    //const Int colAlignB = B->ColAlign();
    //const Int rowAlignB = B->RowAlign();
    const Int sizeB = B->Grid().VCSize();


    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();

    const Int posInSubGrid = B->Grid().VCRank();
    //const Int myLocalRankB = posInSubGrid;
    const Int posInGrid = A.Grid().VCRank();
    A.Resize(m,n); 
    

    
    

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    //mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();
    const Int sizeA = A.Grid().VCSize();

    //const Int myRankViewing = mpi::Rank(viewingCommA);


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    //const Int myLocalRankA = A.Grid().VCRank();


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3 
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    

    const Int index_from = int(std::floor(posInGrid/sizeB));

    

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());
    SyncInfo<D1> syncGeneral = SyncInfo<D1>();

    //

    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocB * nLocB;
    simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);
    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();

    mpi::Comm allreduceComm;

    mpi::Split(viewingCommA, posInSubGrid, posInGrid, allreduceComm);

    if(inBGrid)
    {
        copy::util::InterleaveMatrix(
                mLocB, nLocB,
                B->LockedBuffer(0,0),
                1, B->LDim(),
                sendBuf, 1, mLocB, syncGeneral);

    }
    //Synchronize(syncGeneral);
    mpi::AllReduce( sendBuf, recvBuf, maxSendSize, mpi::SUM, allreduceComm,syncGeneral);

    if(inAGrid)
    {
        int sendWidth = int(n / rowLCM);
        copy::util::InterleaveMatrix(
                m, sendWidth,
                recvBuf + (index_from * m ) , 1, m*(rowLCM/sizeB),
                A.Buffer(0,0),
                1, (rowLCM/sizeA)*A.LDim(),
                syncGeneral);

    }

}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsScatterComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& ScatterComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension 
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers 

    Subgrids in B_vector are assumed to be subset of resources in A grid 


    Resources are assumed to be distribted equally among different subgrids 
    

    */
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    mpi::Group owningGroupA = A.Grid().OwningGroup();


    Int rowStrideA = A.RowStride();


    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }

    
    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }



    //const Int myRankViewing = mpi::Rank(viewingCommA);
    const Int scatterCommSize = mpi::Size( ScatterComm );
    const int sendCounts = int((mLocA*nLocA)/scatterCommSize);
    const int numMatricesInSubGrid  = int(numChildLayers / scatterCommSize);

    std::vector<Int> indexBVec;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }
    const Int indexB = indexBVec[0];
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2> conversionMatrix(mLocA,nLocA), transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(int((mLocA*nLocA)/splitDim),int(splitDim/scatterCommSize));

    Copy(A.LockedMatrix(), conversionMatrix);

    conversionMatrix.Resize(splitDim, int((mLocA*nLocA)/splitDim));

    Transpose(conversionMatrix,transposedMatrix);

    conversionMatrix.Resize(int(mLocA/scatterCommSize),nLocA);

    const Int rowStrideB = B->RowStride();
    const Int sizeB = B->Grid().VCSize();
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;
    const Int posInGrid = A.Grid().VCRank();


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3 
    // Child  0 1 2 0 1 2 0 1 2 0 1 2
    std::vector<int> index_to_put(sizeA,-1);

    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {       
        index_to_put[i] = i;
    }


    //SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(recvTransposedMatrix);

    
    //const int myLocalDataRankA = int(std::floor(posInGrid/sizeB));
    int partialHeight = int(splitDim/ scatterCommSize);
    int partialChildHeight = int(partialHeight / numMatricesInSubGrid);

    conversionMatrix.Resize(partialChildHeight,nLocA);

    for(Int localDataRankA = 0; localDataRankA < int(rowLCM/sizeB); localDataRankA++)
    {

        //comm is useless parameter in this function 
        //Aluminum infer comm from sync object 
        

        // MPI_Scatter((void *)transposedMatrix.Buffer(), sendCounts, mpi::TypeMap<T>(), (void *)recvTransposedMatrix.Buffer(), sendCounts, mpi::TypeMap<T>(), localDataRankA, ScatterComm.GetMPIComm());
        // Synchronize(syncGeneral);

        mpi::Scatter((T *)transposedMatrix.Buffer(), sendCounts, (T *)recvTransposedMatrix.Buffer(), sendCounts, localDataRankA, ScatterComm,
               syncGeneral);

        Synchronize(syncGeneral);

        int sendWidth = int(n / rowLCM);

        for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
        {
            //Transpose(recvTransposedMatrix(Range<int>(partialChildHeight*childLayerSubGrid, partialChildHeight*(childLayerSubGrid+1)), Range<int>(0,n)),conversionMatrix);
            Transpose(recvTransposedMatrix( Range<Int>(0,sendWidth*(mLocA/splitDim)),Range<Int>(partialChildHeight*childLayerSubGrid, partialChildHeight*(childLayerSubGrid+1))),conversionMatrix);
            //dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(0,localDataRankA)
            copy::util::InterleaveMatrix(
                        partialChildHeight*(mLocA/splitDim), sendWidth,
                        conversionMatrix.Buffer()  , 1, partialChildHeight*(mLocA/splitDim),
                        dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(0,localDataRankA),
                        1, dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->LDim() * int(rowLCM/sizeB),
                        syncInfoB);
        }

        
        Synchronize(syncInfoB);



    }

}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsScatterOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& ScatterComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension 
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers 

    Subgrids in B_vector are assumed to be subset of resources in A grid 


    Resources are assumed to be distribted equally among different subgrids 
    

    */
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();





    
    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }

    
    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }



    //const Int myRankViewing = mpi::Rank(viewingCommA);
    const Int scatterCommSize = mpi::Size( ScatterComm );
    const int sendCounts = int((mLocA*nLocA)/scatterCommSize);
    const int numMatricesInSubGrid  = int(numChildLayers / scatterCommSize);

    std::vector<Int> indexBVec;
    std::vector<SyncInfo<D2>> syncInfoBVector;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);
            syncInfoBVector.push_back(SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[i]))->LockedMatrix()) );

        }
    }
    const Int indexB = indexBVec[0];
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2>  transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(int((mLocA*nLocA)/splitDim),int(splitDim/scatterCommSize));

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    const Int maxSendSize = mLocA*nLocA;
    const Int maxRecvSize = int(mLocA*nLocA/scatterCommSize);

    simple_buffer<T,D2> send_buf(maxSendSize, syncInfoA);
    simple_buffer<T,D2> recv_buf(maxRecvSize, syncGeneral);

    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();


    copy::util::InterleaveMatrix(
                        splitDim, int((mLocA*nLocA)/splitDim),
                        A.LockedBuffer()  , 1, splitDim,
                        sendBuf,
                        int((mLocA*nLocA)/splitDim),1,
                        syncInfoA);

    //const Int posInSubGrid = B->Grid().VCRank(); 

    //const Int colStrideB = B->ColStride();
    const Int rowStrideB = B->RowStride();
    //const Int colShiftB = B->ColShift();
    //const Int rowShiftB = B->RowShift();
    //const Int colAlignB = B->ColAlign();
    //const Int rowAlignB = B->RowAlign();
    const Int sizeB = B->Grid().VCSize();


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    const Int posInGrid = A.Grid().VCRank();


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3 
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    //std::vector<bool> require_data(sizeA,false);
    std::vector<int> index_to_put(sizeA,-1);
    //std::vector<int> index_from(sizeA,-1);
    



    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {       
        index_to_put[i] = i;
    }


    //SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    //SyncInfo<D2> syncInfoRecvTransosedMatrix= SyncInfoFromMatrix(recvTransposedMatrix);


    //const bool inAGrid = A.Participating();
    //const bool inBGrid = indexB >=0 ? true:false;

    //const Int maxSendSize = mLocA * nLocA;

    
    //const int myLocalDataRankA = int(std::floor(posInGrid/sizeB));
    int partialHeight = int(splitDim/ scatterCommSize);
    int partialChildHeight = int(partialHeight / numMatricesInSubGrid);
    const Int localSubgridHeight = int(splitDim/ scatterCommSize);
    const Int recvLocalHeight =  (mLocA/splitDim) * (localSubgridHeight/numMatricesInSubGrid);
    int sendWidth = int(n / rowLCM);

    std::vector<Matrix<T,D2>> conversionMatrixVector;
    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        conversionMatrixVector.push_back(Matrix<T,D2>(recvLocalHeight, sendWidth));
    }




    for(Int localDataRankA = 0; localDataRankA < int(rowLCM/sizeB); localDataRankA++)
    {

        //comm is useless parameter in this function 
        //Aluminum infer comm from sync object 
        // mpi::Scatter<T, D2,double, double, double, double, double>((T *)transposedMatrix.Buffer(), sendCounts, (T *)recvTransposedMatrix.Buffer(), sendCounts, localDataRankA, ScatterComm,
        //        syncGeneral);
        // mpi::Scatter<T, D2,double>((T *)transposedMatrix.Buffer(), sendCounts, (T *)recvTransposedMatrix.Buffer(), sendCounts, localDataRankA, ScatterComm,
        //        syncGeneral);

        //MPI_Scatter((void *)transposedMatrix.Buffer(), sendCounts, mpi::TypeMap<T>(), (void *)recvTransposedMatrix.Buffer(), sendCounts, mpi::TypeMap<T>(), localDataRankA, ScatterComm.GetMPIComm());
        //Synchronize(syncInfoRecvTransosedMatrix);

        MPI_Scatter((void *)sendBuf, sendCounts, mpi::TypeMap<T>(), (void *)recvTransposedMatrix.Buffer(), sendCounts, mpi::TypeMap<T>(), localDataRankA, ScatterComm.GetMPIComm());

         // mpi::Scatter(sendBuf, sendCounts, recvTransposedMatrix.Buffer(), sendCounts, localDataRankA, ScatterComm,
         //        syncGeneral);

        Synchronize(syncGeneral);

        

        for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
        {
            //Transpose(recvTransposedMatrix(Range<int>(partialChildHeight*childLayerSubGrid, partialChildHeight*(childLayerSubGrid+1)), Range<int>(0,n)),conversionMatrix);
            //Transpose(recvTransposedMatrix( Range<Int>(0,sendWidth*(mLocA/splitDim)),Range<Int>(partialChildHeight*childLayerSubGrid, partialChildHeight*(childLayerSubGrid+1))),conversionMatrix);
            //dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(0,localDataRankA)
            const Int memoryOffSet = (mLocA/splitDim) * sendWidth * (localSubgridHeight/numMatricesInSubGrid) * childLayerSubGrid;
            
            //DistMatrix<T,STAR,VC,ELEMENT,D2>* B_temp = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]));

            //SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->LockedMatrix());
            

            // copy::util::InterleaveMatrix(
            //             recvLocalHeight, sendWidth,
            //             recvBuf + memoryOffSet , 1, recvLocalHeight,
            //             B_temp->Buffer(0,localDataRankA),
            //             sendWidth, 1,
            //             syncGeneral);

            // copy::util::InterleaveMatrix(
            //             recvLocalHeight, sendWidth,
            //             recvBuf + memoryOffSet , 1, recvLocalHeight,
            //             B_temp->Buffer(0,localDataRankA),
            //             1, recvLocalHeight,
            //             syncGeneral);

            // hydrogen::gpu_blas::Copy<T, T, int>(

            //     TransposeMode::TRANSPOSE,

            //     recvLocalHeight, sendWidth,

            //     recvBuf + memoryOffSet, recvLocalHeight,

            //     B_temp->Buffer(0,localDataRankA), sendWidth,

            //     syncGeneral);

            // cudaDeviceSynchronize();

            Transpose(recvTransposedMatrix( Range<Int>(0,sendWidth*(mLocA/splitDim)),Range<Int>(partialChildHeight*childLayerSubGrid, partialChildHeight*(childLayerSubGrid+1))),conversionMatrixVector[childLayerSubGrid]);

            // std::cout<<"Send width is  "<<sendWidth<<"\n";

            // copy::util::InterleaveMatrix(
            //             recvLocalHeight, sendWidth,
            //             recvTransposedMatrix.Buffer() + memoryOffSet , 1, recvLocalHeight,
            //             conversionMatrixVector[childLayerSubGrid].Buffer(),
            //             sendWidth, 1,
            //             syncInfoBVector[childLayerSubGrid]);


            //dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(0,localDataRankA)
            copy::util::InterleaveMatrix(
                        partialChildHeight*(mLocA/splitDim), sendWidth,
                        conversionMatrixVector[childLayerSubGrid].Buffer()  , 1, partialChildHeight*(mLocA/splitDim),
                        dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(0,localDataRankA),
                        1, dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->LDim() * int(rowLCM/sizeB),
                        syncInfoBVector[childLayerSubGrid]);

            
            //syncInfoBVector[childLayerSubGrid]
        }

        
        //Synchronize(syncInfoRecvTransosedMatrix);



    }
    //Synchronize(syncInfoG);
    //cudaDeviceSynchronize();

}


template<typename T, Device D1, Device D2>
void TranslateBetweenGridsSliceGatherOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Scatter data in Column-Major ordering along the last dimension 
    Last dimension should be divisible number of child layers
    Size of B_vector is equal to the number of child layers 

    Subgrids in B_vector are assumed to be subset of resources in A grid 


    Resources are assumed to be distribted equally among different subgrids 
    

    */
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();





    
    const Int numChildLayers = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numChildLayers;++i)
    {
        B_Vector[i]->Resize(int(m/numChildLayers),n);
    }

    
    if(m%splitDim != 0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "feature dimension should be divisible by split dimension");
    }
    if(splitDim%numChildLayers!=0)
    {
        LogicError("TranslateBetweenGridsScatterOptComm: ",
                   "Split dimension must be divisible by number of children layers or number of splits");
    }



    //const Int myRankViewing = mpi::Rank(viewingCommA);
    const Int gatherCommSize = mpi::Size( gatherComm );
    const int sendCounts = int((mLocA*nLocA)/gatherCommSize);
    const int numMatricesInSubGrid  = int(numChildLayers / gatherCommSize);

    std::vector<Int> indexBVec;
    std::vector<SyncInfo<D2>> syncInfoBVector;

    for(Int i = 0; i<numChildLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);
            syncInfoBVector.push_back(SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[i]))->LockedMatrix()) );

        }
    }
    const Int indexB = indexBVec[0];
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    Matrix<T,D2>  transposedMatrix(int((mLocA*nLocA)/splitDim),splitDim), recvTransposedMatrix(int((mLocA*nLocA)/splitDim),int(splitDim/gatherCommSize)), sendTransposedMatrix(nLocA,mLocA);

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());

    const Int maxSendSize = mLocA*nLocA;
    const Int maxRecvSize = int(mLocA*nLocA*gatherCommSize);

    simple_buffer<T,D2> send_buf(maxSendSize, syncInfoA);
    simple_buffer<T,D2> recv_buf(maxRecvSize, syncGeneral);
    simple_buffer<T,D2> temp_buf((mLocA/numChildLayers)*nLocA*gatherCommSize, syncGeneral);

    T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();
    T* tempBuf = temp_buf.data();


    Transpose(A.LockedMatrix(),sendTransposedMatrix);
    // copy::util::InterleaveMatrix(
    //                     mLocA, nLocA,
    //                     A.LockedBuffer()  , 1, mLocA,
    //                     sendBuf,
    //                     nLocA,1,
    //                     syncInfoA);


    const Int rowStrideB = B->RowStride();

    const Int sizeB = B->Grid().VCSize();


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    const Int posInGrid = A.Grid().VCRank();
    const Int subgridNumber = int(std::floor(posInGrid/sizeB));



    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3 
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    //std::vector<bool> require_data(sizeA,false);
    std::vector<int> index_to_put(sizeA,-1);
    //std::vector<int> index_from(sizeA,-1);
    



    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {       
        index_to_put[i] = i;
    }



    int partialHeight = int(splitDim/ gatherCommSize);
    int partialChildHeight = int(partialHeight / numMatricesInSubGrid);
    const Int localSubgridHeight = int(splitDim/ gatherCommSize);
    const Int recvLocalHeight =  (mLocA/splitDim) * (localSubgridHeight/numMatricesInSubGrid);
    int sendWidth = int(n / rowLCM);
    const Int perSubgridSplitHeight = splitDim / gatherCommSize;
    const Int childLayerSplitHeight = perSubgridSplitHeight / numMatricesInSubGrid;

    std::vector<Matrix<T,D2>> conversionMatrixVector;
    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        conversionMatrixVector.push_back(Matrix<T,D2>(childLayerSplitHeight * nLocA * (mLocA/splitDim), gatherCommSize));
    }


    mpi::AllGather(sendTransposedMatrix.Buffer(), mLocA*nLocA, 
                    recvBuf, mLocA*nLocA, 
                    gatherComm, syncGeneral);

    

    Matrix<T,D2>  tempMatrix(gatherCommSize , childLayerSplitHeight * nLocA * (mLocA/splitDim));

    for(Int childLayerSubGrid = 0; childLayerSubGrid < numMatricesInSubGrid; ++childLayerSubGrid)
    {
        const Int colNumberChildLayer = childLayerSubGrid + (numMatricesInSubGrid * subgridNumber);

        //strieded copy to get appropriate rows (scattering the data among child layers)
        copy::util::InterleaveMatrix(
            childLayerSplitHeight * nLocA, (mLocA/splitDim)*gatherCommSize,
            recvBuf + colNumberChildLayer * childLayerSplitHeight * nLocA  , 1, childLayerSplitHeight * nLocA * numChildLayers,
            conversionMatrixVector[childLayerSubGrid].Buffer(),
            1, childLayerSplitHeight * nLocA,
            syncGeneral);
        Transpose(conversionMatrixVector[childLayerSubGrid], tempMatrix);


        // copy::util::InterleaveMatrix(
        //     childLayerSplitHeight * nLocA * (mLocA/splitDim), gatherCommSize,
        //     conversionMatrixVector[childLayerSubGrid].Buffer()  , 1, childLayerSplitHeight * nLocA * (mLocA/splitDim),
        //     tempBuf,
        //     gatherCommSize, 1,
        //     syncGeneral);

        // copy::util::InterleaveMatrix(
        //     nLocA * gatherCommSize , childLayerSplitHeight * (mLocA/splitDim),
        //     tempBuf  , 1, nLocA * gatherCommSize,
        //     dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(),
        //     childLayerSplitHeight * (mLocA/splitDim), 1,
        //     syncGeneral);
        tempMatrix.Resize(nLocA * gatherCommSize , childLayerSplitHeight * (mLocA/splitDim));

        Transpose(tempMatrix,dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Matrix());

        // copy::util::InterleaveMatrix(
        //     nLocA * gatherCommSize , childLayerSplitHeight * (mLocA/splitDim),
        //     tempMatrix.LockedBuffer()  , 1, nLocA * gatherCommSize,
        //     dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[childLayerSubGrid]]))->Buffer(),
        //     childLayerSplitHeight * (mLocA/splitDim), 1,
        //     syncGeneral);

    }




    


}



template<typename T, Device D1, Device D2>
void TranslateBetweenGridsGatherComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Gather data in Column-Major ordering along the last dimension 
    
    Size of B_vector is equal to the number of parent layers 

    Subgrids in B_vector are assumed to be subset of resources in A grid 

    Resources are assumed to be distribted equally among different subgrids 
    

    */
    EL_DEBUG_CSE

    std::vector<Int> indexBVec;
    const Int numParentLayers = int(B_Vector.size());

    for(Int i = 0; i<numParentLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);

        }
    }
    const Int indexB = indexBVec[0];

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    

    //const Int colStrideB = B->ColStride();
    //const Int rowStrideB = B->RowStride();
    //const Int colShiftB = B->ColShift();
    //const Int rowShiftB = B->RowShift();
    //const Int colAlignB = B->ColAlign();
    //const Int rowAlignB = B->RowAlign();
    const Int sizeB = B->Grid().VCSize();

    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();



    //mpi::Comm const& viewingCommB = B->Grid().ViewingComm();
    //mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    //Int rowStrideA = A.RowStride();
    const Int posInGrid = A.Grid().VCRank();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();


    
    const Int sizeA = A.Grid().VCSize();

     const Int index_from = int(std::floor(posInGrid/sizeB));

    A.Resize(m*numParentLayers,n); 

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    //SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());

    


    //const Int myRankViewing = mpi::Rank(viewingCommA);
    const Int gatherCommSize = mpi::Size( gatherComm );
    const int numMatricesInSubGrid  = int(numParentLayers / gatherCommSize);

    //mlocB and m should be equal
    if(mLocB!=m)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "mLocB and m should be same");

    }
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "Width must be divisible by number of resources in A matrix");
    }
    if(mLocB%splitDim!=0)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "Height in B matrix must be divisible by splitDim");

    }
    const int totalSizeComm = mLocB * numParentLayers * int(n/sizeB);
    
    const int maxSendSize = mLocB*nLocB * numMatricesInSubGrid;

    simple_buffer<T,D2> send_buf(maxSendSize, syncInfoA);  
    T* sendBuf = send_buf.data();


    Matrix<T,D1> conversionMatrix(mLocB,nLocB), transposedMatrix(int((mLocB*nLocB)/splitDim),splitDim), recvTransposedMatrix(int(totalSizeComm/(splitDim*numParentLayers)),splitDim*numParentLayers);

    //SyncInfo<D2> syncInfoConversionMatrix= SyncInfoFromMatrix(conversionMatrix);
    for(Int parentLayerSubGrid = 0; parentLayerSubGrid < numMatricesInSubGrid; ++parentLayerSubGrid)
    {
        Copy(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[parentLayerSubGrid]]))->LockedMatrix(), conversionMatrix);
        conversionMatrix.Resize(splitDim, int((mLocB*nLocB)/splitDim));
        Transpose(conversionMatrix,transposedMatrix);
        copy::util::InterleaveMatrix(
            int((mLocB*nLocB)/splitDim),splitDim,
            transposedMatrix.Buffer()  , 1, int((mLocB*nLocB)/splitDim),
            sendBuf + parentLayerSubGrid*mLocB*nLocB,
            1, int((mLocB*nLocB)/splitDim),
            syncGeneral);

    }

    mpi::AllGather(sendBuf, mLocB*nLocB*numMatricesInSubGrid, 
                    recvTransposedMatrix.Buffer(), mLocB*nLocB*numMatricesInSubGrid, 
                    gatherComm, syncGeneral);



    Matrix<T,D1> resizedRecvMatrix(m*numParentLayers,int(n/sizeB));

    Transpose(recvTransposedMatrix, resizedRecvMatrix);

    copy::util::InterleaveMatrix(
                        m*numParentLayers, int(n/sizeA),
                        resizedRecvMatrix.Buffer() + index_from *m*numParentLayers , 1, m*numParentLayers*(sizeA/sizeB),
                        A.Buffer(0,0),
                        1, m*numParentLayers,
                        syncInfoA);

    // copy::util::InterleaveMatrix(
    //                     m*numParentLayers, int(n/sizeA),
    //                     resizedRecvMatrix.Buffer() + index_from *m*numParentLayers , 1, m*numParentLayers,
    //                     A.Buffer(0,0),
    //                     1, m*numParentLayers,
    //                     syncInfoA);


}


template<typename T, Device D1, Device D2>
void TranslateBetweenGridsGatherOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> & A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector, int splitDim,  mpi::Comm const& gatherComm, SyncInfo<D1> & syncGeneral)
{
    /*
    Gather data in Column-Major ordering along the last dimension 
    
    Size of B_vector is equal to the number of parent layers 

    Subgrids in B_vector are assumed to be subset of resources in A grid 

    Resources are assumed to be distribted equally among different subgrids 
    

    */
    // This Function has some bugs in Interleave function 
    EL_DEBUG_CSE

    std::vector<Int> indexBVec;
    std::vector<SyncInfo<D2>> syncInfoBVector;
    const Int numParentLayers = int(B_Vector.size());



    for(Int i = 0; i<numParentLayers; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            indexBVec.push_back(i);
            syncInfoBVector.push_back(SyncInfoFromMatrix(dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[i]))->LockedMatrix()) );


        }
    }
    const Int indexB = indexBVec[0];

    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));
    

    //const Int colStrideB = B->ColStride();
    //const Int rowStrideB = B->RowStride();
    //const Int colShiftB = B->ColShift();
    //const Int rowShiftB = B->RowShift();
    //const Int colAlignB = B->ColAlign();
    //const Int rowAlignB = B->RowAlign();
    const Int sizeB = B->Grid().VCSize();

    const Int m = B->Height();
    const Int n = B->Width();
    const Int mLocB = B->LocalHeight();
    const Int nLocB = B->LocalWidth();



    //mpi::Comm const& viewingCommB = B->Grid().ViewingComm();
    //mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    //Int rowStrideA = A.RowStride();
    const Int posInGrid = A.Grid().VCRank();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();


    
    const Int sizeA = A.Grid().VCSize();

     const Int index_from = int(std::floor(posInGrid/sizeB));

    A.Resize(m*numParentLayers,n); 

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    //SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());

    


    //const Int myRankViewing = mpi::Rank(viewingCommA);
    const Int gatherCommSize = mpi::Size( gatherComm );
    const int numMatricesInSubGrid  = int(numParentLayers / gatherCommSize);

    //mlocB and m should be equal
    if(mLocB!=m)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "mLocB and m should be same");

    }
    if(n%sizeA!=0)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "Width must be divisible by number of resources in A matrix");
    }
    if(mLocB%splitDim!=0)
    {
        LogicError("TranslateBetweenGridsGatherOptComm: ",
                   "Height in B matrix must be divisible by splitDim");

    }
    const int totalSizeComm = mLocB * numParentLayers * int(n/sizeB);
    
    const int maxSendSize = mLocB*nLocB * numMatricesInSubGrid;

    simple_buffer<T,D2> send_buf(maxSendSize, syncGeneral);  
    T* sendBuf = send_buf.data();


    Matrix<T,D1> conversionMatrix(mLocB,nLocB), transposedMatrix(int((mLocB*nLocB)/splitDim),splitDim), recvTransposedMatrix(int(totalSizeComm/(splitDim*numParentLayers)),splitDim*numParentLayers);

    //SyncInfo<D2> syncInfoConversionMatrix= SyncInfoFromMatrix(conversionMatrix);
    for(Int parentLayerSubGrid = 0; parentLayerSubGrid < numMatricesInSubGrid; ++parentLayerSubGrid)
    {
        
        copy::util::InterleaveMatrix(
            splitDim, int((mLocB*nLocB)/splitDim),
            dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexBVec[parentLayerSubGrid]]))->LockedBuffer()  , 1, splitDim,
            sendBuf + parentLayerSubGrid*mLocB*nLocB,
            int((mLocB*nLocB)/splitDim), 1, 
            syncInfoBVector[parentLayerSubGrid]);

    }
    Synchronize(syncGeneral);

    mpi::AllGather(sendBuf, mLocB*nLocB*numMatricesInSubGrid, 
                    recvTransposedMatrix.Buffer(), mLocB*nLocB*numMatricesInSubGrid, 
                    gatherComm, syncGeneral);



    Matrix<T,D1> resizedRecvMatrix(m*numParentLayers,int(n/sizeB));

    Transpose(recvTransposedMatrix, resizedRecvMatrix);

    copy::util::InterleaveMatrix(
                        m*numParentLayers, int(n/sizeA),
                        resizedRecvMatrix.Buffer() + index_from *m*numParentLayers , 1, m*numParentLayers*(sizeA/sizeB),
                        A.Buffer(0,0),
                        1, m*numParentLayers,
                        syncInfoA);

    // copy::util::InterleaveMatrix(
    //                     m*numParentLayers, int(n/sizeA),
    //                     resizedRecvMatrix.Buffer() + index_from *m*numParentLayers , 1, m*numParentLayers,
    //                     A.Buffer(0,0),
    //                     1, m*numParentLayers,
    //                     syncInfoA);


}

template<typename T, Device D1, Device D2>
void TranslateBetweenGridsBroadcastOptComm
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector,  mpi::Comm const& broadcastComm, SyncInfo<D1> & syncGeneral)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    This function is specific to the LBANN with implementation for specific cases
    Subgrids in B_vector are assumed to be subset of resources in A grid 
    */
    

    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    //mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();

    
    const Int numSubGrids = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numSubGrids;++i)
    {
        B_Vector[i]->Resize(m,n);
    }

    


    //const Int myRankViewing = mpi::Rank(viewingCommA);

    Int indexB = -1;

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            if(indexB!=-1)
            {
                std::printf("Error: rank is in multiple subgrids\n");
            }
            indexB = i;
        }
    }
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));

    const Int posInSubGrid = B->Grid().VCRank(); 

    //const Int colStrideB = B->ColStride();
    const Int rowStrideB = B->RowStride();
    //const Int colShiftB = B->ColShift();
    //const Int rowShiftB = B->RowShift();
    //const Int colAlignB = B->ColAlign();
    //const Int rowAlignB = B->RowAlign();
    const Int sizeB = B->Grid().VCSize();


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    const Int posInGrid = A.Grid().VCRank();


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3 
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    //std::vector<bool> require_data(sizeA,false);
    std::vector<int> index_to_put(sizeA,-1);
    //std::vector<int> index_from(sizeA,-1);
    //Int temp_require_data = posInSubGrid;
    //double total_iter = posInSubGrid;



    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {       
        index_to_put[i] = i;
    }


    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());
    //SyncInfo<D1> syncGeneral = SyncInfo<D1>();

    //

    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocA * nLocA;
    simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);
    T* sendBuf = send_buf.data();
    //T* recvBuf = recv_buf.data();

    
    const int myLocalDataRankA = int(std::floor(posInGrid/sizeB));

    for(Int localDataRankA = 0; localDataRankA < int(rowLCM/sizeB); localDataRankA++)
    {

        if(myLocalDataRankA==localDataRankA && inAGrid)
        {
            copy::util::InterleaveMatrix(
                mLocA, nLocA,
                A.LockedBuffer(0,0),
                1, A.LDim(),
                sendBuf, 1, mLocA, syncInfoA);

        }
        //comm is useless parameter in this function 
        //Aluminum infer comm from sunc object 
        Broadcast(sendBuf, mLocA*nLocA, localDataRankA, broadcastComm,
               syncGeneral);

        Synchronize(syncGeneral);

        
        
        int sendWidth = int(n / rowLCM);
        copy::util::InterleaveMatrix(
                        m, sendWidth,
                        sendBuf  , 1, m*(rowLCM/sizeA),
                        B->Buffer(0,index_to_put[localDataRankA]),
                        1, (rowLCM/sizeB)*B->LDim(),
                        syncInfoB);

        
        Synchronize(syncInfoB);



    }

}


template<typename T, Device D1, Device D2>
void TranslateBetweenGridsBroadcast
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  std::vector<std::unique_ptr<AbstractDistMatrix<T>>>& B_Vector)
{
    //<T,STAR,VC,ELEMENT,D2>
    /*
    This function is specific to the LBANN with implementation for specific cases
    Subgrids in B_vector are assumed to be subset of resources in A grid 
    */
    

    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    //mpi::Group owningGroupA = A.Grid().OwningGroup();

    //const Int colRankA = A.ColRank();
    //const Int rowRankA = A.RowRank();
    //Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    //Int colAlignA = A.ColAlign();
    //Int rowAlignA = A.RowAlign();

    
    const Int numSubGrids = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numSubGrids;++i)
    {
        B_Vector[i]->Resize(m,n);
    }

    


    //const Int myRankViewing = mpi::Rank(viewingCommA);

    Int indexB = -1;

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if(B_Vector[i]->Participating())
        {
            if(indexB!=-1)
            {
                std::printf("Error: rank is in multiple subgrids\n");
            }
            indexB = i;
        }
    }
    DistMatrix<T,STAR,VC,ELEMENT,D2>* B = dynamic_cast<DistMatrix<T,STAR,VC,ELEMENT,D2>*>( &(*B_Vector[indexB]));

    const Int posInSubGrid = B->Grid().VCRank(); 

    //const Int colStrideB = B->ColStride();
    const Int rowStrideB = B->RowStride();
    //const Int colShiftB = B->ColShift();
    //const Int rowShiftB = B->RowShift();
    //const Int colAlignB = B->ColAlign();
    //const Int rowAlignB = B->RowAlign();
    const Int sizeB = B->Grid().VCSize();


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;

    const Int myLocalRankA = A.Grid().VCRank();


    // Parent Subgrid Size: 4 Child Subgrid Size: 3
    // Parent 0 1 2 3 0 1 2 3 0 1 2 3 
    // Child  0 1 2 0 1 2 0 1 2 0 1 2

    std::vector<bool> require_data(sizeA,false);
    std::vector<int> index_to_put(sizeA,-1);
    std::vector<int> index_from(sizeA,-1);
    Int temp_require_data = posInSubGrid;
    double total_iter = posInSubGrid;



    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {
        if(require_data[temp_require_data]==true)
        {
            LogicError("TranslateBetweenGridsBroadcast: ",
                   "Cannot receive input from  same rank twice");
        }
        require_data[temp_require_data] = true;
        index_to_put[temp_require_data] = i;
        index_from[temp_require_data] = int(std::floor(total_iter/sizeA));
        //std::printf("Subgrid B rank:%d Subgrid A rank:%d  Require data from rank:%d Index to put:%d\n",posInSubGrid, myLocalRankA, temp_require_data, i );
        total_iter = total_iter + sizeB;
        temp_require_data =  Mod(temp_require_data + sizeB, sizeA);
        
        
    }

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B->LockedMatrix());
    SyncInfo<D1> syncGeneral = SyncInfo<D1>();

    //

    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocA * nLocA;
    simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);
    T* sendBuf = send_buf.data();
    //T* recvBuf = recv_buf.data();

    


    for(Int localRankA = 0; localRankA < sizeA; localRankA++)
    {
        if(myLocalRankA==localRankA && inAGrid)
        {
            copy::util::InterleaveMatrix(
                mLocA, nLocA,
                A.LockedBuffer(0,0),
                1, A.LDim(),
                sendBuf, 1, mLocA, syncInfoA);

        }
        //comm is useless parameter in this function 
        //Aluminum infer comm from sunc object 
        Broadcast(sendBuf, mLocA*nLocA, localRankA, viewingCommA,
               syncInfoA);

        Synchronize(syncGeneral);

        if(require_data[localRankA])
        {
            int sendWidth = int(n / rowLCM);
            copy::util::InterleaveMatrix(
                            m, sendWidth,
                            sendBuf + (index_from[localRankA] * m ) , 1, m*(rowLCM/sizeA),
                            B->Buffer(0,index_to_put[localRankA]),
                            1, (rowLCM/sizeB)*B->LDim(),
                            syncInfoB);

        }
        Synchronize(syncInfoB);



    }







}

template void TranslateBetweenGridsAllreduce <double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& );
template void TranslateBetweenGridsAllreduce <double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& );
template void TranslateBetweenGridsAllreduceOpt <double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& );
template void TranslateBetweenGridsAllreduceOpt <double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& );

template void TranslateBetweenGridsAllreduceOptComm <double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>&, mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsAllreduceOptComm <double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>&, mpi::Comm const& , SyncInfo<Device::GPU> & );

template void TranslateBetweenGridsBroadcast <double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& );
template void TranslateBetweenGridsBroadcast <double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& );

template void TranslateBetweenGridsBroadcastOptComm<double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsBroadcastOptComm<double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& ,  mpi::Comm const& , SyncInfo<Device::GPU> & );

template void TranslateBetweenGridsAllreduce <float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& );
template void TranslateBetweenGridsAllreduce <float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& );
template void TranslateBetweenGridsAllreduceOpt <float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& );
template void TranslateBetweenGridsAllreduceOpt <float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& );

template void TranslateBetweenGridsAllreduceOptComm <float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>&, mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsAllreduceOptComm <float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>&, mpi::Comm const& , SyncInfo<Device::GPU> & );

template void TranslateBetweenGridsBroadcast <float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& );
template void TranslateBetweenGridsBroadcast <float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& );

template void TranslateBetweenGridsBroadcastOptComm<float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsBroadcastOptComm<float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& ,  mpi::Comm const& , SyncInfo<Device::GPU> & );


//scatter Gather 
template void TranslateBetweenGridsScatterOptComm<double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsGatherOptComm<double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsScatterOptComm<double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );
template void TranslateBetweenGridsGatherOptComm<double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );


template void TranslateBetweenGridsScatterOptComm<float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsGatherOptComm<float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsScatterOptComm<float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );
template void TranslateBetweenGridsGatherOptComm<float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );


template void TranslateBetweenGridsScatterComm<double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsGatherComm<double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsScatterComm<double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );
template void TranslateBetweenGridsGatherComm<double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );
template void TranslateBetweenGridsSliceGatherOptComm<double, Device::CPU,Device::CPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsSliceGatherOptComm<double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<double>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );


template void TranslateBetweenGridsScatterComm<float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsGatherComm<float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsScatterComm<float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );
template void TranslateBetweenGridsGatherComm<float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> & , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );
template void TranslateBetweenGridsSliceGatherOptComm<float, Device::CPU,Device::CPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::CPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::CPU> & );
template void TranslateBetweenGridsSliceGatherOptComm<float, Device::GPU,Device::GPU> (DistMatrix<float,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<std::unique_ptr<AbstractDistMatrix<float>>>& , int ,  mpi::Comm const& , SyncInfo<Device::GPU> & );


/*
//template<typename T, Device D1, Device D2>
void TranslateBetweenGridsBroadcast
(DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> const& A,
  std::vector<DistMatrix<double,STAR,VC,ELEMENT,Device::CPU>>& B_Vector)
{
    //<T,STAR,VC,ELEMENT,D2>

    This function is specific to the LBANN with implementation for specific cases
    Subgrids in B_vector are assumed to be subset of resources in A grid 


    EL_DEBUG_CSE;
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    mpi::Group owningGroupA = A.Grid().OwningGroup();

    const Int colRankA = A.ColRank();
    const Int rowRankA = A.RowRank();
    Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    Int colAlignA = A.ColAlign();
    Int rowAlignA = A.RowAlign();

    
    const Int numSubGrids = int(B_Vector.size());
    const Int sizeA = A.Grid().VCSize();

    for(Int i=0; i<numSubGrids;++i)
    {
        B_Vector[i].Resize(m,n);
    }

    


    const Int myRankViewing = mpi::Rank(viewingCommA);

    Int indexB = -1;

    for(Int i = 0; i<numSubGrids; ++i)
    {
        if(B_Vector[i].Participating())
        {
            if(indexB!=-1)
            {
                std::printf("Error: rank is in multiple subgrids\n");
            }
            indexB = i;
        }
    }

    const Int posInSubGrid = B_Vector[indexB].Grid().VCRank(); 

    const Int colStrideB = B_Vector[indexB].ColStride();
    const Int rowStrideB = B_Vector[indexB].RowStride();
    const Int colShiftB = B_Vector[indexB].ColShift();
    const Int rowShiftB = B_Vector[indexB].RowShift();
    const Int colAlignB = B_Vector[indexB].ColAlign();
    const Int rowAlignB = B_Vector[indexB].RowAlign();
    const Int sizeB = B_Vector[indexB].Grid().VCSize();


    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;


    std::vector<bool> require_data(sizeA,false);
    std::vector<int> index_to_put(sizeA,-1);
    Int temp_require_data = posInSubGrid;

    for(Int i = 0; i < int(rowLCM/sizeB); ++i)
    {
        if(require_data[temp_require_data]==true)
        {
            LogicError("TranslateBetweenGridsBroadcast: ",
                   "Cannot receive input from  same rank twice");
        }
        require_data[temp_require_data] = true;
        temp_require_data =  Mod(temp_require_data + sizeB, sizeA);
        index_to_put[temp_require_data] = i;
    }

    SyncInfo<Device::CPU> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<Device::CPU> syncInfoB = SyncInfoFromMatrix(B_Vector[indexB].LockedMatrix());
    SyncInfo<Device::CPU> syncGeneral = SyncInfo<Device::CPU>();

    //

    const bool inAGrid = A.Participating();
    const bool inBGrid = indexB >=0 ? true:false;

    const Int maxSendSize = mLocA * nLocA;
    simple_buffer<double,Device::CPU> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<double,Device::CPU> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);
    double* sendBuf = send_buf.data();
    double* recvBuf = recv_buf.data();

    const Int myLocalRankA = A.Grid().VCRank();


    for(Int localRankA = 0; localRankA < sizeA; localRankA++)
    {
        if(myLocalRankA==localRankA && inAGrid)
        {
            copy::util::InterleaveMatrix(
                mLocA, nLocA,
                A.LockedBuffer(0,0),
                1, A.LDim(),
                sendBuf, 1, mLocA, syncInfoA);

        }
        //comm is useless parameter in this function 
        //Aluminum infer comm from sunc object 
        Broadcast(sendBuf, mLocA*nLocA, myLocalRankA, viewingCommA,
               syncInfoA);

        Synchronize(syncGeneral);

        if(require_data[localRankA])
        {
            int sendWidth = int(n / rowLCM);
            copy::util::InterleaveMatrix(
                            m, sendWidth,
                            recvBuf, 1, m,
                            B_Vector[indexB].Buffer(0,index_to_put[localRankA]),
                            1, (rowLCM/rowStrideB)*B_Vector[indexB].LDim(),
                            syncInfoB);

        }
        Synchronize(syncInfoB);



    }







}
*/


/*

template<typename T, Device D1, Device D2>
void TranslateBetweenGrids
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  DistMatrix<T,STAR,VC,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE;
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    B.Resize(m, n);
    mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
    mpi::Group owningGroupA = A.Grid().OwningGroup();

    // Just need to ensure that each viewing comm contains the other team's
    // owning comm. Congruence is too strong.

    // Compute the number of process rows and columns that each process
    // needs to send to.
    const Int colStrideB = B.ColStride();
    const Int rowStrideB = B.RowStride();
    const Int colShiftB = B.ColShift();
    const Int rowShiftB = B.RowShift();
    const Int colRankB = B.ColRank();
    const Int rowRankB = B.RowRank();
    const Int colRankA = A.ColRank();
    const Int rowRankA = A.RowRank();
    const Int colStrideA = A.ColStride();
    const Int rowStrideA = A.RowStride();
    //const Int colGCD = GCD(colStrideB, colStrideA);
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    //const Int colLCM = colStrideB*colStrideA / colGCD;
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;
    //const Int numColSends = colStrideB / colGCD;
    const Int numRowSends = rowLCM / rowStrideA ;
    const Int numRowRecvs = rowLCM / rowStrideB;

    const Int colAlignA = A.ColAlign();
    const Int rowAlignA = A.RowAlign();
    const Int colAlignB = B.ColAlign();
    const Int rowAlignB = B.RowAlign();

    const Int myRankViewing = mpi::Rank(viewingCommB);

    const bool inBGrid = B.Participating();
    const bool inAGrid = A.Participating();



    const Int rankBRecv = Mod(B.Grid().Rank(), rowStrideA);



    //Setup for receiving data in B
    const Int sendColOffset = colAlignA;
    const Int recvColOffset =
      Mod(colAlignB,colStrideB);
    const Int sendRowOffset = rowAlignA;
    const Int recvRowOffset =
      Mod(0*rowStrideA+rowAlignB,rowStrideB);

    const Int colShift = Mod(colRankB-recvColOffset, colStrideB);
    const Int rowShift = Mod(rowRankB-recvRowOffset, rowStrideB);

    
    std::printf("B Grid Rank: %d\n",B.Grid().Rank());

    const Int numInB = B.Grid().Rank();

    const Int firstSendRow = Mod(colShift+sendColOffset,colStrideA);
    const Int firstSendCol = Mod(rowShift+sendRowOffset,rowStrideA);

    const Int numColRecvs = Length(colStrideA,colShift,colStrideB);
    Int sendCol = firstSendCol;
    //const Int numRowRecvs = Length(rowStrideA,rowShift,rowStrideB);

    // Recv data
    // For now, simply receive sequentially. Until we switch to
    // nonblocking recv's, we won't be using much of the
    // recvBuf
    Int sendRow = firstSendRow;

    if(!inBGrid && !inAGrid)
        return;

    const Int maxSendSize =
      (n/(rowStrideA*numRowSends)+1) * (m);

    
    // Translate the ranks from A's VC communicator to B's viewing so that
    // we can match send/recv communicators. Since A's VC communicator is not
    // necessarily defined on every process, we instead work with A's owning
    // group and account for row-major ordering if necessary.
    const int sizeA = A.Grid().Size();
    vector<int> rankMap(sizeA), ranks(sizeA);
    if(A.Grid().Order() == COLUMN_MAJOR)
    {
        for(int j=0; j<sizeA; ++j)
            ranks[j] = j;
    }
    else
    {
        // The (i,j) = i + j*colStrideA rank in the column-major ordering is
        // equal to the j + i*rowStrideA rank in a row-major ordering.
        // Since we desire rankMap[i+j*colStrideA] to correspond to process
        // (i,j) in A's grid's rank in this viewing group, ranks[i+j*colStrideA]
        // should correspond to process (i,j) in A's owning group. Since the
        // owning group is ordered row-major in this case, its rank is
        // j+i*rowStrideA. Note that setting
        // ranks[j+i*rowStrideA] = i+j*colStrideA is *NOT* valid.
        for(int i=0; i<colStrideA; ++i)
            for(int j=0; j<rowStrideA; ++j)
                ranks[i+j*colStrideA] = j+i*rowStrideA;
    }
    mpi::Translate(
        owningGroupA, sizeA, ranks.data(), viewingCommB, rankMap.data());

    Int requiredMemory = 0;
    if(inAGrid)
        requiredMemory += maxSendSize;
    if(inBGrid)
        requiredMemory += maxSendSize;

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B.LockedMatrix());

    
    std::vector<simple_buffer<T,D1>> sendBufVector(numRowSends);
    
    for(Int i=0; i<numRowSends; ++i)
    {
        sendBufVector[i].allocate(inAGrid ? maxSendSize : 0);

    }
    


    //simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);

    //T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();




    Int recvRow = 0;
    std::printf("rowStrideA %d rowStrideB %d numRowSends %d LCM %d\n",rowStrideA,rowStrideB,numRowSends,rowLCM);


    //Checking if process are in both A and B grids 
    for (Int rowSend = 0; rowSend < numRowSends; rowSend++)
    {
        const Int recvVCRank = Mod(A.Grid().Rank() + rowSend*rowStrideA, rowStrideB);
        const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);

        if(recvViewingRank==myRankViewing)
        {
            Int sendWidth = Length(nLocA,rowSend,numRowSends);

            Int rowRecv = 0;

            for(rowRecv = 0; rowRecv<numRowRecvs; ++rowRecv)
            {
                const Int sendVCRank = Mod((sendRow + rankBRecv),rowStrideA);
                sendRow = Mod(sendRow+rowStrideB,rowStrideA);
                if(rankMap[sendVCRank]==myRankViewing) break;
            }

            

            const Int recvWidth = ((rowRecv*rowStrideB + numInB)>= Mod(n,rowLCM)) ? floor(n/rowLCM) : floor(n/rowLCM)+1;
            copy::util::InterleaveMatrix(
                mLocA, sendWidth,
                A.LockedBuffer(0,rowSend),
                1, numRowSends*A.LDim(),
                B.Buffer(0,rowRecv),
                1, (numRowRecvs)*B.LDim(),
                syncInfoB);
            Synchronize(syncInfoA);
            Synchronize(syncInfoB);

            printf("APpyling same rank optimization\n");

        }

    }



    

    std::vector<mpi::Request<T>> sendRequests(numRowSends);
    std::vector<bool> sendRequestsUsed(numRowSends,false);
    for(Int rowSend=0; rowSend<numRowSends; ++rowSend)
    {
        Int recvCol = 0; // avoid compiler warnings...
        if(inAGrid)
            recvCol=Mod(Mod(rowRankA-rowAlignA,rowStrideA)+rowAlignB,rowStrideB);
        mpi::Request<T> sendRequest;

        const Int recvVCRank = Mod(A.Grid().Rank() + rowSend*rowStrideA, rowStrideB);
        const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);

        if(inAGrid && recvViewingRank!=myRankViewing)
        {
            printf("I am sending data\n");
            //Pack Data
            Int sendWidth = Length(nLocA,rowSend,numRowSends);
            //std::printf("sendWidth from send %d\n", sendWidth);
            copy::util::InterleaveMatrix(
                    mLocA, sendWidth,
                    A.LockedBuffer(0,rowSend),
                    1, numRowSends*A.LDim(),
                    sendBufVector[rowSend].data(), 1, mLocA, syncInfoA);


            Synchronize(syncInfoA);
            sendRequestsUsed[rowSend] = true;

            
            //std::printf("Row Rank %d colRank: %d Rank: %d sending to rank:%d\n",rowRankA,colRankA,mpi::Rank(viewingCommB), recvViewingRank);
            mpi::ISend
            (sendBufVector[rowSend].data(), mLocA*sendWidth, recvViewingRank,
              viewingCommB, sendRequests[rowSend]);


        }
        //recvRow = Mod(recvRow+colStrideA,colStride);

    }


    //start receiving data from other processes
    sendRow = firstSendRow;

    

    for(Int rowRecv=0; rowRecv<numRowRecvs; ++rowRecv)
    {

        const Int sendVCRank = Mod((sendRow + rankBRecv),rowStrideA);

        if(inBGrid && rankMap[sendVCRank]!=myRankViewing)
        {
            

            printf("I am recv data\n");
            const Int sendColShift =
              Shift(sendRow, colAlignA, colStrideA);
            //const Int sendHeight = Length(m, sendColShift, colLCM);
            const Int localColOffset =
              (sendColShift-colShiftB) / colStrideB;
            

            
            const Int sendRowShift =
              Shift(sendCol, rowAlignA, rowStrideA) +
              rowRecv*rowStrideA;
            //const Int sendWidth = Length(n, sendRowShift, rowLCM);
            const Int sendWidth = ((rowRecv*rowStrideB + numInB)>= Mod(n,rowLCM)) ? floor(n/rowLCM) : floor(n/rowLCM)+1;
            //std::printf("rowLCM %d n %d Mod %d LDIM %d  widthNonCorrect %d Condn %d\n", rowLCM,n,Mod(n,rowLCM),B.LDim(),floor(n/rowLCM),rowRecv*rowStrideB + numInB );
            
            const Int localRowOffset =
              (sendRowShift-rowShiftB) / rowStrideB;



            //const Int sendVCRank = sendRow+sendCol*colStrideA;
            
            //std::printf("sendWidth:%d mLocA: %d sendVCRank: %d rankMap[sendVCRank]: %d recvrank %d\n",sendWidth,mLocA,sendVCRank,rankMap[sendVCRank],mpi::Rank(viewingCommB));
            mpi::Recv(
                recvBuf, m*sendWidth, rankMap[sendVCRank],
                viewingCommB, syncInfoB);

            // Unpack the data
            copy::util::InterleaveMatrix(
                m, sendWidth,
                recvBuf, 1, m,
                B.Buffer(0,rowRecv),
                1, (numRowRecvs)*B.LDim(),
                syncInfoB);

            

            

        }
        // Set up the next send col
        sendCol = Mod(sendCol+rowStrideB,rowStrideA);
        sendRow = Mod(sendRow+rowStrideB,rowStrideA);

        


    }
    if(inAGrid)
        for (Int i=0;i<numRowSends;++i)
        {
            if(sendRequestsUsed[i])
            {
                mpi::Wait(sendRequests[i]);
            }
            
        }
    //std::printf("FINAL: RANK: %d\n",El::mpi::Rank(El::mpi::NewWorldComm()));

    //mpi::Barrier(viewingCommB);
    
    


}
*/





template<typename T, Device D1, Device D2>
void TranslateBetweenGrids
(DistMatrix<T,STAR,VC,ELEMENT,D1> const& A,
  DistMatrix<T,STAR,VC,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE;
    Int m = A.Height();
    Int n = A.Width();
    const Int mLocA = A.LocalHeight();
    const Int nLocA = A.LocalWidth();

    
    mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
    mpi::Group owningGroupA = A.Grid().OwningGroup();

    // Just need to ensure that each viewing comm contains the other team's
    // owning comm. Congruence is too strong.

    // Compute the number of process rows and columns that each process
    // needs to send to.
    
    const Int colRankA = A.ColRank();
    const Int rowRankA = A.RowRank();
    Int colStrideA = A.ColStride();
    Int rowStrideA = A.RowStride();
    Int colAlignA = A.ColAlign();
    Int rowAlignA = A.RowAlign();
    SyncInfo<D1> syncGeneral = SyncInfo<D1>();


    const bool inAGrid = A.Participating();
    

    Int recvMetaData[6];

    Int metaData[6];
    if(inAGrid)
    {
        
        metaData[0] = m;
        metaData[1] = n;
        metaData[2] = colStrideA;
        metaData[3] = rowStrideA;
        metaData[4] = colAlignA;
        metaData[5] = rowAlignA;

        
    }
    else
    {
        metaData[0] = 0;
        metaData[1] = 0;
        metaData[2] = 0;
        metaData[3] = 0;
        metaData[4] = 0;
        metaData[5] = 0;
    }
    
    const std::vector<Int> sendMetaData (metaData,metaData + 6 );


    //const Int sendMetaData[6] = inAGrid ? {m,n,colStrideA,rowStrideA,colAlignA,rowAlignA} : {0,0,0,0,0,0};
    // sendMetaData[0] = m;
    // sendMetaData[1] = n;
    // sendMetaData[2] = colStrideA;
    // sendMetaData[3] = rowStrideA;
    // sendMetaData[4] = colAlignA;
    // sendMetaData[5] = rowAlignA;

    
    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    Synchronize(syncGeneral);
    

    mpi::AllReduce( sendMetaData.data(), recvMetaData, 6, mpi::MAX, viewingCommB,syncGeneral);
    Synchronize(syncGeneral);

    m = recvMetaData[0];
    n = recvMetaData[1];
    colStrideA = recvMetaData[2];
    rowStrideA = recvMetaData[3];
    colAlignA = recvMetaData[4];
    rowAlignA = recvMetaData[5];

    
    //std::printf("Recv Meta Data m:%d n:%d colStrideA: %d rowStrideA: %d colAlignA: %d rowAlignA: %d\n",m,n,colStrideA,rowStrideA,colAlignA,rowAlignA);

    B.Resize(m, n);
    const Int colStrideB = B.ColStride();
    const Int rowStrideB = B.RowStride();
    const Int colShiftB = B.ColShift();
    const Int rowShiftB = B.RowShift();
    const Int colRankB = B.ColRank();
    const Int rowRankB = B.RowRank();
    const Int colAlignB = B.ColAlign();
    const Int rowAlignB = B.RowAlign();
    const bool inBGrid = B.Participating();


    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B.LockedMatrix());



    //const Int colGCD = GCD(colStrideB, colStrideA);
    const Int rowGCD = GCD(rowStrideB, rowStrideA);
    //const Int colLCM = colStrideB*colStrideA / colGCD;
    const Int rowLCM = rowStrideB*rowStrideA / rowGCD;
    //const Int numColSends = colStrideB / colGCD;
    const Int numRowSends = rowLCM / rowStrideA ;
    const Int numRowRecvs = rowLCM / rowStrideB;

    const Int myRankViewing = mpi::Rank(viewingCommB);

    



    const Int rankBRecv = Mod(B.Grid().Rank(), rowStrideA);



    //Setup for receiving data in B
    const Int sendColOffset = colAlignA;
    const Int recvColOffset =
      Mod(colAlignB,colStrideB);
    const Int sendRowOffset = rowAlignA;
    const Int recvRowOffset =
      Mod(0*rowStrideA+rowAlignB,rowStrideB);

    const Int colShift = Mod(colRankB-recvColOffset, colStrideB);
    const Int rowShift = Mod(rowRankB-recvRowOffset, rowStrideB);

    
    //std::printf("B Grid Rank: %d\n",B.Grid().Rank());

    const Int numInB = B.Grid().Rank();

    const Int firstSendRow = Mod(colShift+sendColOffset,colStrideA);
    const Int firstSendCol = Mod(rowShift+sendRowOffset,rowStrideA);

    const Int numColRecvs = Length(colStrideA,colShift,colStrideB);
    Int sendCol = firstSendCol;
    //const Int numRowRecvs = Length(rowStrideA,rowShift,rowStrideB);

    // Recv data
    // For now, simply receive sequentially. Until we switch to
    // nonblocking recv's, we won't be using much of the
    // recvBuf
    Int sendRow = firstSendRow;

    if(!inBGrid && !inAGrid)
        return;

    const Int maxSendSize =
      (n/(rowStrideA*numRowSends)+1) * (m);

    
    // Translate the ranks from A's VC communicator to B's viewing so that
    // we can match send/recv communicators. Since A's VC communicator is not
    // necessarily defined on every process, we instead work with A's owning
    // group and account for row-major ordering if necessary.
    const int sizeA = A.Grid().Size();
    vector<int> rankMap(sizeA), ranks(sizeA);
    if(A.Grid().Order() == COLUMN_MAJOR)
    {
        for(int j=0; j<sizeA; ++j)
            ranks[j] = j;
    }
    else
    {
        // The (i,j) = i + j*colStrideA rank in the column-major ordering is
        // equal to the j + i*rowStrideA rank in a row-major ordering.
        // Since we desire rankMap[i+j*colStrideA] to correspond to process
        // (i,j) in A's grid's rank in this viewing group, ranks[i+j*colStrideA]
        // should correspond to process (i,j) in A's owning group. Since the
        // owning group is ordered row-major in this case, its rank is
        // j+i*rowStrideA. Note that setting
        // ranks[j+i*rowStrideA] = i+j*colStrideA is *NOT* valid.
        for(int i=0; i<colStrideA; ++i)
            for(int j=0; j<rowStrideA; ++j)
                ranks[i+j*colStrideA] = j+i*rowStrideA;
    }
    mpi::Translate(
        owningGroupA, sizeA, ranks.data(), viewingCommB, rankMap.data());

    Int requiredMemory = 0;
    if(inAGrid)
        requiredMemory += maxSendSize;
    if(inBGrid)
        requiredMemory += maxSendSize;

    

    
    std::vector<simple_buffer<T,D1>> sendBufVector(numRowSends);
    
    for(Int i=0; i<numRowSends; ++i)
    {
        sendBufVector[i].allocate(inAGrid ? maxSendSize : 0);

    }
    


    //simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
    simple_buffer<T,D2> recv_buf(inBGrid ? maxSendSize : 0, syncInfoB);

    //T* sendBuf = send_buf.data();
    T* recvBuf = recv_buf.data();




    Int recvRow = 0;
    //std::printf("rowStrideA %d rowStrideB %d numRowSends %d LCM %d\n",rowStrideA,rowStrideB,numRowSends,rowLCM);


    //Checking if process are in both A and B grids 
    for (Int rowSend = 0; rowSend < numRowSends; rowSend++)
    {
        const Int recvVCRank = Mod(A.Grid().Rank() + rowSend*rowStrideA, rowStrideB);
        const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);

        if(recvViewingRank==myRankViewing)
        {
            Int sendWidth = Length(nLocA,rowSend,numRowSends);

            Int rowRecv = 0;

            for(rowRecv = 0; rowRecv<numRowRecvs; ++rowRecv)
            {
                const Int sendVCRank = Mod((sendRow + rankBRecv),rowStrideA);
                sendRow = Mod(sendRow+rowStrideB,rowStrideA);
                if(rankMap[sendVCRank]==myRankViewing) break;
            }

            

            const Int recvWidth = ((rowRecv*rowStrideB + numInB)>= Mod(n,rowLCM)) ? floor(n/rowLCM) : floor(n/rowLCM)+1;
            copy::util::InterleaveMatrix(
                mLocA, sendWidth,
                A.LockedBuffer(0,rowSend),
                1, numRowSends*A.LDim(),
                B.Buffer(0,rowRecv),
                1, (numRowRecvs)*B.LDim(),
                syncInfoB);
            Synchronize(syncInfoA);
            Synchronize(syncInfoB);

            //printf("APpyling same rank optimization\n");

        }

    }



    

    std::vector<mpi::Request<T>> sendRequests(numRowSends);
    std::vector<bool> sendRequestsUsed(numRowSends,false);
    for(Int rowSend=0; rowSend<numRowSends; ++rowSend)
    {
        Int recvCol = 0; // avoid compiler warnings...
        if(inAGrid)
            recvCol=Mod(Mod(rowRankA-rowAlignA,rowStrideA)+rowAlignB,rowStrideB);
        mpi::Request<T> sendRequest;

        const Int recvVCRank = Mod(A.Grid().Rank() + rowSend*rowStrideA, rowStrideB);
        const Int recvViewingRank = B.Grid().VCToViewing(recvVCRank);

        if(inAGrid && recvViewingRank!=myRankViewing)
        {
            //printf("I am sending data\n");
            //Pack Data
            Int sendWidth = Length(nLocA,rowSend,numRowSends);
            //std::printf("sendWidth from send %d\n", sendWidth);
            copy::util::InterleaveMatrix(
                    mLocA, sendWidth,
                    A.LockedBuffer(0,rowSend),
                    1, numRowSends*A.LDim(),
                    sendBufVector[rowSend].data(), 1, mLocA, syncInfoA);


            Synchronize(syncInfoA);
            sendRequestsUsed[rowSend] = true;

            
            //std::printf("Row Rank %d colRank: %d Rank: %d sending to rank:%d\n",rowRankA,colRankA,mpi::Rank(viewingCommB), recvViewingRank);
            mpi::ISend
            (sendBufVector[rowSend].data(), mLocA*sendWidth, recvViewingRank,
              viewingCommB, sendRequests[rowSend]);


        }
        //recvRow = Mod(recvRow+colStrideA,colStride);

    }


    //start receiving data from other processes
    sendRow = firstSendRow;

    

    for(Int rowRecv=0; rowRecv<numRowRecvs; ++rowRecv)
    {

        const Int sendVCRank = Mod((sendRow + rankBRecv),rowStrideA);

        if(inBGrid && rankMap[sendVCRank]!=myRankViewing)
        {
            

            //printf("I am recv data\n");
            const Int sendColShift =
              Shift(sendRow, colAlignA, colStrideA);
            //const Int sendHeight = Length(m, sendColShift, colLCM);
            const Int localColOffset =
              (sendColShift-colShiftB) / colStrideB;
            

            
            const Int sendRowShift =
              Shift(sendCol, rowAlignA, rowStrideA) +
              rowRecv*rowStrideA;
            //const Int sendWidth = Length(n, sendRowShift, rowLCM);
            const Int sendWidth = ((rowRecv*rowStrideB + numInB)>= Mod(n,rowLCM)) ? floor(n/rowLCM) : floor(n/rowLCM)+1;
            //std::printf("rowLCM %d n %d Mod %d LDIM %d  widthNonCorrect %d Condn %d\n", rowLCM,n,Mod(n,rowLCM),B.LDim(),floor(n/rowLCM),rowRecv*rowStrideB + numInB );
            
            const Int localRowOffset =
              (sendRowShift-rowShiftB) / rowStrideB;



            //const Int sendVCRank = sendRow+sendCol*colStrideA;
            
            //std::printf("sendWidth:%d mLocA: %d sendVCRank: %d rankMap[sendVCRank]: %d recvrank %d\n",sendWidth,mLocA,sendVCRank,rankMap[sendVCRank],mpi::Rank(viewingCommB));
            mpi::Recv(
                recvBuf, m*sendWidth, rankMap[sendVCRank],
                viewingCommB, syncInfoB);

            // Unpack the data
            copy::util::InterleaveMatrix(
                m, sendWidth,
                recvBuf, 1, m,
                B.Buffer(0,rowRecv),
                1, (numRowRecvs)*B.LDim(),
                syncInfoB);

            

            

        }
        // Set up the next send col
        sendCol = Mod(sendCol+rowStrideB,rowStrideA);
        sendRow = Mod(sendRow+rowStrideB,rowStrideA);

        


    }
    if(inAGrid)
        for (Int i=0;i<numRowSends;++i)
        {
            if(sendRequestsUsed[i])
            {
                mpi::Wait(sendRequests[i]);
            }
            
        }
    //std::printf("FINAL: RANK: %d\n",El::mpi::Rank(El::mpi::NewWorldComm()));

    //mpi::Barrier(viewingCommB);
    sendBufVector.clear();
    // delete recvBuf;
    // delete recv_buf;
    
    


}


template<typename T, Device D1, Device D2>
void TranslateBetweenGrids
(const DistMatrix<T,STAR,STAR,ELEMENT,D1>& A,
  DistMatrix<T,STAR,STAR,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE;
    //LogicError("TranslateBetweenGrids is no longer supported. "
               //"If you have reached this message, please open "
               //"an issue at https://github.com/llnl/elemental.");
//#ifdef EL_TRANSLATE_BETWEEN_GRIDS_REENABLE__
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);

    // Attempt to distinguish between the owning groups of A and B both being
    // subsets of the same viewing communicator, the owning group of A being
    // the same as the viewing communicator of B (A is the *parent* of B),
    // and the viewing communicator of A being the owning communicator of B
    // (B is the *parent* of A).
    //
    // TODO(poulson): Decide whether these condition can be simplified.
    mpi::Comm const& commA = A.Grid().VCComm();
    mpi::Comm const& commB = B.Grid().VCComm();
    mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
    mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
    const int commSizeA = mpi::Size(commA);
    const int commSizeB = mpi::Size(commB);
    const int viewingCommSizeA = mpi::Size(viewingCommA);
    const int viewingCommSizeB = mpi::Size(viewingCommB);
    bool usingViewingA=false, usingViewingB=false;
    //mpi::Comm activeCommA, activeCommB;

    mpi::Comm const& activeCommA = (viewingCommSizeA == viewingCommSizeB) ?
    						A.Grid().ViewingComm() :
    							viewingCommSizeA == commSizeB ?
    							A.Grid().ViewingComm():
    							
    								commSizeA == viewingCommSizeB ?
    								A.Grid().VCComm() :
    								A.Grid().VCComm()
    							

    						;

    mpi::Comm const& activeCommB = (viewingCommSizeA == viewingCommSizeB) ?
    						B.Grid().ViewingComm():
    							viewingCommSizeA == commSizeB ?
    							B.Grid().VCComm():
    							
    								commSizeA == viewingCommSizeB ?
    								B.Grid().ViewingComm() :
    								B.Grid().VCComm()
    							

    						;

    usingViewingA = (viewingCommSizeA == viewingCommSizeB) ?
    						true :
    							viewingCommSizeA == commSizeB ?
    							true :
    							
    								commSizeA == viewingCommSizeB ?
    								false :
    								false
    							

    						;

    usingViewingB = (viewingCommSizeA == viewingCommSizeB) ?
    						true :
    							viewingCommSizeA == commSizeB ?
    							false :
    							
    								commSizeA == viewingCommSizeB ?
    								true :
    								false
    							

    						;


    if(!mpi::Congruent(activeCommA, activeCommB))
            LogicError("communicators were not congruent");


    
    const Int rankA = A.RedundantRank();
    const Int rankB = B.RedundantRank();

    simple_buffer<T,D1> sendBuffer(rankA == 0 ? height*width : 0);
    simple_buffer<T,D2> bcastBuffer(B.Participating() ? height*width : 0);

    SyncInfo<D1> syncInfoA = SyncInfoFromMatrix(A.LockedMatrix());
    SyncInfo<D2> syncInfoB = SyncInfoFromMatrix(B.LockedMatrix());

    // Send from the root of A to the root of B's matrix's grid
    mpi::Request<T> sendRequest;
    if(rankA == 0)
    {
        if (sendBuffer.size() != size_t(height*width))
            RuntimeError("TranslateBetweenGrids: Bad sendBuffer size!");

        util::InterleaveMatrix(
            height, width,
            A.LockedBuffer(), 1, A.LDim(),
            sendBuffer.data(), 1, height, syncInfoA);
        // TODO(poulson): Use mpi::Translate instead?
        const Int recvRank = (usingViewingB ? B.Grid().VCToViewing(0) : 0);
        mpi::ISend(
            sendBuffer.data(), height*width, recvRank, activeCommB, sendRequest);
    }

    // Receive on the root of B's matrix's grid and then broadcast
    // over the owning communicator
    if(B.Participating())
    {
        if (bcastBuffer.size() != size_t(height*width))
            RuntimeError("TranslateBetweenGrids: Bad bcastBuffer size!");
        if(rankB == 0)
        {
            // TODO(poulson): Use mpi::Translate instead?
            const Int sendRank =
              (usingViewingA ? A.Grid().VCToViewing(0) : 0);
            mpi::Recv(bcastBuffer.data(), height*width, sendRank, activeCommB,
                      syncInfoB);
        }

        mpi::Broadcast(bcastBuffer.data(), height*width, 0, B.RedundantComm(),
                       syncInfoB);

        util::InterleaveMatrix(
            height, width,
            bcastBuffer.data(), 1, height,
            B.Buffer(),  1, B.LDim(), syncInfoB);
    }

    if(rankA == 0)
        mpi::Wait(sendRequest);
//#endif // EL_TRANSLATE_BETWEEN_GRIDS_REENABLE__
}

} // namespace copy
} // namespace El


//template void TranslateBetweenGridsBroadcast <double, Device::GPU,Device::GPU> (DistMatrix<double,STAR,VC,ELEMENT,Device::GPU> const& , std::vector<DistMatrix<double,STAR,VC,ELEMENT,Device::GPU>>& );
#endif // ifndef EL_BLAS_COPY_TRANSLATEBETWEENGRIDS_HPP




// template TranslateBetweenGridsBroadcast<double, Device::CPU,Device::CPU>;


