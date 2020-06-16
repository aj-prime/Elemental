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

    std::printf("MPI_RANK:%d colStrideB:%d rowStrideB:%d colShiftB:%d rowShiftB:%d colRankB:%d rowRankB:%d colAlignB:%d rowAlignB:%d colStrideA:%d rowStrideA:%d  colRankA:%d rowRankA:%d colAlignA:%d rowAlignA:%d\n"
        ,mpi::Rank(viewingCommB),colStride,rowStride,colShiftB,rowShiftB,colRank,rowRank, colAlignB,rowAlignB,    colStrideA,rowStrideA,colRankA,rowRankA,colAlignA,rowAlignA);
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

    std::printf("numColSends: %d numRowSends: %d\n", numColSends,numRowSends);
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
                std::printf("sendHeight: %d sendWidth: %d\n", sendHeight,sendWidth);
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

    std::vector<T*> sendBufVector(numRowSends);
    std::vector<T*> recvBufVector(numRowRecvs);

    for(Int i=0; i<numRowSends; ++i)
    {
        simple_buffer<T,D1> send_buf(inAGrid ? maxSendSize : 0, syncInfoA);
        sendBufVector[i] = send_buf.data();

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

            //Pack Data
            Int sendWidth = Length(nLocA,rowSend,numRowSends);
            //std::printf("sendWidth from send %d\n", sendWidth);
            copy::util::InterleaveMatrix(
                    mLocA, sendWidth,
                    A.LockedBuffer(0,rowSend),
                    1, numRowSends*A.LDim(),
                    sendBufVector[rowSend], 1, mLocA, syncInfoA);


            Synchronize(syncInfoA);
            sendRequestsUsed[rowSend] = true;

            
            //std::printf("Row Rank %d colRank: %d Rank: %d sending to rank:%d\n",rowRankA,colRankA,mpi::Rank(viewingCommB), recvViewingRank);
            mpi::ISend
            (sendBufVector[rowSend], mLocA*sendWidth, recvViewingRank,
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

            // Set up the next send col
            sendCol = Mod(sendCol+rowStrideB,rowStrideA);
            sendRow = Mod(sendRow+rowStrideB,rowStrideA);

            

        }

        


    }
    if(inAGrid)
        for (Int i=0;i<numRowSends;++i)
        {
            if(sendRequestsUsed[i])
            {
                mpi::Wait(sendRequests[i]);
            }
            
        }
    
    


}




template<typename T, Device D1, Device D2>
void TranslateBetweenGrids
(const DistMatrix<T,STAR,STAR,ELEMENT,D1>& A,
  DistMatrix<T,STAR,STAR,ELEMENT,D2>& B)
{
#ifdef EL_TRANSLATE_BETWEEN_GRIDS_REENABLE__
    
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
    mpi::Comm activeCommA, activeCommB;
    if(viewingCommSizeA == viewingCommSizeB)
    {
        usingViewingA = true;
        usingViewingB = true;
        activeCommA = viewingCommA;
        activeCommB = viewingCommB;
        if(!mpi::Congruent(viewingCommA, viewingCommB))
            LogicError("Viewing communicators were not congruent");
    }
    else if(viewingCommSizeA == commSizeB)
    {
        usingViewingA = true;
        usingViewingB = false;
        activeCommA = viewingCommA;
        activeCommB = commB;
        if(!mpi::Congruent(viewingCommA, commB))
            LogicError("Communicators were not congruent");
    }
    else if(commSizeA == viewingCommSizeB)
    {
        usingViewingA = false;
        usingViewingB = true;
        activeCommA = commA;
        activeCommB = viewingCommB;
        if(!mpi::Congruent(commA, viewingCommB))
            LogicError("Communicators were not congruent");
    }
    else
    {
        usingViewingA = false;
        usingViewingB = false;
        activeCommA = commA;
        activeCommB = commB;
        LogicError("Unsupported TranslateBetweenGrids instance");
    }

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
#endif // EL_TRANSLATE_BETWEEN_GRIDS_REENABLE__

}

} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_TRANSLATEBETWEENGRIDS_HPP
