/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

/*
  Testing interaction of 2 matrices with different distributions.
  Not sure why it's call "DifferentGrids".  Maybe the partitioning
  is based on a grid.
*/
#include <El.hpp>
#include <chrono>
using namespace El;

int
main( int argc, char* argv[] )
{
    Environment env( argc, argv );
    mpi::Comm comm = mpi::NewWorldComm();
    mpi::Comm comm_sqrt, comm_sqrt_sec;


    const Int commSize = mpi::Size( comm );
    const Int rank = mpi::Rank(comm);
    printf("Rank is %d\n", rank);

    
   
    

    try
    {
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        const bool colMajorSqrt = Input("--colMajorSqrt","colMajor sqrt?",true);
        const Int m = Input("--height","height of matrix",50);
        const Int n = Input("--width","width of matrix",100);
        const bool print = Input("--print","print matrices?",false);
        const Int iters = Input("--iters","Iterations (default:100)?",100);
        const Int grid1_width = Input("--g1Width","width of grid 1?",2);
        const Int grid2_width = Input("--g2Width","width of grid 2?",2);
        const Int grid1_height = Input("--g1Height","height of grid 1?",2);
        const Int grid2_height = Input("--g2Height","height of grid 2?",2);

        const bool grid2_order = Input("--g2order","Start Grid2 from Process 0?",false);
        ProcessInput();
        PrintInputReport();

        const GridOrder order = ( colMajor ? COLUMN_MAJOR : ROW_MAJOR );
        const GridOrder orderGrid = ( colMajorSqrt ? COLUMN_MAJOR : ROW_MAJOR );

        // size of grids.
        const Int grid1Size = grid1_height * grid1_width;
        const Int grid2Size = grid2_height * grid2_width;

        
        std::vector<int> grid1Ranks(grid1Size);
        std::vector<int> grid2Ranks(grid2Size);

        //Ranks in Grid1
        for( Int i=0; i<grid1Size; ++i )
            grid1Ranks[i] = i;

        //Ranks in Grid2
        Int counter=0;

        if(grid2_order)
        {
            for( Int i=0; i<grid2Size; ++i )
            {
                grid2Ranks[counter] = i;
                counter+=1;
            }

        }
        else
        {
            for( Int i=commSize-1; i>=commSize - 1 - grid2Size; --i )
            {
                grid2Ranks[counter] = i;
                counter+=1;
            }
        }

        mpi::Group group, grid1Group,grid2Group;

        std::printf("Grid1size: %d Grid2size %d\n",grid1Size, grid2Size);



        mpi::CommGroup( comm, group );
        mpi::Incl( group, grid1Ranks.size(), grid1Ranks.data(), grid1Group );
        mpi::Incl( group, grid2Ranks.size(), grid2Ranks.data(), grid2Group );


        const Grid grid( std::move(comm), order );

        const Grid grid1(
            mpi::NewWorldComm(), grid1Group, grid1_height, orderGrid );
        const Grid grid2(
            mpi::NewWorldComm(), grid2Group, grid2_height, orderGrid );

 

        // A is distibuted on Grid1, ASqrt is distributed Grid2.
        

        DistMatrix<double,STAR,VC,ELEMENT,Device::CPU> A(grid1), ASqrt(grid2);
        
        
        //DistMatrix<double> A(grid), ASqrt(sqrtGrid_sec);
        
        Identity( A, m, n );
        
        if( A.Participating() && print)
        {
            printf("Height: %d and Width: %d \n",A.LocalHeight(), A.LocalWidth());
            //Print( A, "A" );
        }
        

            
        auto duration_all =0;
        for(Int i=0 ;i< iters; ++i){
            //A*=2;
            mpi::Barrier();
            auto start = std::chrono::high_resolution_clock::now();
            ASqrt = A;
            //mpi::Barrier();
            auto end = std::chrono::high_resolution_clock::now();
            

            auto duration = duration_cast<std::chrono::microseconds>(end - start); 
            //std::cout<<"here:"<<duration.count()<<"\n";
            duration_all = duration_all+duration.count();

        }

        std::cout << "Rank:"<<rank<< " Total Time taken (A->B):" << duration_all/iters << endl; 
        
        if( ASqrt.Participating() )
        {
            if( print )
                Print( ASqrt, "ASqrt := A" );
            ASqrt *= 2;
            if( print )
                Print( ASqrt, "ASqrt := 2 ASqrt" );
        }
        //A = ASqrt;

        if( print && A.Participating() )
            Print( A, "A := ASqrt" );

        //const Grid newGrid( mpi::NewWorldComm(), order );
        //A.SetGrid( newGrid );
        //if( print )
            //Print( A, "A after changing grid" );

        
    }
    catch( std::exception& e ) { ReportException(e); }

    return 0;
}
