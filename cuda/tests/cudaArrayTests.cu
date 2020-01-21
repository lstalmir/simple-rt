#include <gtest/gtest.h>
#include "../cudaCommon.h"
#include "../cudaMemory.h"

__global__
void AddElements( const int* A, const int* B, int* C )
{
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    C[ threadId ] = A[ threadId ] + B[ threadId ];
}

/***************************************************************************************\

Test:
    AddElements

Description:
    Add elements of two arrays.

\***************************************************************************************/
TEST( cudaArrayTests, AddElements )
{
    RT::CUDA::Array<int> A( 5 );
    RT::CUDA::Array<int> B( 5 );
    RT::CUDA::Array<int> C( 5 );

    const int A_data[5] = { 0, 1, 2, 3, 4 };
    const int B_data[5] = { 101, 202, 303, 404, 505 };
    const int C_expected[5] = { 101, 203, 305, 407, 509 };
    int C_actual[5];

    std::memcpy( A.Host(), A_data, sizeof( A_data ) );
    std::memcpy( B.Host(), B_data, sizeof( B_data ) );

    A.Update();
    B.Update();

    AddElements<<<1,5>>>( A.Device(), B.Device(), C.Device() );

    C.Sync();
    C.HostCopyTo( C_actual );

    EXPECT_EQ( C_expected[0], C_actual[0] );
    EXPECT_EQ( C_expected[1], C_actual[1] );
    EXPECT_EQ( C_expected[2], C_actual[2] );
    EXPECT_EQ( C_expected[3], C_actual[3] );
    EXPECT_EQ( C_expected[4], C_actual[4] );
}
