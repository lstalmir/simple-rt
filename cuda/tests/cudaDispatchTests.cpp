#include <gtest/gtest.h>
#include "../cudaCommon.h"

TEST( cudaDispatchTests, InvocationCount )
{
    const unsigned int numInvocations = 12345;

    RT::CUDA::DispatchParameters dispatchParams( numInvocations );
    dispatchParams.MaxThreadsPerBlock( 128 );

    EXPECT_EQ( dispatchParams.NumThreadsPerBlock, 128 );
    EXPECT_GE( dispatchParams.NumBlocksPerGrid * dispatchParams.NumThreadsPerBlock, numInvocations );
}
