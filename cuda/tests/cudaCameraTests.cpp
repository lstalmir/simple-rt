#include <gtest/gtest.h>
#include "../cudaCamera.h"

// Reference implementation
#include "../../omp/ompCamera.h"

namespace
{
    template<int horizontalRays, int verticalRays>
    static auto SpawnPrimaryRays_TestTemplate()
    {
        RT::OMP::Camera ompCamera;
        ompCamera.Direction = RT::vec4( 1, 0, 0 );
        ompCamera.Origin = RT::vec4( 0, 0, 0 );
        ompCamera.Up = RT::vec4( 0, 1, 0 );
        ompCamera.AspectRatio = 1;
        ompCamera.HorizontalFOV = RT::Radians( 75 );

        const auto expectedPrimaryRays = ompCamera
            .SpawnPrimaryRays( horizontalRays, verticalRays );

        RT::CUDA::Array<RT::CUDA::CameraData> cudaCameraDeviceMemory( 1 );
        RT::CUDA::Camera cudaCamera( cudaCameraDeviceMemory, 0 );
        cudaCamera.Memory.Host().Direction = RT::vec4( 1, 0, 0 );
        cudaCamera.Memory.Host().Origin = RT::vec4( 0, 0, 0 );
        cudaCamera.Memory.Host().Up = RT::vec4( 0, 1, 0 );
        cudaCamera.Memory.Host().AspectRatio = 1;
        cudaCamera.Memory.Host().HorizontalFOV = RT::Radians( 75 );
        cudaCamera.Memory.Update();

        auto actualPrimaryRaysDeviceMemory = cudaCamera
            .SpawnPrimaryRays( horizontalRays, verticalRays );

        std::vector<RT::CUDA::RayData> actualPrimaryRays( actualPrimaryRaysDeviceMemory.Size() );
        actualPrimaryRaysDeviceMemory.Sync();
        actualPrimaryRaysDeviceMemory.HostCopyTo( actualPrimaryRays.data() );

        ASSERT_EQ( expectedPrimaryRays.size(), actualPrimaryRays.size() );

        for( size_t i = 0; i < expectedPrimaryRays.size(); ++i )
        {
            EXPECT_NEAR( expectedPrimaryRays[i].Origin.x, actualPrimaryRays[i].Origin.x, 0.001f ) << "Ray #" << i;
            EXPECT_NEAR( expectedPrimaryRays[i].Origin.y, actualPrimaryRays[i].Origin.y, 0.001f ) << "Ray #" << i;
            EXPECT_NEAR( expectedPrimaryRays[i].Origin.z, actualPrimaryRays[i].Origin.z, 0.001f ) << "Ray #" << i;

            EXPECT_NEAR( expectedPrimaryRays[i].Direction.x, actualPrimaryRays[i].Direction.x, 0.001f ) << "Ray #" << i;
            EXPECT_NEAR( expectedPrimaryRays[i].Direction.y, actualPrimaryRays[i].Direction.y, 0.001f ) << "Ray #" << i;
            EXPECT_NEAR( expectedPrimaryRays[i].Direction.z, actualPrimaryRays[i].Direction.z, 0.001f ) << "Ray #" << i;
        }
    }
}

/***************************************************************************************\

Test:
    SpawnPrimaryRays

Description:
    Generate primary rays for given camera and pixel grid size.
    Compare generated set with OMP implementation result.

\***************************************************************************************/
TEST( cudaCameraTests, SpawnPrimaryRays_3x3_Test )
{
    return SpawnPrimaryRays_TestTemplate<3, 3>();
}

TEST( cudaCameraTests, SpawnPrimaryRays_6x6_Test )
{
    return SpawnPrimaryRays_TestTemplate<6, 6>();
}

TEST( cudaCameraTests, SpawnPrimaryRays_16x16_Test )
{
    return SpawnPrimaryRays_TestTemplate<16, 16>();
}

TEST( cudaCameraTests, SpawnPrimaryRays_1280x720_Test )
{
    return SpawnPrimaryRays_TestTemplate<1280, 720>();
}
