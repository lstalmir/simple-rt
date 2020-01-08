#include <gtest/gtest.h>
#include "../cudaCamera.h"

// Reference implementation
#include "../../omp/ompCamera.h"

TEST( cudaCameraTests, SpawnPrimaryRays_Test )
{
    constexpr int horizontalRays = 3;
    constexpr int verticalRays = 3;

    RT::OMP::Camera ompCamera;
    ompCamera.Direction = RT::vec4( 1, 0, 0 );
    ompCamera.Origin = RT::vec4( 0, 0, 0 );
    ompCamera.Up = RT::vec4( 0, 1, 0 );
    ompCamera.AspectRatio = 1;
    ompCamera.HorizontalFOV = RT::Radians( 75 );

    auto expectedPrimaryRays = ompCamera.SpawnPrimaryRays( horizontalRays, verticalRays );

    RT::CUDA::Camera cudaCamera;
    cudaCamera.HostMemory.Direction = RT::vec4( 1, 0, 0 );
    cudaCamera.HostMemory.Origin = RT::vec4( 0, 0, 0 );
    cudaCamera.HostMemory.Up = RT::vec4( 0, 1, 0 );
    cudaCamera.HostMemory.AspectRatio = 1;
    cudaCamera.HostMemory.HorizontalFOV = RT::Radians( 75 );

    RT::CUDA::Array<RT::CUDA::CameraData> cudaCameraDeviceMemory( 1 );
    cudaCameraDeviceMemory.UpdateDeviceMemory( &cudaCamera.HostMemory, 1 );
    cudaCamera.DeviceMemory = RT::CUDA::ArrayView<RT::CUDA::CameraData>( cudaCameraDeviceMemory, 0 );

    auto actualPrimaryRaysDeviceMemory = cudaCamera.SpawnPrimaryRays( horizontalRays, verticalRays );

    cudaDeviceSynchronize();

    ASSERT_EQ( expectedPrimaryRays.size(), actualPrimaryRaysDeviceMemory.Size() );

    std::vector<RT::CUDA::RayData> actualPrimaryRays( actualPrimaryRaysDeviceMemory.Size() );
    actualPrimaryRaysDeviceMemory.GetDeviceMemory( actualPrimaryRays.data() );

    for( size_t i = 0; i < expectedPrimaryRays.size(); ++i )
    {
        EXPECT_NEAR( expectedPrimaryRays[i].Origin.x, actualPrimaryRays[i].Origin.x, 0.001f );
        EXPECT_NEAR( expectedPrimaryRays[i].Origin.y, actualPrimaryRays[i].Origin.y, 0.001f );
        EXPECT_NEAR( expectedPrimaryRays[i].Origin.z, actualPrimaryRays[i].Origin.z, 0.001f );

        EXPECT_NEAR( expectedPrimaryRays[i].Direction.x, actualPrimaryRays[i].Direction.x, 0.001f );
        EXPECT_NEAR( expectedPrimaryRays[i].Direction.y, actualPrimaryRays[i].Direction.y, 0.001f );
        EXPECT_NEAR( expectedPrimaryRays[i].Direction.z, actualPrimaryRays[i].Direction.z, 0.001f );
    }
}
