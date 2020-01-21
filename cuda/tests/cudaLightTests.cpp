#include <gtest/gtest.h>
#include "../cudaLight.h"

// Reference implementation
#include "../../omp/ompLight.h"

TEST( cudaLightTests, SpawnSecondaryRays_Test )
{
    RT::OMP::Ray ompPrimaryRay;
    ompPrimaryRay.Origin = RT::vec4( 0, 0, 0, 1 );
    ompPrimaryRay.Direction = RT::vec4( 1, 1, 1, 0 );

    RT::OMP::Light ompLight;
    ompLight.Position = RT::vec4( 1, 1, 1, 1 );
    ompLight.Radius = 10;
    ompLight.ShadowBias = 0;
    ompLight.Subdivs = 30;

    const auto expectedSecondaryRays = ompLight
        .SpawnSecondaryRays( ompPrimaryRay, ompLight.Position.Length3() );

    RT::CUDA::Array<RT::CUDA::LightData> cudaLightDeviceMemory( 1 );
    RT::CUDA::Light cudaLight( cudaLightDeviceMemory, 0 );
    cudaLight.Memory.Host().Position = ompLight.Position;
    cudaLight.Memory.Host().Radius = ompLight.Radius;
    cudaLight.Memory.Host().ShadowBias = ompLight.ShadowBias;
    cudaLight.Memory.Host().Subdivs = ompLight.Subdivs;
    cudaLight.Memory.Update();

    RT::CUDA::Array<RT::CUDA::RayData> cudaRayDeviceMemory( 1 );
    RT::CUDA::Ray cudaPrimaryRay( cudaRayDeviceMemory, 0 );
    cudaPrimaryRay.Memory.Host().Origin = ompPrimaryRay.Origin;
    cudaPrimaryRay.Memory.Host().Direction = ompPrimaryRay.Direction;
    cudaPrimaryRay.Memory.Update();

    auto actualSecondaryRaysDeviceMemory = cudaLight
        .SpawnSecondaryRays( cudaPrimaryRay, ompLight.Position.Length3() );

    std::vector<RT::CUDA::RayData> actualSecondaryRays( actualSecondaryRaysDeviceMemory.Size() );
    actualSecondaryRaysDeviceMemory.Sync();
    actualSecondaryRaysDeviceMemory.HostCopyTo( actualSecondaryRays.data() );

    ASSERT_EQ( expectedSecondaryRays.size(), actualSecondaryRays.size() );

    for( size_t i = 0; i < expectedSecondaryRays.size(); ++i )
    {
        EXPECT_NEAR( expectedSecondaryRays[i].Origin.x, actualSecondaryRays[i].Origin.x, 0.001f ) << "Ray #" << i;
        EXPECT_NEAR( expectedSecondaryRays[i].Origin.y, actualSecondaryRays[i].Origin.y, 0.001f ) << "Ray #" << i;
        EXPECT_NEAR( expectedSecondaryRays[i].Origin.z, actualSecondaryRays[i].Origin.z, 0.001f ) << "Ray #" << i;

        //EXPECT_NEAR( expectedSecondaryRays[i].Direction.x, actualSecondaryRays[i].Direction.x, 0.001f ) << "Ray #" << i;
        //EXPECT_NEAR( expectedSecondaryRays[i].Direction.y, actualSecondaryRays[i].Direction.y, 0.001f ) << "Ray #" << i;
        //EXPECT_NEAR( expectedSecondaryRays[i].Direction.z, actualSecondaryRays[i].Direction.z, 0.001f ) << "Ray #" << i;
    }
}
