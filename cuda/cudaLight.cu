#include "cudaLight.h"
#include <algorithm>

namespace RT
{
    namespace CUDA
    {
        __global__
        void SpawnSecondaryRaysKernel( const LightData* pLight, const RayData* pPrimaryRay, float_t intersectionDistance, RayData* pRays )
        {

        }

        Array<Light::RayType::DataType> Light::SpawnSecondaryRays( const RayType& primaryRay, float_t intersectionDistance ) const
        {
            Array<RayType::DataType> rays( HostMemory.Subdivs );

            DispatchParameters dispatchParams( HostMemory.Subdivs );

            // Execute spawning kernel
            SpawnSecondaryRaysKernel<<<dispatchParams.NumBlocksPerGrid, dispatchParams.NumThreadsPerBlock>>>
                (DeviceMemory.Data(), primaryRay.DeviceMemory.Data(), intersectionDistance, rays.Data());

            return rays;
        }
    }
}
