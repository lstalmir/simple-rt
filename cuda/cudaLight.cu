#include "cudaLight.h"
#include <algorithm>

NAMESPACE_RT_CUDA
{
    __global__
    void SpawnSecondaryRays( const LightData * pLight, float_t intersectionDistance, RayData* pRays )
    {

    }

    Array<Light::RayType::DataType> Light::SpawnSecondaryRays( const RayType & primaryRay, float_t intersectionDistance ) const
    {
        Array<RayType::DataType> rays( HostMemory.Subdivs );

        DispatchParameters dispatchParams( HostMemory.Subdivs );

        // Execute spawning kernel
        SpawnSecondaryRays DISPATCH( dispatchParams )
            (DeviceMemory.Data(), intersectionDistance, rays.Data());

        return rays;
    }
}}
