#include "cudaCamera.h"

NAMESPACE_RT_CUDA
{
    __global__
    void SpawnPrimaryRays( const CameraData * pCamera, int hCount, int vCount, RayData * pRays )
    {

    }

    Array<Camera::RayType::DataType> Camera::SpawnPrimaryRays( int hCount, int vCount )
    {
        Array<RayType::DataType> rays( hCount * vCount );

        DispatchParameters dispatchParams( hCount * vCount );

        // Execute spawning kernel
        SpawnPrimaryRays DISPATCH( dispatchParams )
            (DeviceMemory.Data(), hCount, vCount, rays.Data());

        return rays;
    }
}}
