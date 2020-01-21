#include "cudaLight.h"
#include <algorithm>
#include <curand_kernel.h>

namespace RT
{
    namespace CUDA
    {
        __global__
        void SpawnSecondaryRaysKernel( const LightData* pLight, const RayData* pPrimaryRay, float_t intersectionDistance, RayData* pRays )
        {
            // Get global invocation index
            const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
            
            curandState randState;
            curand_init( 0, threadId, 0, &randState );

            RayData secondaryRay;

            // Ray from intersection to light
            intersectionDistance = intersectionDistance - pLight->ShadowBias;

            secondaryRay.Origin = pPrimaryRay->Direction * intersectionDistance + pPrimaryRay->Origin;

            // Simulate light size for soft shadows
            vec4 noise = vec4(
                curand_uniform( &randState ),
                curand_uniform( &randState ),
                curand_uniform( &randState ), 0 );

            secondaryRay.Direction = pLight->Position - secondaryRay.Origin + noise;
            secondaryRay.Direction.Normalize3();

            pRays[threadId] = secondaryRay;
        }

        Light::Light( const Array<LightData>& array, int index )
            : DataWrapper( array, index )
        {
        }

        Array<Light::RayType::DataType> Light::SpawnSecondaryRays( const RayType& primaryRay, float_t intersectionDistance ) const
        {
            Array<RayType::DataType> rays( Memory.Host().Subdivs );

            DispatchParameters dispatchParams( Memory.Host().Subdivs );

            // Execute spawning kernel
            SpawnSecondaryRaysKernel<<<dispatchParams.NumBlocksPerGrid, dispatchParams.NumThreadsPerBlock>>>
                (Memory.Device(), primaryRay.Memory.Device(), intersectionDistance, rays.Device());

            return rays;
        }
    }
}
