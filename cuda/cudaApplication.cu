#include "cudaApplication.h"
#include <curand_kernel.h>

namespace RT
{
    namespace CUDA
    {
        __global__
        void SpawnShadowRaysKernel(
            const SecondaryRayData* pRays,
            const unsigned int numRays,
            const LightData* pLights,
            const unsigned int numLights,
            RayData* pShadowRays )
        {
            // Get global invocation index
            const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

            curandState randState;
            curand_init( 0, threadId, 0, &randState );

            const int rayIdx = threadId / numLights;
            const int lightIdx = threadId % numLights;
            const int firstShadowRayIdx = threadId * RT_LIGHT_SUBDIVS;

            if( rayIdx >= numRays )
            {
                return;
            }

            const SecondaryRayData& ray = pRays[rayIdx];
            const LightData light = pLights[lightIdx];

            if( ray.Intersection.Distance.x == INFINITY )
            {
                // no intersection
                return;
            }

            RayData shadowRay;

            // Ray from intersection to light
            float intersectionDistance = ray.Intersection.Distance.x - light.ShadowBias;

            shadowRay.Origin = ray.Ray.Direction * intersectionDistance + ray.Ray.Origin;

            for( int i = 0; i < RT_LIGHT_SUBDIVS; ++i )
            {
                // Simulate light size for soft shadows
                vec4 noise = vec4(
                    curand_uniform( &randState ),
                    curand_uniform( &randState ),
                    curand_uniform( &randState ), 0 );

                shadowRay.Direction = light.Position - shadowRay.Origin + noise;
                shadowRay.Direction.Normalize3();

                pShadowRays[firstShadowRayIdx + i] = shadowRay;
            }
        }
        
        __device__
        bool IntersectRayBox( RayData ray, Box box )
        {
            float tmin = (box.Min.x - ray.Origin.x) / ray.Direction.x;
            float tmax = (box.Max.x - ray.Origin.x) / ray.Direction.x;

            if( tmin > tmax )
            {   // swap
                float t = tmin;
                tmin = tmax;
                tmax = t;
            }

            float tymin = (box.Min.y - ray.Origin.y) / ray.Direction.y;
            float tymax = (box.Max.y - ray.Origin.y) / ray.Direction.y;

            if( tymin > tymax )
            {   // swap
                float t = tymin;
                tymin = tymax;
                tymax = t;
            }

            if( (tmin > tymax) || (tymin > tmax) )
                return false;

            if( tymin > tmin )
                tmin = tymin;

            if( tymax < tmax )
                tmax = tymax;

            float tzmin = (box.Min.z - ray.Origin.z) / ray.Direction.z;
            float tzmax = (box.Max.z - ray.Origin.z) / ray.Direction.z;

            if( tzmin > tzmax )
            {   // swap
                float t = tzmin;
                tzmin = tzmax;
                tzmax = t;
            }

            if( (tmin > tzmax) || (tzmin > tmax) )
                return false;

            if( tzmin > tmin )
                tmin = tzmin;

            if( tzmax < tmax )
                tmax = tzmax;

            return true;
        }

        __device__
        vec4 IntersectRayTriangle( RayData ray, Triangle triangle )
        {
            // Compute triangle edges relative to (0,0,0)
            const auto E1 = triangle.B - triangle.A;
            const auto E2 = triangle.C - triangle.A;

            // Compute factor denominator
            const auto P = ray.Direction.Cross( E2 );
            const float denominator = P.Dot( E1 );

            // First condition of intersection:
            if( denominator < -1e-6f || denominator > 1e-6f )
            {
                // Calculate distance from V0 to ray origin
                const auto T = ray.Origin - triangle.A;

                // Calculate u parameter and test bounds
                const auto U = P.Dot( T ) / denominator;

                // Second condition of intersection:
                if( U >= 0 && U <= 1 )
                {
                    const auto Q = T.Cross( E1 );

                    // Calculate v parameter and test bounds
                    const auto V = Q.Dot( ray.Direction ) / denominator;

                    // Third and fourth condition of intersection:
                    if( V >= 0 && (U + V) <= 1 )
                    {
                        // Calculate t, if t > 0, the ray intersects the triangle
                        const auto distance = Q.Dot( E2 ) / denominator;

                        if( distance > 0 )
                        {
                            return vec4( distance, U, V, V );
                        }
                    }
                }
            }

            // No intersection
            return vec4( INFINITY );
        }

        __global__
        void ObjectIntersectionKernel(
            SecondaryRayData* pRays,
            const unsigned int numRays,
            const ObjectData* pObject,
            const Triangle* pTriangles )
        {
            // Get global invocation index
            const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

            if( threadId >= numRays )
            {
                return;
            }

            auto& ray = pRays[threadId];
            ObjectData object = *pObject;

            // Ray-Object intersection, test bounding box and spawn per-triangle intersections
            if( !RT_ENABLE_BOUNDING_BOXES || IntersectRayBox( ray.Ray, object.BoundingBox ) )
            {
                for( int t = 0; t < object.NumTriangles; ++t )
                {
                    Triangle triangle = pTriangles[object.FirstTriangle + t];

                    #if RT_ENABLE_BACKFACE_CULL
                    if( ray.Ray.Direction.Dot( triangle.Normal ) >= 0 )
                    {
                        // Backface culling
                        continue;
                    }
                    #endif

                    // Ray-Triangle intersection
                    RT::vec4 intersection = IntersectRayTriangle( ray.Ray, triangle );

                    if( intersection.x < ray.Intersection.Distance.x )
                    {
                        // Update intersection
                        ray.Intersection.Triangle = triangle;
                        ray.Intersection.Distance = intersection;
                        ray.Intersection.Color = object.Color;
                        ray.Intersection.Ior = object.Ior;
                    }
                }
            }
        }

        __global__
        void CountObjectIntersections(
            const SecondaryRayData* pRays,
            const unsigned int numRays,
            int* pNumIntersections )
        {
            // Get global invocation index
            const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

            if( threadId >= numRays )
            {
                return;
            }

            if( pRays[threadId].Intersection.Distance.x < INFINITY )
            {
                atomicAdd( pNumIntersections, 1 );
            }
        }
        
        __device__
        int IntersectShadowRaysKernel(
            const RayData* pRays,
            const ObjectData& object,
            const Triangle* pTriangles )
        {
            int numIntersections = 0;

            for( int r = 0; r < RT_LIGHT_SUBDIVS; ++r )
            {
                // Cache in local memory
                RayData ray = pRays[r];

                if( !RT_ENABLE_BOUNDING_BOXES || IntersectRayBox( ray, object.BoundingBox ) )
                {
                    // Bounding-box test passed, check each triangle to check if the ray actually intersects the object
                    // Since we are in light ray pass, only one intersection is enough
                    for( int t = 0; t < object.NumTriangles; ++t )
                    {
                        Triangle triangle = pTriangles[object.FirstTriangle + t];

                        #if RT_ENABLE_BACKFACE_CULL
                        if( ray.Direction.Dot( triangle.Normal ) >= 0 )
                        {
                            // Backface culling
                            continue;
                        }
                        #endif

                        RT::vec4 intersectionPoint = IntersectRayTriangle( ray, triangle );

                        if( intersectionPoint.x < INFINITY )
                        {
                            // The light ray hits other object
                            numIntersections++;

                            break;
                        }
                    }
                }
            }

            return numIntersections;
        }

        __global__
        void IntersectShadowRaysKernel(
            const SecondaryRayData* pRays,
            const unsigned int numRays,
            const RayData* pShadowRays,
            const unsigned int numLights,
            const ObjectData* pObjects,
            const unsigned int numObjects,
            const Triangle* pTriangles,
            int* pNumIntersections )
        {
            // Get global invocation index
            const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

            const int rayIdx = threadId / (numLights * numObjects);
            const int objIdx = threadId % numObjects;

            const int lightIdx = (rayIdx * numLights) + (threadId / (numRays * numObjects));

            const int firstShadowRayIdx = lightIdx * RT_LIGHT_SUBDIVS;

            if( rayIdx >= numRays )
            {
                return;
            }

            SecondaryRayData ray = pRays[rayIdx];

            if( ray.Intersection.Distance.x == INFINITY )
            {
                // No intersection
                return;
            }

            // Intersect with object
            int numIntersections = IntersectShadowRaysKernel(
                pShadowRays + firstShadowRayIdx,
                pObjects[objIdx],
                pTriangles );

            atomicAdd( &pNumIntersections[lightIdx], numIntersections );
        }

        __global__
        void EvaluateLightIntensityKernel(
            const SecondaryRayData* pRays,
            const unsigned int numRays,
            const int* pNumIntersections,
            const unsigned int numLights,
            SecondaryRayData* pPrimaryRays )
        {
            // Get global invocation index
            const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

            const int rayIdx = threadId / numLights;

            if( rayIdx >= numRays )
            {
                return;
            }

            SecondaryRayData ray = pRays[rayIdx];

            if( ray.Intersection.Distance.x == INFINITY )
            {
                // No intersection
                return;
            }

            // If shadow ray intersects more than one object, numIntersections is greater than RT_LIGHT_SUBDIVS
            // Clamp to have RT_LIGHT_SUBDIVS - numIntersections non-negative
            const int numIntersections = min( pNumIntersections[threadId], RT_LIGHT_SUBDIVS );

            float lightIntensity =
                (RT_LIGHT_SUBDIVS - numIntersections) * 1.0f +
                (numIntersections) * 0.2f;

            lightIntensity /= RT_LIGHT_SUBDIVS;

            // Update primary ray color intensity factor
            {
                const int primaryRayIdx = pRays[threadId].PrimaryRayIndex;

                SecondaryRayData& primaryRay = pPrimaryRays[primaryRayIdx];

                atomicAdd( &primaryRay.Intersection.Intensity, lightIntensity );
            }
        }
        
        __global__
        void SpawnSecondaryRaysKernel(
            const SecondaryRayData* pRays,
            const unsigned int numRays,
            SecondaryRayData* pSecondaryRays )
        {

        }

        __global__
        void FinalizePrimaryRaysKernel(
            const SecondaryRayData* pRays,
            const unsigned int numRays,
            unsigned char* pImageBytes )
        {
            struct uchar3 { unsigned char r, g, b; };

            // Get global invocation index
            const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

            if( threadId >= numRays )
            {
                return;
            }

            SecondaryRayData ray = pRays[threadId];

            uchar3 finalRayColor;
            finalRayColor.r = min( 255, static_cast<int>(ray.Intersection.Color.x * ray.Intersection.Intensity) );
            finalRayColor.g = min( 255, static_cast<int>(ray.Intersection.Color.y * ray.Intersection.Intensity) );
            finalRayColor.b = min( 255, static_cast<int>(ray.Intersection.Color.z * ray.Intersection.Intensity) );

            reinterpret_cast<uchar3*>(pImageBytes)[threadId] = finalRayColor;
        }

        int Application::ComputeIntersections( Array<SecondaryRayData> rays )
        {
            DispatchParameters params( rays.Size() );

            cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

            for( int i = 0; i < m_Scene.Objects.size(); ++i )
            {
                ArrayView<ObjectData> object( m_Scene.Private.ObjectDeviceMemory, i );

                ObjectIntersectionKernel<<<params.NumBlocksPerGrid, params.NumThreadsPerBlock>>>(
                    rays.Device(),
                    rays.Size(),
                    object.Device(),
                    m_Scene.Private.TriangleDeviceMemory.Device() );
            }

            Array<int> numIntersections( 1 );

            CountObjectIntersections<<<params.NumBlocksPerGrid, params.NumThreadsPerBlock>>>(
                rays.Device(),
                rays.Size(),
                numIntersections.Device() );

            // Ugh
            numIntersections.Sync();

            cudaDeviceSetCacheConfig( cudaFuncCachePreferNone );

            return numIntersections.Host( 0 );
        }

        void Application::ProcessLightIntersections( Array<SecondaryRayData> primaryRays, Array<SecondaryRayData> rays )
        {
            // For each ray-light pair
            DispatchParameters params( rays.Size() * m_Scene.Lights.size() );

            // Previously dispatched kernels have already finished executing
            m_ShadowRays = Array<RayData>( rays.Size() * m_Scene.Lights.size() * RT_LIGHT_SUBDIVS );

            SpawnShadowRaysKernel<<<params.NumBlocksPerGrid, params.NumThreadsPerBlock>>>(
                rays.Device(),
                rays.Size(),
                m_Scene.Private.LightDeviceMemory.Device(),
                m_Scene.Private.LightDeviceMemory.Size(),
                m_ShadowRays.Device() );

            params = DispatchParameters( rays.Size() * m_Scene.Lights.size() * m_Scene.Objects.size() );
            
            // Previously dispatched kernels have already finished executing
            m_NumShadowIntersections = Array<int>( rays.Size() * m_Scene.Lights.size() );
            
            IntersectShadowRaysKernel<<<params.NumBlocksPerGrid, params.NumThreadsPerBlock>>>(
                rays.Device(),
                rays.Size(),
                m_ShadowRays.Device(),
                m_Scene.Private.LightDeviceMemory.Size(),
                m_Scene.Private.ObjectDeviceMemory.Device(),
                m_Scene.Private.ObjectDeviceMemory.Size(),
                m_Scene.Private.TriangleDeviceMemory.Device(),
                m_NumShadowIntersections.Device() );
            
            params = DispatchParameters( rays.Size() * m_Scene.Lights.size() );
            
            EvaluateLightIntensityKernel<<<params.NumBlocksPerGrid, params.NumThreadsPerBlock>>>(
                rays.Device(),
                rays.Size(),
                m_NumShadowIntersections.Device(),
                m_Scene.Private.LightDeviceMemory.Size(),
                primaryRays.Device() );
        }

        Array<SecondaryRayData> Application::SpawnSecondaryRays( Array<SecondaryRayData> rays, int numIntersections )
        {
            Array<SecondaryRayData> secondaryRays;

            #if RT_MAX_RAY_DEPTH > 0
            secondaryRays = Array<SecondaryRayData>( numIntersections );

            // Process all rays
            DispatchParameters params( rays.Size() );

            SpawnSecondaryRaysKernel<<<params.NumBlocksPerGrid, params.NumThreadsPerBlock>>>(
                rays.Device(),
                rays.Size(),
                secondaryRays.Device() );
            #endif

            return secondaryRays;
        }

        void Application::FinalizePrimaryRays( Array<SecondaryRayData> primaryRays, Array<png_byte> imageData )
        {
            DispatchParameters params( primaryRays.Size() );

            FinalizePrimaryRaysKernel<<<params.NumBlocksPerGrid, params.NumThreadsPerBlock>>>(
                primaryRays.Device(),
                primaryRays.Size(),
                imageData.Device() );
        }
    }
}
