#pragma once
#include "../Vec.h"
#include "cudaCommon.h"
#include "cudaMemory.h"
#include "cudaRay.h"

namespace RT
{
    namespace CUDA
    {
        struct CameraData
        {
            vec4 Origin;
            vec4 Direction;
            vec4 Up;
            float_t HorizontalFOV;
            float_t AspectRatio;
        };

        struct Camera : DataWrapper<CameraData>
        {
            using RayType = RT::CUDA::Ray;

            Camera() = default;
            Camera( const Array<CameraData>& array, int index );

            Array<RayType::DataType> SpawnPrimaryRays( int hCount, int vCount );
        };
    }
}
