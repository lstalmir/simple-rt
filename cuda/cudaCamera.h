#pragma once
#include "../Vec.h"
#include "cudaCommon.h"
#include "cudaMemory.h"
#include "cudaRay.h"

NAMESPACE_RT_CUDA
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

        Array<RayType::DataType> SpawnPrimaryRays( int hCount, int vCount );
    };
}}
