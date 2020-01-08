#pragma once
#include "../Optimizations.h"
#include "../Intrin.h"
#include "../Vec.h"
#include "cudaCommon.h"
#include "cudaMemory.h"
#include "cudaRay.h"

NAMESPACE_RT_CUDA
{
    struct LightData
    {
        vec4 Position;
        float_t ShadowBias = 0.04f;
        int Subdivs = 10;
        float_t Radius = 1.f;
    };

    struct Light : DataWrapper<LightData>
    {
        using RayType = RT::CUDA::Ray;

        Array<RayType::DataType> SpawnSecondaryRays( const RayType& primaryRay, float_t intersectionDistance ) const;
    };
}}
