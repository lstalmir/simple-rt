#pragma once
#include "../Optimizations.h"
#include "../Vec.h"
#include "cudaCommon.h"

NAMESPACE_RT_CUDA
{
    struct RayData
    {
        vec4 Origin;
        vec4 Direction;
    };

    struct Ray : DataWrapper<RayData>
    {
    };
}}
