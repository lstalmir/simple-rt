#pragma once
#include "../Optimizations.h"
#include "../Vec.h"
#include "cudaCommon.h"

namespace RT
{
    namespace CUDA
    {
        struct RayData
        {
            vec4 Origin;
            vec4 Direction;
        };

        struct Ray : DataWrapper<RayData>
        {
        };
    }
}
