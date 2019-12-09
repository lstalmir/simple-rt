#pragma once
#include "../Optimizations.h"
#include "../Vec.h"

namespace RT::OMP
{
    struct Triangle
    {
        vec4 A, B, C;
        vec4 An, Bn, Cn;
        #ifdef RT_ENABLE_BACKFACE_CULL
        vec4 Normal;
        #endif
    };
}
