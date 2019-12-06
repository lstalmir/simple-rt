#pragma once
#include "../Optimizations.h"
#include "../Vec.h"
#include <limits>

namespace RT::OMP
{
    struct Box
    {
        vec4 Min;
        vec4 Max;

        inline Box()
        {
            Min = vec4( std::numeric_limits<float_t>::infinity() );
            Max = vec4( -std::numeric_limits<float_t>::infinity() );
        }
    };
}
