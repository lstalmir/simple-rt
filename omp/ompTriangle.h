#pragma once
#include "../Vec.h"

namespace RT::OMP
{
    template<bool EnableIntrinsics = true>
    struct Triangle
    {
        vec4 A, B, C;
    };
}
