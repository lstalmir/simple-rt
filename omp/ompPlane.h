#pragma once
#include "../Vec.h"

namespace RT::OMP
{
    template<bool EnableIntrinsics = true>
    struct alignas(32) Plane
    {
        vec4 Origin;
        vec4 Normal;
    };
}
