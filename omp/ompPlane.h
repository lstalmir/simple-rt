#pragma once
#include "../Optimizations.h"
#include "../Vec.h"

namespace RT::OMP
{
    struct alignas(32) Plane
    {
        vec4 Origin;
        vec4 Normal;
    };
}
