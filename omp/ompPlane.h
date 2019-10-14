#pragma once
#include "ompVector.h"

namespace RT
{
    struct alignas(32) Plane
    {
        vec4 Origin;
        vec4 Normal;
    };
}
