#pragma once
#include "ompVector.h"

namespace RT
{
    struct alignas(32) Ray
    {
        vec4 Origin;
        vec4 Direction;
    };
}
