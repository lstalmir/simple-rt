#pragma once
#include "Vector.h"

namespace RT
{
    struct Plane
    {
        vec4 Origin;
        vec4 Normal;

        Plane() = default;
        Plane( const vec4& origin, const vec4& normal );
    };
}
