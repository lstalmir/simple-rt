#pragma once
#include "Vector.h"

namespace RT
{
    struct Plane
    {
        Vector3 Origin;
        Vector3 Normal;

        Plane() = default;
        Plane( const Vector3& origin, const Vector3& normal );
    };
}
