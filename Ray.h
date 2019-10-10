#pragma once
#include "Plane.h"
#include "Vector.h"

namespace RT
{
    struct Ray
    {
        Vector3 Origin;
        Vector3 Direction;

        bool Intersects( const Plane& plane ) const;
    };
}
