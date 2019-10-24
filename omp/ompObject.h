#pragma once
#include "../Vec.h"
#include "ompBox.h"
#include "ompTriangle.h"
#include <vector>

namespace RT::OMP
{
    template<bool EnableIntrinsics = true>
    struct Object
    {
        using BoxType = Box<EnableIntrinsics>;
        using TriangleType = Triangle<EnableIntrinsics>;

        BoxType BoundingBox;
        std::vector<TriangleType> Triangles;
        RT::vec4 Color;
    };
}
