#pragma once
#include "../Vec.h"
#include "ompBox.h"
#include "ompTriangle.h"
#include <vector>

namespace RT::OMP
{
    struct Object
    {
        Box BoundingBox;
        std::vector<Triangle> Triangles;
        RT::vec4 Color;
    };
}
