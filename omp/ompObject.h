#pragma once
#include "../Vec.h"
#include "ompTriangle.h"
#include <vector>

namespace RT::OMP
{
    struct Object
    {
        std::vector<Triangle> Triangles;
    };
}
