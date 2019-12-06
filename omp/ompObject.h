#pragma once
#include "../Optimizations.h"
#include "../Vec.h"
#include "ompBox.h"
#include "ompTriangle.h"
#include <vector>

namespace RT::OMP
{
    struct Object
    {
        using BoxType = RT::OMP::Box;
        using TriangleType = RT::OMP::Triangle;

        BoxType BoundingBox;
        std::vector<TriangleType> Triangles;
        RT::vec4 Color;
    };

    struct ArrayObject
    {
        using BoxType = RT::OMP::Box;
        using TriangleType = RT::OMP::Triangle;

        // Number of objects in the array
        uint32_t Count = 0;

        // Number of triangles in each object
        uint32_t* NumTriangles = nullptr;
        // First triangle offset of each object
        uint32_t* StartTriangles = nullptr;

        BoxType* BoundingBoxes = nullptr;
        TriangleType* Trianges = nullptr;
        RT::vec4* Colors = nullptr;
    };
}
