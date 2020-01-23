#pragma once
#include "../Optimizations.h"
#include "../Vec.h"
#include "cudaCommon.h"
#include "cudaBox.h"
#include "cudaMemory.h"
#include "cudaTriangle.h"

namespace RT
{
    namespace CUDA
    {
        struct ObjectData
        {
            Box BoundingBox;
            vec4 Color;
            float Ior = 0.f;
            int FirstTriangle;
            int NumTriangles;
        };

        struct Object : DataWrapper<ObjectData>
        {
            using BoxType = RT::CUDA::Box;
            using TriangleType = RT::CUDA::Triangle;

            Object() = default;
            Object( const Array<ObjectData>& array, int index );

            Array<TriangleType> Triangles;
        };
    }
}
