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
        };

        struct Object : DataWrapper<ObjectData>
        {
            using BoxType = RT::CUDA::Box;
            using TriangleType = RT::CUDA::Triangle;

            Array<TriangleType> DeviceTriangles;
            int FirstTriangle;
            int NumTriangles;
        };
    }
}
