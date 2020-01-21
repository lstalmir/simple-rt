#pragma once
#include "../Optimizations.h"
#include "../Vec.h"
#include "cudaCommon.h"
#include "cudaMemory.h"
#include "cudaTriangle.h"

namespace RT
{
    namespace CUDA
    {
        struct RayData
        {
            vec4 Origin;
            vec4 Direction;
        };

        struct Ray : DataWrapper<RayData>
        {
            Ray() = default;
            Ray( const Array<RayData>& array, int index );
        };

        enum class SecondaryRayType
        {
            ePrimary,
            eReflection,
            eRefraction
        };

        struct IntersectionData
        {
            vec4 Distance;
            Triangle Triangle;
            vec4 Color;
            float Ior;
        };

        struct SecondaryRayData
        {
            RayData Ray;
            int PrimaryRayIndex;
            int PreviousRayIndex;
            int Depth;
            IntersectionData Intersection;
            SecondaryRayType Type;
        };

        struct SecondaryRay : DataWrapper<SecondaryRayData>
        {
            SecondaryRay() = default;
            SecondaryRay( const Array<SecondaryRayData>& array, int index );
        };
    }
}
