#pragma once
#include "ompIntrin.h"
#include "ompPlane.h"
#include "ompRay.h"

namespace RT
{
    inline Ray Reflect( const Ray& ray, const Plane& plane, const vec4& intersectionPoint )
    {
        // Use following equation to compute reflection ray
        // (ray.Direction - 2 * (ray.Direction DOT plane.Normal) * plane.Normal

        // Assume the ray is in the direction to the plane

        __m128 xmm0, xmm1, xmm2, xmm3;
        __m128i xmm3i;

        xmm0 = _mm_load_ps( &plane.Normal.data );
        xmm1 = _mm_load_ps( &ray.Direction.data );

        xmm3 = _mm_set1_ps( 2 );

        xmm2 = Dot( xmm1, xmm0 );
        xmm2 = _mm_mul_ps( xmm2, xmm0 );
        xmm2 = _mm_mul_ps( xmm2, xmm3 );
        xmm0 = _mm_sub_ps( xmm1, xmm2 );

        Ray reflectedRay;
        reflectedRay.Origin = intersectionPoint;
        _mm_store_ps( &reflectedRay.Direction.data, xmm0 );

        return reflectedRay;
    }
}
