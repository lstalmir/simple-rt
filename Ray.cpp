#include "Ray.h"

bool RT::Ray::Intersects( const RT::Plane& plane ) const
{
    // Use following equation to check if ray intersects the plane
    // (ray.Begin + ray.Direction * T - plane.Origin) DOT plane.Normal = 0, T > 0

    float denominator, t;
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128i xmm3i;

    xmm3i = _mm_set1_epi32( 0x80000000 );
    xmm3 = _mm_castsi128_ps( xmm3i );

    xmm0 = _mm_load_ps( &plane.Normal.data );
    xmm0 = _mm_xor_ps( xmm0, xmm3 );
    xmm1 = _mm_load_ps( &Direction.data );

    // Compute denominator first to check if it is close to 0
    xmm2 = dot( xmm0, xmm1 );
    _mm_store_ss( &denominator, xmm2 );

    if( denominator > 1e-6f )
    {
        xmm1 = _mm_load_ps( &plane.Origin.data );
        xmm3 = _mm_load_ps( &Origin.data );

        // Get vector between origins
        xmm1 = _mm_sub_ps( xmm1, xmm3 );

        // Dot with plane's normal and divide by denominator
        xmm1 = dot( xmm1, xmm0 );
        xmm0 = _mm_div_ss( xmm1, xmm2 );

        _mm_store_ss( &t, xmm0 );

        // Intersection with ray occurs only in the direction of the ray
        return t >= 0;
    }

    // No intersection
    return false;
}
