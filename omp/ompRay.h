#pragma once
#include "../Vec.h"
#include "ompBox.h"
#include "ompPlane.h"
#include "ompTriangle.h"

namespace RT::OMP
{
    template<bool EnableIntrinsics = true>
    struct alignas(32) Ray
    {
        vec4 Origin;
        vec4 Direction;

        vec4 Intersect( const Plane<EnableIntrinsics> & plane ) const;
        vec4 Intersect( const Triangle<EnableIntrinsics> & triangle ) const;
        bool Intersect( const Box<EnableIntrinsics> & box ) const;
        Ray Reflect( const Plane<EnableIntrinsics> & plane, const vec4 & intersectionPoint ) const;
    };

    template<>
    inline vec4 Ray<true>::Intersect( const Plane<true> & plane ) const
    {
        // Use following equation to check if ray intersects the plane
        // (ray.Origin + ray.Direction * T - plane.Origin) DOT plane.Normal = 0, T > 0
        
        float denominator, t;
        __m128 N, D, O, P, T, DENOM, TEST, ZEROS;
        __m128i xmm3i;

        ZEROS = _mm_setzero_ps();

        N = _mm_load_ps( &plane.Normal.data );
        D = _mm_load_ps( &Direction.data );

        // Compute denominator first to check if it is close to 0
        DENOM = Dot( N, D );
        denominator = _mm_cvtss_f32( DENOM );

        // TODO: remove branch
        if( denominator < -1e-6f || denominator > 1e-6f )
        {
            P = _mm_load_ps( &plane.Origin.data );
            O = _mm_load_ps( &Origin.data );

            // Get vector between origins
            P = _mm_sub_ps( P, O );

            // Dot with plane's normal and divide by denominator
            T = Dot( P, N );
            T = _mm_div_ps( T, DENOM );

            TEST = _mm_cmpge_ps( T, ZEROS );
            if( _mm_cvtss_i32( TEST ) )
            {
                vec4 intersectionFactor;
                _mm_store_ps( &intersectionFactor.data, T );

                return intersectionFactor;
            }
        }
        
        // No intersection
        return vec4( std::numeric_limits<float_t>::infinity() );
    }

    template<>
    inline vec4 Ray<false>::Intersect( const Plane<false> & plane ) const
    {
        // TODO
        return vec4( std::numeric_limits<float_t>::infinity() );
    }


    template<>
    inline vec4 Ray<true>::Intersect( const Triangle<true> & triangle ) const
    {
        // M�ller�Trumbore intersection algorithm
        //
        // | t |          1          | Q DOT tri.Edge2     |
        // | u | = --------------- * | P DOT T             |
        // | v |   P DOT tri.Edge1   | Q DOT ray.Direction |
        //
        // where P = ray.Direction CROSS tri.Edge2
        //       T = ray.Origin - tri.Vert0
        //       Q = T CROSS tri.Edge1
        //       u > 0 and u < 1
        //       v > 0 and v < 1
        //       u + v < 1
        //
        // Ray does not intersect the triangle if:
        //       P DOT tri.Edge1 ~= 0 or
        //       u < 0 or u > 1 or
        //       v < 0 or v > 1 or
        //       u + v > 1
        
        float denominator, u, v, t;
        __m128 V0, V1, V2, E1, E2, P, Q, T, U, V, TEST, D, O, F, DENOM, DIST, ZEROS, ONES;

        // Translate triangle to (0,0,0), compute tri.Edge1 and tri.Edge2
        V0 = _mm_load_ps( &triangle.A.data );
        V1 = _mm_load_ps( &triangle.B.data );
        V2 = _mm_load_ps( &triangle.C.data );

        // Compute triangle edges relative to (0,0,0)
        E1 = _mm_sub_ps( V1, V0 );
        E2 = _mm_sub_ps( V2, V0 );

        // Compute factor denominator
        D = _mm_load_ps( &Direction.data );
        P = Cross( D, E2 );
        DENOM = Dot( P, E1 );

        denominator = _mm_cvtss_f32( DENOM );

        // First condition of intersection:
        if( denominator < -1e-6f || denominator > 1e-6f )
        {
            ZEROS = _mm_setzero_ps();
            ONES = _mm_set1_ps( 1.0f );

            // F = 1 / denom
            F = _mm_rcp_ps( DENOM );

            // Calculate distance from V0 to ray origin
            O = _mm_load_ps( &Origin.data );
            T = _mm_sub_ps( O, V0 );

            // Calculate u parameter and test bounds
            U = Dot( P, T );
            U = _mm_mul_ps( F, U );

            // Second condition of intersection:
            TEST = _mm_and_ps(
                _mm_cmpge_ps( U, ZEROS ),
                _mm_cmple_ps( U, ONES ) );
            if( _mm_cvtss_i32( TEST ) )
            {
                Q = Cross( T, E1 );

                // Calculate v parameter and test bounds
                V = Dot( Q, D );
                V = _mm_mul_ps( F, V );

                // Third and fourth condition of intersection:
                TEST = _mm_and_ps(
                    _mm_cmpge_ps( V, ZEROS ),
                    _mm_cmple_ps( _mm_add_ps( U, V ), ONES ) );
                if( _mm_cvtss_i32( TEST ) )
                {
                    // Calculate t, if t > 0, the ray intersects the triangle
                    DIST = Dot( Q, E2 );
                    DIST = _mm_mul_ps( F, DIST );

                    TEST = _mm_cmpge_ps( DIST, ZEROS );
                    if( _mm_cvtss_i32( TEST ) )
                    {
                        vec4 intersectionFactor;
                        _mm_store_ps( &intersectionFactor.data, DIST );

                        return intersectionFactor;
                    }
                }
            }
        }

        // No intersection
        return vec4( std::numeric_limits<float_t>::infinity() );
    }

    template<>
    inline vec4 Ray<false>::Intersect( const Triangle<false> & triangle ) const
    {
        // TODO
        return vec4( std::numeric_limits<float_t>::infinity() );
    }


    template<>
    inline bool Ray<true>::Intersect( const Box<true> & box ) const
    {
        __m128 TMIN1, TMIN2, TMAX1, TMAX2, O, D, TEST;

        // Load ray
        O = _mm_load_ps( &Origin.data );
        D = _mm_load_ps( &Direction.data );
        D = _mm_rcp_ps( D );

        // Load bounding box extents
        TMIN1 = _mm_load_ps( &box.Min.data );
        TMAX1 = _mm_load_ps( &box.Max.data );
        TMIN1 = _mm_sub_ps( TMIN1, O );
        TMIN1 = _mm_mul_ps( TMIN1, D );
        TMAX1 = _mm_sub_ps( TMAX1, O );
        TMAX1 = _mm_mul_ps( TMAX1, D );
        TMIN2 = _mm_min_ps( TMIN1, TMAX1 );
        TMAX2 = _mm_max_ps( TMIN1, TMAX1 );
        TMAX1 = _mm_shuffle_ps( TMAX2, TMAX2, _MM_SHUFFLE( 0, 2, 1, 0 ) );
        TMAX2 = _mm_shuffle_ps( TMAX2, TMAX2, _MM_SHUFFLE( 1, 2, 1, 0 ) );
        TMAX1 = _mm_min_ps( TMAX1, TMAX2 );
        TMIN1 = _mm_shuffle_ps( TMIN2, TMIN2, _MM_SHUFFLE( 2, 0, 1, 0 ) );
        TMIN2 = _mm_shuffle_ps( TMIN2, TMIN2, _MM_SHUFFLE( 2, 1, 1, 0 ) );
        TMIN1 = _mm_max_ps( TMIN1, TMIN2 );
        TMAX1 = _mm_shuffle_ps( TMAX1, TMAX1, _MM_SHUFFLE( 3, 2, 0, 1 ) );
        TEST = _mm_cmple_ps( TMIN1, TMAX1 );

        // Read most significant bit of each component
        return _mm_movemask_ps( TEST ) == 0xF;
    }

    template<>
    inline bool Ray<false>::Intersect( const Box<false> & box ) const
    {
        // TODO
        return false;
    }


    template<>
    inline Ray<true> Ray<true>::Reflect( const Plane<true> & plane, const vec4 & intersectionPoint ) const
    {
        // Use following equation to compute reflection ray
        // (ray.Direction - 2 * (ray.Direction DOT plane.Normal) * plane.Normal

        // Assume the ray is in the direction to the plane
        Ray<true> reflectedRay;

        __m128 xmm0, xmm1, xmm2, xmm3;
        __m128i xmm3i;

        xmm0 = _mm_load_ps( &plane.Normal.data );
        xmm1 = _mm_load_ps( &Direction.data );

        xmm3 = _mm_set1_ps( 2 );

        xmm2 = Dot( xmm1, xmm0 );
        xmm2 = _mm_mul_ps( xmm2, xmm0 );
        xmm2 = _mm_mul_ps( xmm2, xmm3 );
        xmm0 = _mm_sub_ps( xmm1, xmm2 );

        xmm2 = _mm_load_ps( &Origin.data );
        xmm3 = _mm_load_ps( &intersectionPoint.data );
        xmm1 = _mm_fmadd_ps( xmm1, xmm3, xmm2 );

        _mm_store_ps( &reflectedRay.Origin.data, xmm1 );
        _mm_store_ps( &reflectedRay.Direction.data, xmm0 );

        return reflectedRay;
    }

    template<>
    inline Ray<false> Ray<false>::Reflect( const Plane<false> & plane, const vec4 & intersectionPoint ) const
    {
        Ray<false> nullRay;
        nullRay.Origin = vec4( 0 );
        nullRay.Direction = vec4( 0 );

        // TODO
        return nullRay;
    }
}
