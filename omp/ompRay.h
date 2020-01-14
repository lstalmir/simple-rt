#pragma once
#include "../Optimizations.h"
#include "../Vec.h"
#include "ompBox.h"
#include "ompPlane.h"
#include "ompTriangle.h"
#include <algorithm>

namespace RT::OMP
{
    struct alignas(32) Ray
    {
        vec4 Origin;
        vec4 Direction;

        vec4 Intersect( const Plane & plane ) const;
        vec4 Intersect( const Triangle & triangle ) const;
        bool Intersect( const Box & box ) const;
        Ray Reflect( const vec4 & normal, const vec4 & intersectionPoint ) const;
        float Fresnel( const vec4 & normal, float ior ) const;
    };

    inline vec4 Ray::Intersect( const Plane& plane ) const
    {
        // Use following equation to check if ray intersects the plane
        // (ray.Origin + ray.Direction * T - plane.Origin) DOT plane.Normal = 0, T > 0

        #if RT_ENABLE_INTRINSICS

        float denominator;
        __m128 N, D, O, P, T, DENOM, TEST, ZEROS;

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

        #else

        const float denominator = plane.Normal.Dot( Direction );

        // TODO: remove branch
        if( denominator < -1e-6f || denominator > 1e-6f )
        {
            const auto P = plane.Origin - Origin;
            const auto T = P.Dot( plane.Normal ) / denominator;

            if( T >= 0 )
            {
                return vec4( T );
            }
        }

        #endif

        // No intersection
        return vec4( std::numeric_limits<float_t>::infinity() );
    }


    inline vec4 Ray::Intersect( const Triangle& triangle ) const
    {
        // Moller-Trumbore intersection algorithm
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

        #if RT_ENABLE_INTRINSICS

        float denominator;
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
                        DIST = _mm_movelh_ps( DIST, U );
                        DIST = _mm_shuffle_ps( DIST, DIST, _MM_SHUFFLE( 2, 0, 2, 0 ) );
                        DIST = _mm_movelh_ps( DIST, V );

                        vec4 intersectionFactor;
                        _mm_store_ps( &intersectionFactor.data, DIST );

                        return intersectionFactor;
                    }
                }
            }
        }

        #else

        // Compute triangle edges relative to (0,0,0)
        const auto E1 = triangle.B - triangle.A;
        const auto E2 = triangle.C - triangle.A;

        // Compute factor denominator
        const auto P = Direction.Cross( E2 );
        const float denominator = P.Dot( E1 );

        // First condition of intersection:
        if( denominator < -1e-6f || denominator > 1e-6f )
        {
            // Calculate distance from V0 to ray origin
            const auto T = Origin - triangle.A;

            // Calculate u parameter and test bounds
            const auto U = P.Dot( T ) / denominator;

            // Second condition of intersection:
            if( U >= 0 && U <= 1 )
            {
                const auto Q = T.Cross( E1 );

                // Calculate v parameter and test bounds
                const auto V = Q.Dot( Direction ) / denominator;

                // Third and fourth condition of intersection:
                if( V >= 0 && (U + V) <= 1 )
                {
                    // Calculate t, if t > 0, the ray intersects the triangle
                    const auto distance = Q.Dot( E2 ) / denominator;

                    if( distance > 0 )
                    {
                        return vec4( distance, U, V, V );
                    }
                }
            }
        }

        #endif

        // No intersection
        return vec4( std::numeric_limits<float_t>::infinity() );
    }


    inline bool Ray::Intersect( const Box& box ) const
    {
        #if RT_ENABLE_INTRINSICS

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
        
        #else

        float tmin = (box.Min.x - Origin.x) / Direction.x;
        float tmax = (box.Max.x - Origin.x) / Direction.x;

        if( tmin > tmax ) std::swap( tmin, tmax );

        float tymin = (box.Min.y - Origin.y) / Direction.y;
        float tymax = (box.Max.y - Origin.y) / Direction.y;

        if( tymin > tymax ) std::swap( tymin, tymax );

        if( (tmin > tymax) || (tymin > tmax) )
            return false;

        if( tymin > tmin )
            tmin = tymin;

        if( tymax < tmax )
            tmax = tymax;

        float tzmin = (box.Min.z - Origin.z) / Direction.z;
        float tzmax = (box.Max.z - Origin.z) / Direction.z;

        if( tzmin > tzmax ) std::swap( tzmin, tzmax );

        if( (tmin > tzmax) || (tzmin > tmax) )
            return false;

        if( tzmin > tmin )
            tmin = tzmin;

        if( tzmax < tmax )
            tmax = tzmax;

        return true;

        #endif
    }


    inline Ray Ray::Reflect( const vec4& normal, const vec4& intersectionPoint ) const
    {
        // Use following equation to compute reflection ray
        // (ray.Direction - 2 * (ray.Direction DOT plane.Normal) * plane.Normal

        // Assume the ray is in the direction to the plane
        Ray reflectedRay;

        #if RT_ENABLE_INTRINSICS

        __m128 xmm0, xmm1, xmm2, xmm3;

        xmm0 = _mm_load_ps( &normal.data );
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

        #else

        reflectedRay.Direction = Direction - normal * (2 * Direction.Dot( normal ));
        reflectedRay.Origin = Direction * intersectionPoint + Origin;

        #endif

        return reflectedRay;
    }


    inline float Ray::Fresnel( const vec4& normal, float ior ) const
    {
        if( ior == 0 )
        {
            return 0;
        }

        #if RT_ENABLE_INTRINSICS && 0

        __m128 N, D, IOR, COSI, ETAI, ETAT, TEST;

        const __m128 ONES = _mm_set1_ps( 1.f );
        const __m128 ZEROS = _mm_setzero_ps();

        N = _mm_load_ps( &normal.data );
        D = _mm_load_ps( &Direction.data );
        COSI = Dot( D, N );
        COSI = _mm_max_ps( _mm_set1_ps( -1.f ), COSI );
        COSI = _mm_min_ps( ONES, COSI );

        ETAI = ONES;
        ETAT = _mm_set1_ps( ior );

        TEST = _mm_cmpgt_ps( COSI, ZEROS );
        if( _mm_cvtss_i32( TEST ) )
        {
            __m128 TMP = ETAI;
            ETAI = ETAT;
            ETAT = TMP;
        }

        // Compute sini using Snell's law
        __m128 SINT, COSI2;

        COSI2 = _mm_mul_ps( COSI, COSI );
        COSI2 = _mm_sub_ps( ONES, COSI2 );
        COSI2 = _mm_max_ps( ZEROS, COSI2 );
        COSI2 = _mm_sqrt_ps( COSI2 );
        SINT = _mm_div_ps( ETAI, ETAT );
        SINT = _mm_mul_ps( SINT, COSI2 );

        TEST = _mm_cmpge_ps( SINT, ONES );
        if( _mm_cvtss_i32( TEST ) )
        {
            return 1;
        }



        #else

        float cosi = std::min( 1.f, std::max( -1.f, Direction.Dot( normal ) ) );
        float etai = 1;
        float etat = ior;

        if( cosi > 0 )
        {
            std::swap( etai, etat );
        }

        // Compute sini using Snell's law
        const float sint = etai / etat * sqrtf( std::max( 0.f, 1 - cosi * cosi ) );

        // Total internal reflection
        if( sint >= 1 )
        {
            return 1;
        }

        const float cost = sqrtf( std::max( 0.f, 1 - sint * sint ) );

        cosi = fabsf( cosi );

        const float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        const float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));

        return (Rs * Rs + Rp * Rp) / 2;

        #endif
    }
}
