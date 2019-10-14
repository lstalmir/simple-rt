#pragma once
#include "Plane.h"
#include "SSE.h"
#include "Triangle.h"
#include "Vector.h"

namespace RT
{
    struct Ray
    {
        vec4 Origin;
        vec4 Direction;

        inline vec4 Intersect( const Plane& plane ) const
        {
            // Use following equation to check if ray intersects the plane
            // (ray.Begin + ray.Direction * T - plane.Origin) DOT plane.Normal = 0, T > 0

            float denominator, t;
            __m128 xmm0, xmm1, xmm2, xmm3, xmm4;
            __m128i xmm3i;

            xmm0 = _mm_load_ps( &plane.Normal.data );
            xmm4 = _mm_load_ps( &Direction.data );

            // Compute denominator first to check if it is close to 0
            xmm2 = xmm::dot( xmm0, xmm4 );
            denominator = _mm_cvtss_f32( xmm2 );

            // TODO: remove branch
            if( denominator < -1e-6f || denominator > 1e-6f )
            {
                xmm1 = _mm_load_ps( &plane.Origin.data );
                xmm3 = _mm_load_ps( &Origin.data );

                // Get vector between origins
                xmm1 = _mm_sub_ps( xmm1, xmm3 );

                // Dot with plane's normal and divide by denominator
                xmm1 = xmm::dot( xmm1, xmm0 );
                xmm0 = _mm_div_ps( xmm1, xmm2 );

                t = _mm_cvtss_f32( xmm0 );

                if( t > 0 )
                {
                    // Compute intersection point
                    xmm0 = _mm_fmadd_ps( xmm4, xmm0, xmm3 );

                    vec4 intersectionPoint;
                    _mm_store_ps( &intersectionPoint.data, xmm0 );

                    return intersectionPoint;
                }
            }
            
            // No intersection
            return vec4( std::numeric_limits<float_t>::infinity() );
        }

        inline vec4 Intersect( const Triangle& triangle ) const
        {
            // Möller–Trumbore intersection algorithm
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
            __m128 V0, V1, V2, E1, E2, P, Q, T, U, V, D, O, F, DENOM, DIST;

            // Translate triangle to (0,0,0), compute tri.Edge1 and tri.Edge2
            V0 = _mm_load_ps( &triangle.A.data );
            V1 = _mm_load_ps( &triangle.B.data );
            V2 = _mm_load_ps( &triangle.C.data );
            
            // Compute triangle edges relative to (0,0,0)
            E1 = _mm_sub_ps( V1, V0 );
            E2 = _mm_sub_ps( V2, V0 );

            // Compute factor denominator
            D = _mm_load_ps( &Direction.data );
            P = xmm::cross3( D, E2 );
            DENOM = xmm::dot( P, E1 );

            denominator = _mm_cvtss_f32( DENOM );

            // First condition of intersection:
            if( denominator < -1e-6f || denominator > 1e-6f )
            {
                // F = 1 / denom
                F = _mm_rcp_ps( DENOM );

                // Calculate distance from V0 to ray origin
                O = _mm_load_ps( &Origin.data );
                T = _mm_sub_ps( O, V0 );

                // Calculate u parameter and test bounds
                U = xmm::dot( P, T );
                U = _mm_mul_ps( F, U );

                u = _mm_cvtss_f32( U );

                // Second condition of intersection:
                if( u >= 0 && u <= 1 )
                {
                    Q = xmm::cross3( T, E1 );

                    // Calculate v parameter and test bounds
                    V = xmm::dot( Q, D );
                    V = _mm_mul_ps( F, V );

                    v = _mm_cvtss_f32( V );

                    // Third and fourth condition of intersection:
                    if( v >= 0 && v <= 1 && u + v <= 1 )
                    {
                        // Calculate t, if t > 0, the ray intersects the triangle
                        DIST = xmm::dot( Q, E2 );
                        DIST = _mm_mul_ps( F, DIST );

                        t = _mm_cvtss_f32( DIST );

                        if( t > 0 )
                        {
                            // Compute intersection point
                            O = _mm_fmadd_ps( D, DIST, O );

                            vec4 intersectionPoint;
                            _mm_store_ps( &intersectionPoint.data, O );

                            return intersectionPoint;
                        }
                    }
                }
            }

            // No intersection
            return vec4( std::numeric_limits<float_t>::infinity() );
        }

        inline Ray Reflect( const Plane& plane, const vec4& intersectionPoint ) const
        {
            // Use following equation to compute reflection ray
            // (ray.Direction - 2 * (ray.Direction DOT plane.Normal) * plane.Normal

            // Assume the ray is in the direction to the plane

            __m128 xmm0, xmm1, xmm2, xmm3;
            __m128i xmm3i;

            xmm0 = _mm_load_ps( &plane.Normal.data );
            xmm1 = _mm_load_ps( &Direction.data );

            xmm3 = _mm_set1_ps( 2 );

            xmm2 = xmm::dot( xmm1, xmm0 );
            xmm2 = _mm_mul_ps( xmm2, xmm0 );
            xmm2 = _mm_mul_ps( xmm2, xmm3 );
            xmm0 = _mm_sub_ps( xmm1, xmm2 );

            Ray reflectedRay;
            reflectedRay.Origin = intersectionPoint;
            _mm_store_ps( &reflectedRay.Direction.data, xmm0 );

            return reflectedRay;
        }
    };
}
