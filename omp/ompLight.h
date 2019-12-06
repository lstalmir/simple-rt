#pragma once
#include "../Optimizations.h"
#include "../Intrin.h"
#include "../Vec.h"
#include "ompRay.h"
#include <omp.h>
#include <vector>

namespace RT::OMP
{
    struct Light
    {
        using RayType = RT::OMP::Ray;

        RT::vec4 Position;
        RT::float_t ShadowBias = 0.04f;
        int Subdivs = 10;

        std::vector<Ray> SpawnSecondaryRays( const Ray& primaryRay, RT::float_t intersectionDistance ) const;
    };


    inline std::vector<Ray> Light::SpawnSecondaryRays( const Ray& primaryRay, RT::float_t intersectionDistance ) const
    {
        std::vector<Ray> secondaryRays( Subdivs );

        #if !RT_DISABLE_INTRINSICS

        // Ray from intersection to light
        __m128 DIST, O, D, LO, BIAS;
        DIST = _mm_set1_ps( intersectionDistance );
        BIAS = _mm_set1_ps( ShadowBias );
        DIST = _mm_sub_ps( DIST, BIAS );
        O = _mm_load_ps( &primaryRay.Origin.data );
        D = _mm_load_ps( &primaryRay.Direction.data );
        O = _mm_fmadd_ps( DIST, D, O );

        Ray secondaryRay;
        _mm_store_ps( &secondaryRay.Origin.data, O );

        for( int i = 0; i < Subdivs; ++i )
        {
            // Simulate light size for soft shadows
            __m128 NOISE = _mm_set_ps( 0,
                static_cast<RT::float_t>(rand()) / RAND_MAX,
                static_cast<RT::float_t>(rand()) / RAND_MAX,
                static_cast<RT::float_t>(rand()) / RAND_MAX );

            LO = _mm_load_ps( &Position.data );
            LO = _mm_add_ps( LO, NOISE );
            D = _mm_sub_ps( LO, O );
            D = Normalize3( D );

            _mm_store_ps( &secondaryRay.Direction.data, D );

            // TODO: Store directly in the vector
            secondaryRays[i] = secondaryRay;
        }

        #else
        // TODO
        #endif

        return secondaryRays;
    }
}
