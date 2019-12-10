#pragma once
#include "../Optimizations.h"
#include "../Intrin.h"
#include "../Vec.h"
#include "ompRay.h"
#include <omp.h>
#include <vector>

namespace RT::OMP
{
    struct alignas(32) Camera
    {
        using RayType = RT::OMP::Ray;

        vec4 Origin;
        vec4 Direction;
        vec4 Up;
        float_t HorizontalFOV;
        float_t AspectRatio;

        std::vector<Ray> SpawnPrimaryRays( int horizontal_count, int vertical_count );
    };

    inline std::vector<Ray> Camera::SpawnPrimaryRays( int horizontal_count, int vertical_count )
    {
        std::vector<Ray> rays( vertical_count * horizontal_count *
            (RT_ENABLE_ANTIALIASING ? 4 : 1) );

        #if RT_ENABLE_INTRINSICS

        // Camera properties
        __m128 O, D, U, R;
        O = _mm_load_ps( &Origin.data );
        D = _mm_load_ps( &Direction.data );
        U = _mm_load_ps( &Up.data );
        R = Cross( U, D );
        U = Cross( D, R );

        // Compute horizontal and vertical steps
        __m128 STEP, HSTEP, VSTEP, NUM;
        __m128i NUMi;
        STEP = _mm_set1_ps( HorizontalFOV );
        NUMi = _mm_set_epi32( 0, 0, vertical_count, horizontal_count );
        NUM = _mm_cvtepi32_ps( NUMi );
        VSTEP = _mm_set_ps( 0, 0, AspectRatio, 1 );
        STEP = _mm_div_ps( STEP, NUM );
        STEP = _mm_tan_ps( STEP );
        STEP = _mm_div_ps( STEP, VSTEP );
        VSTEP = _mm_shuffle_ps( STEP, STEP, _MM_SHUFFLE( 1, 1, 1, 1 ) );
        VSTEP = _mm_mul_ps( VSTEP, U );
        HSTEP = _mm_shuffle_ps( STEP, STEP, _MM_SHUFFLE( 0, 0, 0, 0 ) );
        HSTEP = _mm_mul_ps( HSTEP, R );

        // Temporary variables
        __m128  HNUM, VNUM;
        NUMi = _mm_srli_epi32( NUMi, 1 );
        NUM = _mm_cvtepi32_ps( NUMi );
        HNUM = _mm_shuffle_ps( NUM, NUM, _MM_SHUFFLE( 0, 0, 0, 0 ) );
        VNUM = _mm_shuffle_ps( NUM, NUM, _MM_SHUFFLE( 1, 1, 1, 1 ) );

        __m128 HOFFSET, VOFFSET;
        VOFFSET = _mm_mul_ps( VNUM, VSTEP );

        if( (vertical_count & 1) == 0 )
        {
            // Adjust start offset when number of vertical rays is even
            __m128 TMP = _mm_set1_ps( 0.5f );
            TMP = _mm_mul_ps( TMP, VSTEP );
            VOFFSET = _mm_add_ps( VOFFSET, TMP );
        }

        __m128 HOFFSET_START;
        HOFFSET_START = _mm_mul_ps( HNUM, HSTEP );

        if( (horizontal_count & 1) == 0 )
        {
            // Adjust start offset when number of horizontal rays is even
            __m128 TMP = _mm_set1_ps( 0.5f );
            TMP = _mm_mul_ps( TMP, HSTEP );
            HOFFSET_START = _mm_add_ps( HOFFSET_START, TMP );
        }

        __m128 RAY_D;

        #if RT_ENABLE_ANTIALIASING
        // Adjust hstep and vstep
        __m128 TMP = _mm_set1_ps( 0.25f );
        const __m128 HSUBSTEP = _mm_mul_ps( HSTEP, TMP );
        const __m128 VSUBSTEP = _mm_mul_ps( VSTEP, TMP );
        #endif

        for( int y_ind = 0; y_ind < vertical_count; ++y_ind )
        {
            HOFFSET = HOFFSET_START;

            for( int x_ind = 0; x_ind < horizontal_count; ++x_ind )
            {
                RAY_D = _mm_add_ps( HOFFSET, D );
                RAY_D = _mm_add_ps( VOFFSET, RAY_D );

                #if RT_ENABLE_ANTIALIASING
                __m128 SUBRAY_D = _mm_sub_ps( RAY_D, HSUBSTEP );
                SUBRAY_D = _mm_sub_ps( SUBRAY_D, VSUBSTEP );
                SUBRAY_D = Normalize3( SUBRAY_D );

                Ray* pRay = &rays[4 * (y_ind * horizontal_count + x_ind)];
                _mm_store_ps( &pRay->Direction.data, SUBRAY_D );
                _mm_store_ps( &pRay->Origin.data, O );

                SUBRAY_D = _mm_add_ps( RAY_D, HSUBSTEP );
                SUBRAY_D = _mm_sub_ps( SUBRAY_D, VSUBSTEP );
                SUBRAY_D = Normalize3( SUBRAY_D );

                pRay++;
                _mm_store_ps( &pRay->Direction.data, SUBRAY_D );
                _mm_store_ps( &pRay->Origin.data, O );

                SUBRAY_D = _mm_sub_ps( RAY_D, HSUBSTEP );
                SUBRAY_D = _mm_add_ps( SUBRAY_D, VSUBSTEP );
                SUBRAY_D = Normalize3( SUBRAY_D );

                pRay++;
                _mm_store_ps( &pRay->Direction.data, SUBRAY_D );
                _mm_store_ps( &pRay->Origin.data, O );

                SUBRAY_D = _mm_add_ps( RAY_D, HSUBSTEP );
                SUBRAY_D = _mm_add_ps( SUBRAY_D, VSUBSTEP );
                SUBRAY_D = Normalize3( SUBRAY_D );

                pRay++;
                _mm_store_ps( &pRay->Direction.data, SUBRAY_D );
                _mm_store_ps( &pRay->Origin.data, O );
                #else
                RAY_D = Normalize3( RAY_D );

                Ray* pRay = &rays[y_ind * horizontal_count + x_ind];
                _mm_store_ps( &pRay->Direction.data, RAY_D );
                _mm_store_ps( &pRay->Origin.data, O );
                #endif

                HOFFSET = _mm_sub_ps( HOFFSET, HSTEP );
            }

            VOFFSET = _mm_sub_ps( VOFFSET, VSTEP );
        }

        #else

        // Camera properties
        const auto right = Up.Cross( Direction );
        const auto up = Direction.Cross( right );

        // Compute horizontal and vertical steps
        const auto vstep = up * std::tanf( HorizontalFOV / vertical_count ) / AspectRatio;
        const auto hstep = right * std::tanf( HorizontalFOV / horizontal_count );

        // Temporary variables
        auto voffset = (vertical_count / 2.f) * vstep;

        if( (vertical_count & 1) == 0 )
        {
            // Adjust start offset when number of vertical rays is even
            voffset += 0.5f * vstep;
        }

        auto hoffset_start = (horizontal_count / 2.f) * hstep;

        if( (horizontal_count & 1) == 0 )
        {
            // Adjust start offset when number of horizontal rays is even
            hoffset_start += 0.5f * hstep;
        }

        for( int y_ind = 0; y_ind < vertical_count; ++y_ind )
        {
            auto hoffset = hoffset_start;

            for( int x_ind = 0; x_ind < horizontal_count; ++x_ind )
            {
                auto ray_d = voffset + hoffset + Direction;
                ray_d.Normalize3();

                Ray* pRay = &rays[y_ind * horizontal_count + x_ind];
                pRay->Direction = ray_d;
                pRay->Origin = Origin;

                hoffset -= hstep;
            }

            voffset -= vstep;
        }

        #endif

        return rays;
    }
}
