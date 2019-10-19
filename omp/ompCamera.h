#pragma once
#include "ompIntrin.h"
#include "ompRay.h"
#include "ompVector.h"

namespace RT
{
    struct alignas(32) Camera
    {
        vec4 Origin;
        vec4 Direction;
        vec4 Up;
        float_t HorizontalFOV;
        float_t AspectRatio;
        float_t FocalLength;
    };

    inline void SpawnPrimaryRays( const Camera& camera, int horizontal_count, int vertical_count, Ray* pRays )
    {
        // Camera properties
        __m128 O, D, U, R;
        O = _mm_load_ps( &camera.Origin.data );
        D = _mm_load_ps( &camera.Direction.data );
        U = _mm_load_ps( &camera.Up.data );
        R = Cross( U, D );
        U = Cross( D, R );

        // Compute horizontal and vertical steps
        __m128 STEP, HSTEP, VSTEP, NUM;
        __m128i NUMi;
        STEP = _mm_set1_ps( camera.HorizontalFOV );
        NUMi = _mm_set_epi32( 0, 0, vertical_count, horizontal_count );
        NUM = _mm_cvtepi32_ps( NUMi );
        VSTEP = _mm_set_ps( 0, 0, camera.AspectRatio, 1 );
        STEP = _mm_mul_ps( STEP, VSTEP );
        STEP = _mm_div_ps( STEP, NUM );
        STEP = _mm_tan_ps( STEP );
        VSTEP = _mm_set1_ps( camera.FocalLength );
        STEP = _mm_mul_ps( STEP, VSTEP );
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

        __m128 HOFFSET, VOFFSET, NEG;
        __m128i NEGi;
        NEGi = _mm_set1_epi32( 0x80000000 );
        NEG = _mm_castsi128_ps( NEGi );
        VOFFSET = _mm_mul_ps( VNUM, VSTEP );
        VOFFSET = _mm_xor_ps( VOFFSET, NEG );

        if( (vertical_count & 1) == 0 )
        {
            // Adjust start offset when number of vertical rays is even
            __m128 TMP = _mm_set1_ps( 0.5f );
            TMP = _mm_mul_ps( TMP, VSTEP );
            VOFFSET = _mm_add_ps( VOFFSET, TMP );
        }

        __m128 HOFFSET_START;
        HOFFSET_START = _mm_mul_ps( HNUM, HSTEP );
        HOFFSET_START = _mm_xor_ps( HOFFSET_START, NEG );

        if( (horizontal_count & 1) == 0 )
        {
            // Adjust start offset when number of horizontal rays is even
            __m128 TMP = _mm_set1_ps( 0.5f );
            TMP = _mm_mul_ps( TMP, HSTEP );
            HOFFSET_START = _mm_add_ps( HOFFSET_START, TMP );
        }

        __m128 RAY_D;

        for( int y_ind = 0; y_ind < vertical_count; ++y_ind )
        {
            HOFFSET = HOFFSET_START;

            for( int x_ind = 0; x_ind < horizontal_count; ++x_ind )
            {
                RAY_D = _mm_add_ps( HOFFSET, D );
                RAY_D = _mm_add_ps( VOFFSET, RAY_D );
                RAY_D = Normalize3( RAY_D );

                Ray* pRay = &pRays[y_ind * horizontal_count + x_ind];
                _mm_store_ps( &pRay->Direction.data, RAY_D );
                _mm_store_ps( &pRay->Origin.data, O );

                HOFFSET = _mm_add_ps( HOFFSET, HSTEP );
            }

            VOFFSET = _mm_add_ps( VOFFSET, VSTEP );
        }
    }
}
