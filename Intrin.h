#pragma once
#include <immintrin.h>

#if !defined _MSC_VER
#define USE_SSE2
#include <sse_mathfun_extension.h>
#define _mm_tan_ps tan_ps
#define _mm_cot_ps cot_ps
#define _mm_cvtss_i32 _mm_cvtss_si32
#endif

namespace RT
{
    inline float Radians( float angle )
    {
        return angle * (3.1415926536f / 180.0f);
    }

    inline __m128 Dot( __m128 a, __m128 b )
    {
        __m128 xmm0, xmm1, xmm2;
        xmm0 = _mm_mul_ps( a, b );
        xmm1 = _mm_movehdup_ps( xmm0 );
        xmm2 = _mm_add_ps( xmm0, xmm1 );
        xmm1 = _mm_movehl_ps( xmm1, xmm2 );
        xmm2 = _mm_add_ps( xmm2, xmm1 );
        return _mm_shuffle_ps( xmm2, xmm2, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    }

    inline __m128 Length4( __m128 a )
    {
        return _mm_sqrt_ps( Dot( a, a ) );
    }

    inline __m128 Length3( __m128 a )
    {
        __m128 xmm0;
        __m128i xmm0i;
        xmm0i = _mm_set_epi32( 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );
        xmm0 = _mm_castsi128_ps( xmm0i );
        a = _mm_and_ps( a, xmm0 );
        return _mm_sqrt_ps( Dot( a, a ) );
    }

    inline __m128 Length2( __m128 a )
    {
        __m128 xmm0;
        __m128i xmm0i;
        xmm0i = _mm_set_epi32( 0, 0, 0xFFFFFFFF, 0xFFFFFFFF );
        xmm0 = _mm_castsi128_ps( xmm0i );
        a = _mm_and_ps( a, xmm0 );
        xmm0 = _mm_mul_ps( a, a );
        xmm0 = _mm_shuffle_ps( xmm0, xmm0, _MM_SHUFFLE( 3, 2, 0, 1 ) );
        xmm0 = _mm_add_ps( xmm0, xmm0 );
        xmm0 = _mm_movelh_ps( xmm0, xmm0 );
        return _mm_sqrt_ps( xmm0 );
    }

    inline __m128 Length1( __m128 a )
    {
        return _mm_shuffle_ps( a, a, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    }

    inline __m128 Normalize4( __m128 a )
    {
        return _mm_div_ps( a, Length4( a ) );
    }

    inline __m128 Normalize3( __m128 a )
    {
        __m128 xmm0;
        __m128i xmm0i;
        xmm0i = _mm_set_epi32( 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );
        xmm0 = _mm_castsi128_ps( xmm0i );
        a = _mm_and_ps( a, xmm0 );
        xmm0 = _mm_sqrt_ps( Dot( a, a ) );
        return _mm_div_ps( a, xmm0 );
    }

    inline __m128 Normalize2( __m128 a )
    {
        __m128 xmm0;
        __m128i xmm0i;
        xmm0i = _mm_set_epi32( 0, 0, 0xFFFFFFFF, 0xFFFFFFFF );
        xmm0 = _mm_castsi128_ps( xmm0i );
        a = _mm_and_ps( a, xmm0 );
        xmm0 = _mm_mul_ps( a, a );
        xmm0 = _mm_shuffle_ps( xmm0, xmm0, _MM_SHUFFLE( 3, 2, 0, 1 ) );
        xmm0 = _mm_add_ps( xmm0, xmm0 );
        xmm0 = _mm_movelh_ps( xmm0, xmm0 );
        xmm0 = _mm_sqrt_ps( xmm0 );
        return _mm_div_ps( a, xmm0 );
    }

    inline __m128 Normalize1( __m128 )
    {
        return _mm_set_ps( 0, 0, 0, 1 );
    }

    inline __m128 Abs( __m128 a )
    {
        __m128i xmm0i, xmm1i;
        xmm0i = _mm_castps_si128( a );
        xmm1i = _mm_set1_epi32( 0x80000000 );
        xmm0i = _mm_and_si128( xmm0i, xmm1i );
        return _mm_castsi128_ps( xmm0i );
    }

    inline __m128 Cross( __m128 a, __m128 b )
    {
        __m128 xmm0, xmm1, xmm2, xmm3;
        xmm0 = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 3, 0, 2, 1 ) );
        xmm1 = _mm_shuffle_ps( b, b, _MM_SHUFFLE( 3, 1, 0, 2 ) );
        xmm2 = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 3, 1, 0, 2 ) );
        xmm3 = _mm_shuffle_ps( b, b, _MM_SHUFFLE( 3, 0, 2, 1 ) );
        xmm0 = _mm_mul_ps( xmm0, xmm1 );
        xmm1 = _mm_mul_ps( xmm2, xmm3 );
        return _mm_sub_ps( xmm0, xmm1 );
    }
}
