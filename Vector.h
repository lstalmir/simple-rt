#pragma once
#include <immintrin.h>

namespace RT
{
#ifdef RT_ENABLE_DOUBLE_MATH
    typedef double float_t;
#else
    typedef float float_t;
#endif

    template<int>
    struct alignas(16) VectorN;

    template<>
    struct alignas(16) VectorN<4> { union { float_t data, x = 0; }; float_t y = 0, z = 0, w = 0; };
    static_assert( sizeof( VectorN<4> ) == 16 );
    typedef VectorN<4> Vector4;
    
    template<>
    struct alignas(16) VectorN<3> { union { float_t data, x = 0; }; float_t y = 0, z = 0; private: float_t _unused1 = 0; };
    static_assert( sizeof( VectorN<3> ) == 16 );
    typedef VectorN<3> Vector3;

    template<>
    struct alignas(16) VectorN<2> { union { float_t data, x = 0; }; float_t y = 0; private: float_t _unused1 = 0, _unused2 = 0; };
    static_assert( sizeof( VectorN<2> ) == 16 );
    typedef VectorN<2> Vector2;

    template<>
    struct alignas(16) VectorN<1> { union { float_t data, x = 0; }; private: float_t _unused1 = 0, _unused2 = 0, _unused3 = 0; };
    static_assert( sizeof( VectorN<1> ) == 16 );
    typedef VectorN<1> Vector1;

    inline __m128 dot( __m128 a, __m128 b )
    {
        __m128 xmm0, xmm1, xmm2;
        xmm0 = _mm_mul_ps( a, b );
        xmm1 = _mm_movehdup_ps( xmm0 );
        xmm2 = _mm_add_ps( xmm0, xmm1 );
        xmm1 = _mm_movehl_ps( xmm1, xmm2 );
        xmm2 = _mm_add_ps( xmm2, xmm1 );
        return _mm_shuffle_ps( xmm2, xmm2, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    }

    inline __m128 normalize4( __m128 a )
    {
        __m128 xmm0;
        xmm0 = dot( a, a );
        xmm0 = _mm_sqrt_ps( xmm0 );
        return _mm_div_ps( a, xmm0 );
    }

    inline __m128 normalize3( __m128 a )
    {
        __m128 xmm0, xmm1;
        __m128i xmm1i;
        xmm1i = _mm_set_epi32( 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF );
        xmm1 = _mm_castsi128_ps( xmm1i );
        a = _mm_and_ps( a, xmm1 );
        xmm0 = dot( a, a );
        xmm0 = _mm_sqrt_ps( xmm0 );
        return _mm_div_ps( a, xmm0 );
    }

    inline __m128 normalize2( __m128 a )
    {
        __m128 xmm0, xmm1;
        __m128i xmm1i;
        xmm1i = _mm_set_epi32( 0, 0, 0xFFFFFFFF, 0xFFFFFFFF );
        xmm1 = _mm_castsi128_ps( xmm1i );
        a = _mm_and_ps( a, xmm1 );
        xmm0 = dot( a, a );
        xmm0 = _mm_sqrt_ps( xmm0 );
        return _mm_div_ps( a, xmm0 );
    }

    inline __m128 normalize1( __m128 )
    {
        return _mm_set_ps( 0, 0, 0, 1 );
    }
}
