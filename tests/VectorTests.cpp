#include <gtest/gtest.h>
#include "../Vector.h"

/***************************************************************************************\

Test:
    Normalize4

Description:
    Normalize 4-component vector taking into account all 4 components.
    Function should return normalized (1-length) 4-component vector.

\***************************************************************************************/
TEST( VectorTests, Normalize4 )
{
    RT::Vector4 vec4;
    vec4.x = 10;
    vec4.y = 10;
    vec4.z = 10;
    vec4.w = 10;

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::normalize4( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_FLOAT_EQ( vec4.x, 0.5f );
    EXPECT_FLOAT_EQ( vec4.y, 0.5f );
    EXPECT_FLOAT_EQ( vec4.z, 0.5f );
    EXPECT_FLOAT_EQ( vec4.w, 0.5f );
}

/***************************************************************************************\

Test:
    Normalize4

Description:
    Normalize 4-component vector taking into account first 3 components.
    Function should return normalized (1-length) 3-component vector with W zeroed.

\***************************************************************************************/
TEST( VectorTests, Normalize3 )
{
    RT::Vector4 vec4;
    vec4.x = 10;
    vec4.y = 10;
    vec4.z = 10;
    vec4.w = 10;

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::normalize3( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_FLOAT_EQ( vec4.x, 0.5773500f );
    EXPECT_FLOAT_EQ( vec4.y, 0.5773500f );
    EXPECT_FLOAT_EQ( vec4.z, 0.5773500f );
    EXPECT_FLOAT_EQ( vec4.w, 0.0f );
}

/***************************************************************************************\

Test:
    Normalize4

Description:
    Normalize 4-component vector taking into account first 2 components.
    Function should return normalized (1-length) 2-component vector with Z and W zeroed.

\***************************************************************************************/
TEST( VectorTests, Normalize2 )
{
    RT::Vector4 vec4;
    vec4.x = 10;
    vec4.y = 10;
    vec4.z = 10;
    vec4.w = 10;

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::normalize2( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_FLOAT_EQ( vec4.x, 0.7071067f );
    EXPECT_FLOAT_EQ( vec4.y, 0.7071067f );
    EXPECT_FLOAT_EQ( vec4.z, 0.0f );
    EXPECT_FLOAT_EQ( vec4.w, 0.0f );
}

/***************************************************************************************\

Test:
    Normalize4

Description:
    Normalize 4-component vector taking into account only first component.
    The function should always return (1,0,0,0).

\***************************************************************************************/
TEST( VectorTests, Normalize1 )
{
    RT::Vector4 vec4;
    vec4.x = 10;
    vec4.y = 10;
    vec4.z = 10;
    vec4.w = 10;

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::normalize1( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_FLOAT_EQ( vec4.x, 1.0f );
    EXPECT_FLOAT_EQ( vec4.y, 0.0f );
    EXPECT_FLOAT_EQ( vec4.z, 0.0f );
    EXPECT_FLOAT_EQ( vec4.w, 0.0f );
}
