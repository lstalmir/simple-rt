#include <gtest/gtest.h>
#include "../Vector.h"
#include "../SSE.h"

/***************************************************************************************\

Test:
    Length4

Description:
    Get length of 4-component vector taking into account all 4 components.

\***************************************************************************************/
TEST( ompVectorTests, Length4 )
{
    RT::vec4 vec4( 10 );

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::xmm::length4( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_NEAR( vec4.x, 20.0f, 0.01f );
    EXPECT_NEAR( vec4.y, 20.0f, 0.01f );
    EXPECT_NEAR( vec4.z, 20.0f, 0.01f );
    EXPECT_NEAR( vec4.w, 20.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Length3

Description:
    Get length of 4-component vector taking into account first 3 components.

\***************************************************************************************/
TEST( ompVectorTests, Length3 )
{
    RT::vec4 vec4( 10 );

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::xmm::length3( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_NEAR( vec4.x, 17.32f, 0.01f );
    EXPECT_NEAR( vec4.y, 17.32f, 0.01f );
    EXPECT_NEAR( vec4.z, 17.32f, 0.01f );
    EXPECT_NEAR( vec4.w, 17.32f, 0.01f );
}

/***************************************************************************************\

Test:
    Length2

Description:
    Get length of 4-component vector taking into account first 2 components.

\***************************************************************************************/
TEST( ompVectorTests, Length2 )
{
    RT::vec4 vec4( 10 );

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::xmm::length2( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_NEAR( vec4.x, 14.14f, 0.01f );
    EXPECT_NEAR( vec4.y, 14.14f, 0.01f );
    EXPECT_NEAR( vec4.z, 14.14f, 0.01f );
    EXPECT_NEAR( vec4.w, 14.14f, 0.01f );
}

/***************************************************************************************\

Test:
    Length1

Description:
    Get length of 4-component vector taking into account only first component.

\***************************************************************************************/
TEST( ompVectorTests, Length1 )
{
    RT::vec4 vec4( 10 );

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::xmm::length1( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_NEAR( vec4.x, 10.0f, 0.01f );
    EXPECT_NEAR( vec4.y, 10.0f, 0.01f );
    EXPECT_NEAR( vec4.z, 10.0f, 0.01f );
    EXPECT_NEAR( vec4.w, 10.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Normalize4

Description:
    Normalize 4-component vector taking into account all 4 components.
    Function should return normalized (1-length) 4-component vector.

\***************************************************************************************/
TEST( ompVectorTests, Normalize4 )
{
    RT::vec4 vec4( 10 );

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::xmm::normalize4( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_NEAR( vec4.x, 0.5f, 0.01f );
    EXPECT_NEAR( vec4.y, 0.5f, 0.01f );
    EXPECT_NEAR( vec4.z, 0.5f, 0.01f );
    EXPECT_NEAR( vec4.w, 0.5f, 0.01f );
}

/***************************************************************************************\

Test:
    Normalize4

Description:
    Normalize 4-component vector taking into account first 3 components.
    Function should return normalized (1-length) 3-component vector with W zeroed.

\***************************************************************************************/
TEST( ompVectorTests, Normalize3 )
{
    RT::vec4 vec4( 10 );

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::xmm::normalize3( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_NEAR( vec4.x, 0.57f, 0.01f );
    EXPECT_NEAR( vec4.y, 0.57f, 0.01f );
    EXPECT_NEAR( vec4.z, 0.57f, 0.01f );
    EXPECT_NEAR( vec4.w, 0.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Normalize4

Description:
    Normalize 4-component vector taking into account first 2 components.
    Function should return normalized (1-length) 2-component vector with Z and W zeroed.

\***************************************************************************************/
TEST( ompVectorTests, Normalize2 )
{
    RT::vec4 vec4( 10 );

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::xmm::normalize2( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_NEAR( vec4.x, 0.71f, 0.01f );
    EXPECT_NEAR( vec4.y, 0.71f, 0.01f );
    EXPECT_NEAR( vec4.z, 0.0f, 0.01f );
    EXPECT_NEAR( vec4.w, 0.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Normalize4

Description:
    Normalize 4-component vector taking into account only first component.
    The function should always return (1,0,0,0).

\***************************************************************************************/
TEST( ompVectorTests, Normalize1 )
{
    RT::vec4 vec4( 10 );

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = RT::xmm::normalize1( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    EXPECT_NEAR( vec4.x, 1.0f, 0.01f );
    EXPECT_NEAR( vec4.y, 0.0f, 0.01f );
    EXPECT_NEAR( vec4.z, 0.0f, 0.01f );
    EXPECT_NEAR( vec4.w, 0.0f, 0.01f );
}
