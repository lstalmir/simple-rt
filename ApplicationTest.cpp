#include "ApplicationTest.h"
#include "Plane.h"
#include "Ray.h"
#include "Vector.h"
#include <gtest/gtest.h>
#include <iostream>

/***************************************************************************************\

Function:
    ApplicationTest::ApplicationTest

Description:
    Constructor

\***************************************************************************************/
RT::ApplicationTest::ApplicationTest( const RT::CommandLineArguments& cmdargs )
    : Application( cmdargs )
{
    m_pTests.emplace( "Vector_Normalize4", &ApplicationTest::TEST_Vector_Normalize4 );
    m_pTests.emplace( "Vector_Normalize3", &ApplicationTest::TEST_Vector_Normalize3 );
    m_pTests.emplace( "Vector_Normalize2", &ApplicationTest::TEST_Vector_Normalize2 );
    m_pTests.emplace( "Ray_Intersect_Plane", &ApplicationTest::TEST_Ray_Intersect_Plane );
}

/***************************************************************************************\

Function:
    ApplicationTest::Run

Description:
    Run selected tests.

\***************************************************************************************/
int RT::ApplicationTest::Run()
{
    auto testit = m_pTests.find( m_CommandLineArguments.Test );
    if( testit != m_pTests.end() )
    {
        int result = (this->*(testit->second))();
        if( result == 0 )
        {
            std::cout << "TEST_" << m_CommandLineArguments.Test << ": PASSED\n";
        }
        return result;
    }
    std::cerr << "Unknown test name: " << m_CommandLineArguments.Test << "\n";
    return EINVAL;
}

/***************************************************************************************\

Function:
    ApplicationTest::TEST_Vector_Normalize4

Description:
    Test 4-component vector normalization.

\***************************************************************************************/
int RT::ApplicationTest::TEST_Vector_Normalize4()
{
    Vector4 vec4;
    vec4.x = 10;
    vec4.y = 10;
    vec4.z = 10;
    vec4.w = 10;

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = normalize4( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    if( (int)(vec4.x * 100000) != 50000 ||
        (int)(vec4.y * 100000) != 50000 ||
        (int)(vec4.z * 100000) != 50000 ||
        (int)(vec4.w * 100000) != 50000 )
    {
        std::cerr << __FUNCTION__ << ": FAILED (Invalid output vector ["
            << vec4.x << " "
            << vec4.y << " "
            << vec4.z << " "
            << vec4.w << "])\n";
        return 1;
    }

    return 0;
}

/***************************************************************************************\

Function:
    ApplicationTest::TEST_Vector_Normalize3

Description:
    Test 3-component vector normalization.

\***************************************************************************************/
int RT::ApplicationTest::TEST_Vector_Normalize3()
{
    Vector4 vec4;
    vec4.x = 10;
    vec4.y = 10;
    vec4.z = 10;
    vec4.w = 10;

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = normalize3( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    if( (int)(vec4.x * 100000) != 57735 ||
        (int)(vec4.y * 100000) != 57735 ||
        (int)(vec4.z * 100000) != 57735 ||
        vec4.w != 0 )
    {
        std::cerr << __FUNCTION__ << ": FAILED (Invalid output vector ["
            << vec4.x << " "
            << vec4.y << " "
            << vec4.z << " "
            << vec4.w << "])\n";
        return 1;
    }

    return 0;
}

/***************************************************************************************\

Function:
    ApplicationTest::TEST_Vector_Normalize2

Description:
    Test 2-component vector normalization.

\***************************************************************************************/
int RT::ApplicationTest::TEST_Vector_Normalize2()
{
    Vector4 vec4;
    vec4.x = 10;
    vec4.y = 10;
    vec4.z = 10;
    vec4.w = 10;

    __m128 xmm0;
    xmm0 = _mm_load_ps( &vec4.data );
    xmm0 = normalize2( xmm0 );
    _mm_store_ps( &vec4.data, xmm0 );

    if( (int)(vec4.x * 100000) != 70710 ||
        (int)(vec4.y * 100000) != 70710 ||
        vec4.z != 0 ||
        vec4.w != 0 )
    {
        std::cerr << __FUNCTION__ << ": FAILED (Invalid output vector ["
            << vec4.x << " "
            << vec4.y << " "
            << vec4.z << " "
            << vec4.w << "])\n";
        return 1;
    }

    return 0;
}

/***************************************************************************************\

Function:
    ApplicationTest::TEST_Ray_Intersect_Plane

Description:
    Test ray-plane intersection detection.

\***************************************************************************************/
int RT::ApplicationTest::TEST_Ray_Intersect_Plane()
{
    __m128 xmm0;

    Ray ray;
    ray.Direction.x = 1;
    ray.Direction.y = 1;
    ray.Direction.z = 1;

    xmm0 = _mm_load_ps( &ray.Direction.data );
    xmm0 = normalize3( xmm0 );
    _mm_store_ps( &ray.Direction.data, xmm0 );

    Plane plane;
    plane.Origin.x = 10;
    plane.Origin.y = 10;
    plane.Origin.z = 10;
    plane.Normal.x = -1;
    plane.Normal.y = 0;
    plane.Normal.z = 0;

    xmm0 = _mm_load_ps( &plane.Normal.data );
    xmm0 = normalize3( xmm0 );
    _mm_store_ps( &plane.Normal.data, xmm0 );

    if( !ray.Intersects( plane ) )
    {
        std::cerr << __FUNCTION__ << ": FAILED (Invalid ray-plane intersection result)\n";
        return 1;
    }

    return 0;
}
