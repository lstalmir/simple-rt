#include <gtest/gtest.h>
#include "../ompIntersect.h"
#include "../ompIntrin.h"
#include "../ompRay.h"
#include "../ompPlane.h"
#include "../ompTriangle.h"

/***************************************************************************************\

Test:
    Intersect_Plane

Description:
    Get intersection point of ray and plane.

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Plane )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );
    
    RT::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4 intersectionPoint = RT::Intersect( ray, plane );

    EXPECT_NEAR( intersectionPoint.x, 1.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.y, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.z, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.w, 0.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Intersect_Plane_Parallel

Description:
    Get intersection point of parallel ray and plane.

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Plane_Parallel )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, 0, 0 );

    RT::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4 intersectionPoint = RT::Intersect( ray, plane );

    EXPECT_FLOAT_EQ( intersectionPoint.x, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.y, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.z, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.w, std::numeric_limits<RT::float_t>::infinity() );
}

/***************************************************************************************\

Test:
    Intersect_Plane_Behind

Description:
    Get intersection point of parallel ray and plane.

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Plane_Behind )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, 0, 0 );

    RT::Plane plane;
    plane.Origin = RT::vec4( -1, 0, 0 );
    plane.Normal = RT::vec4( 1, 0, 0 );

    RT::vec4 intersectionPoint = RT::Intersect( ray, plane );

    EXPECT_FLOAT_EQ( intersectionPoint.x, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.y, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.z, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.w, std::numeric_limits<RT::float_t>::infinity() );
}

/***************************************************************************************\

Test:
    Intersect_Triangle

Description:
    Get intersection point of ray and triangle.

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Triangle )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::Triangle tri;
    tri.A = RT::vec4( 1, -2, 2 );
    tri.B = RT::vec4( 1, -2, -2 );
    tri.C = RT::vec4( 1, 2, -2 );

    RT::vec4 intersectionPoint = RT::Intersect( ray, tri );

    EXPECT_NEAR( intersectionPoint.x, 1.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.y, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.z, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.w, 0.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Intersect_Triangle_Parallel

Description:
    Get intersection point of parallel ray and triangle.

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Triangle_Parallel )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 0, 1, 0 );

    RT::Triangle tri;
    tri.A = RT::vec4( 1, -2, 2 );
    tri.B = RT::vec4( 1, -2, -2 );
    tri.C = RT::vec4( 1, 2, -2 );

    RT::vec4 intersectionPoint = RT::Intersect( ray, tri );

    EXPECT_FLOAT_EQ( intersectionPoint.x, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.y, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.z, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.w, std::numeric_limits<RT::float_t>::infinity() );
}

/***************************************************************************************\

Test:
    Intersect_Triangle_Miss

Description:

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Triangle_Miss )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::Triangle tri;
    tri.A = RT::vec4( 100, -2, 2 );
    tri.B = RT::vec4( 100, -2, -2 );
    tri.C = RT::vec4( 100, 2, -2 );

    RT::vec4 intersectionPoint = RT::Intersect( ray, tri );

    EXPECT_FLOAT_EQ( intersectionPoint.x, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.y, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.z, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.w, std::numeric_limits<RT::float_t>::infinity() );
}

/***************************************************************************************\

Test:
    Intersect_Triangle

Description:
    Get intersection point of ray and triangle.

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Triangle_Behind )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( -1, 1, 0 );

    RT::Triangle tri;
    tri.A = RT::vec4( 1, -2, 2 );
    tri.B = RT::vec4( 1, -2, -2 );
    tri.C = RT::vec4( 1, 2, -2 );

    RT::vec4 intersectionPoint = RT::Intersect( ray, tri );

    EXPECT_FLOAT_EQ( intersectionPoint.x, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.y, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.z, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.w, std::numeric_limits<RT::float_t>::infinity() );
}
