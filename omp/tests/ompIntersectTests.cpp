#include <gtest/gtest.h>
#include "../../Intrin.h"
#include "../ompRay.h"
#include "../ompRay2.h"
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
    RT::OMP::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );
    
    RT::OMP::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4 intersectionPoint = ray.Intersect( plane );

    EXPECT_NEAR( intersectionPoint.x * ray.Direction.x + ray.Origin.x, 1.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.y * ray.Direction.y + ray.Origin.y, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.z * ray.Direction.z + ray.Origin.z, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.w * ray.Direction.w + ray.Origin.w, 0.0f, 0.01f );
}

TEST( ompIntersectTests, Intersect_Plane_2x2 )
{
    RT::OMP::Ray2x2 ray;
    ray.Origin[ 0 ] = RT::vec4( 0, 1, 0 );
    ray.Direction[ 0 ] = RT::vec4( 1, -1, 0 );
    ray.Origin[ 1 ] = RT::vec4( 0, 1, 0 );
    ray.Direction[ 1 ] = RT::vec4( 1, 0, 0 );
    ray.Origin[ 2 ] = RT::vec4( 0, 1, 0 );
    ray.Direction[ 2 ] = RT::vec4( 0, 1, 0 );
    ray.Origin[ 3 ] = RT::vec4( 0, 1, 0 );
    ray.Direction[ 3 ] = RT::vec4( 0, 0, 1 );

    RT::OMP::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4_2x2 intersectionPoints = ray.Intersect( plane );

    EXPECT_NEAR( intersectionPoints.m[ 0 ].x * ray.Direction[ 0 ].x + ray.Origin[ 0 ].x, 1.0f, 0.01f );
    EXPECT_NEAR( intersectionPoints.m[ 0 ].y * ray.Direction[ 0 ].y + ray.Origin[ 0 ].y, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoints.m[ 0 ].z * ray.Direction[ 0 ].z + ray.Origin[ 0 ].z, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoints.m[ 0 ].w * ray.Direction[ 0 ].w + ray.Origin[ 0 ].w, 0.0f, 0.01f );

    EXPECT_EQ( intersectionPoints.m[ 1 ].x, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 1 ].y, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 1 ].z, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 1 ].w, INFINITY );

    EXPECT_EQ( intersectionPoints.m[ 2 ].x, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 2 ].y, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 2 ].z, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 2 ].w, INFINITY );

    EXPECT_EQ( intersectionPoints.m[ 3 ].x, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 3 ].y, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 3 ].z, INFINITY );
    EXPECT_EQ( intersectionPoints.m[ 3 ].w, INFINITY );
}

/***************************************************************************************\

Test:
    Intersect_Plane_Parallel

Description:
    Get intersection point of parallel ray and plane.

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Plane_Parallel )
{
    RT::OMP::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, 0, 0 );

    RT::OMP::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4 intersectionPoint = ray.Intersect( plane );

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
    RT::OMP::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, 0, 0 );

    RT::OMP::Plane plane;
    plane.Origin = RT::vec4( -1, 0, 0 );
    plane.Normal = RT::vec4( 1, 0, 0 );

    RT::vec4 intersectionPoint = ray.Intersect( plane );

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
    RT::OMP::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::OMP::Triangle tri;
    tri.A = RT::vec4( 1, -2, 2 );
    tri.B = RT::vec4( 1, -2, -2 );
    tri.C = RT::vec4( 1, 2, -2 );

    RT::vec4 intersectionPoint = ray.Intersect( tri );

    EXPECT_NEAR( intersectionPoint.x * ray.Direction.x + ray.Origin.x, 1.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.y * ray.Direction.y + ray.Origin.y, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.z * ray.Direction.z + ray.Origin.z, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.w * ray.Direction.w + ray.Origin.w, 0.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Intersect_Triangle_Parallel

Description:
    Get intersection point of parallel ray and triangle.

\***************************************************************************************/
TEST( ompIntersectTests, Intersect_Triangle_Parallel )
{
    RT::OMP::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 0, 1, 0 );

    RT::OMP::Triangle tri;
    tri.A = RT::vec4( 1, -2, 2 );
    tri.B = RT::vec4( 1, -2, -2 );
    tri.C = RT::vec4( 1, 2, -2 );

    RT::vec4 intersectionPoint = ray.Intersect( tri );

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
    RT::OMP::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::OMP::Triangle tri;
    tri.A = RT::vec4( 100, -2, 2 );
    tri.B = RT::vec4( 100, -2, -2 );
    tri.C = RT::vec4( 100, 2, -2 );

    RT::vec4 intersectionPoint = ray.Intersect( tri );

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
    RT::OMP::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( -1, 1, 0 );

    RT::OMP::Triangle tri;
    tri.A = RT::vec4( 1, -2, 2 );
    tri.B = RT::vec4( 1, -2, -2 );
    tri.C = RT::vec4( 1, 2, -2 );

    RT::vec4 intersectionPoint = ray.Intersect( tri );

    EXPECT_FLOAT_EQ( intersectionPoint.x, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.y, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.z, std::numeric_limits<RT::float_t>::infinity() );
    EXPECT_FLOAT_EQ( intersectionPoint.w, std::numeric_limits<RT::float_t>::infinity() );
}
