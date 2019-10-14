#include <gtest/gtest.h>
#include "../Ray.h"
#include "../SSE.h"

/***************************************************************************************\

Test:
    Intersect_Plane

Description:
    Get intersection point of ray and plane.

\***************************************************************************************/
TEST( ompRayTests, IntersectPlane )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4 intersectionPoint = ray.Intersect( plane );

    EXPECT_NEAR( intersectionPoint.x, 1.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.y, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.z, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.w, 0.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Intersect_Plane_NoIntersection

Description:
    Get intersection point of parallel ray and plane.

\***************************************************************************************/
TEST( ompRayTests, Intersect_Plane_NoIntersection )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, 0, 0 );

    RT::Plane plane;
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
    Intersect_Plane_PlaneBehindRay

Description:
    Get intersection point of parallel ray and plane.

\***************************************************************************************/
TEST( ompRayTests, Intersect_Plane_PlaneBehindRay )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, 0, 0 );

    RT::Plane plane;
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
    Intersect_Plane

Description:
    Get intersection point of ray and plane.

\***************************************************************************************/
TEST( ompRayTests, IntersectTriangle )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::Triangle tri;
    tri.A = RT::vec4( 1, -2, 2 );
    tri.B = RT::vec4( 1, -2, -2 );
    tri.C = RT::vec4( 1, 2, -2 );

    RT::vec4 intersectionPoint = ray.Intersect( tri );

    EXPECT_NEAR( intersectionPoint.x, 1.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.y, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.z, 0.0f, 0.01f );
    EXPECT_NEAR( intersectionPoint.w, 0.0f, 0.01f );
}

/***************************************************************************************\

Test:
    Reflect

Description:
    Get reflected ray.

\***************************************************************************************/
TEST( ompRayTests, Reflect )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4 intersectionPoint = ray.Intersect( plane );
    RT::Ray reflectedRay = ray.Reflect( plane, intersectionPoint );

    EXPECT_NEAR( reflectedRay.Origin.x, intersectionPoint.x, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.y, intersectionPoint.y, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.z, intersectionPoint.z, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.w, intersectionPoint.w, 0.01f );

    EXPECT_NEAR( reflectedRay.Direction.x, 1.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.y, 1.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.z, 0.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.w, 0.0f, 0.01f );
}
