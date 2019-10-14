#include <gtest/gtest.h>
#include "../ompIntersect.h"
#include "../ompIntrin.h"
#include "../ompRay.h"
#include "../ompReflect.h"
#include "../ompPlane.h"

/***************************************************************************************\

Test:
    Reflect

Description:
    Get reflected ray.

\***************************************************************************************/
TEST( ompReflectTests, Reflect )
{
    RT::Ray ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4 intersectionPoint = RT::Intersect( ray, plane );
    RT::Ray reflectedRay = RT::Reflect( ray, plane, intersectionPoint );

    EXPECT_NEAR( reflectedRay.Origin.x, intersectionPoint.x, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.y, intersectionPoint.y, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.z, intersectionPoint.z, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.w, intersectionPoint.w, 0.01f );

    EXPECT_NEAR( reflectedRay.Direction.x, 1.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.y, 1.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.z, 0.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.w, 0.0f, 0.01f );
}
