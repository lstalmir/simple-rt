#include <gtest/gtest.h>
#include "../../Intrin.h"
#include "../ompRay.h"
#include "../ompPlane.h"

/***************************************************************************************\

Test:
    Reflect

Description:
    Get reflected ray.

\***************************************************************************************/
TEST( ompReflectTests, Reflect )
{
    RT::OMP::Ray<true> ray;
    ray.Origin = RT::vec4( 0, 1, 0 );
    ray.Direction = RT::vec4( 1, -1, 0 );

    RT::OMP::Plane<true> plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    RT::vec4 intersectionPoint = ray.Intersect( plane );
    RT::OMP::Ray<true> reflectedRay = ray.Reflect( plane, intersectionPoint );

    EXPECT_NEAR( reflectedRay.Origin.x, intersectionPoint.x * ray.Direction.x + ray.Origin.x, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.y, intersectionPoint.y * ray.Direction.y + ray.Origin.y, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.z, intersectionPoint.z * ray.Direction.z + ray.Origin.z, 0.01f );
    EXPECT_NEAR( reflectedRay.Origin.w, intersectionPoint.w * ray.Direction.w + ray.Origin.w, 0.01f );

    EXPECT_NEAR( reflectedRay.Direction.x, 1.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.y, 1.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.z, 0.0f, 0.01f );
    EXPECT_NEAR( reflectedRay.Direction.w, 0.0f, 0.01f );
}
