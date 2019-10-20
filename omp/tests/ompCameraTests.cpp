#include <gtest/gtest.h>
#include "../ompCamera.h"
#include "../ompRay.h"
#include <fstream>

/***************************************************************************************\

Test:
    Intersect_Plane

Description:
    Get intersection point of ray and plane.

\***************************************************************************************/
TEST( ompCameraTests, SpawnPrimiaryRays_Test )
{
    constexpr int horizontalRays = 3;
    constexpr int verticalRays = 3;

    RT::OMP::Camera camera;
    camera.Direction = RT::vec4( 1, 0, 0 );
    camera.Origin = RT::vec4( 0, 0, 0 );
    camera.Up = RT::vec4( 0, 1, 0 );
    camera.AspectRatio = 1;
    camera.HorizontalFOV = RT::Radians( 75 );

    auto primaryRays = camera.SpawnPrimaryRays( horizontalRays, verticalRays );

    std::ofstream out( "primary-rays.dat" );
    for( int i = 0; i < horizontalRays * verticalRays; ++i )
    {
        out << primaryRays[i].Origin.x << " "
            << primaryRays[i].Origin.y << " "
            << primaryRays[i].Origin.z << " "
            << primaryRays[i].Direction.x << " "
            << primaryRays[i].Direction.y << " "
            << primaryRays[i].Direction.z << "\n";
    }
}
