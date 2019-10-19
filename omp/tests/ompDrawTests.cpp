#include <gtest/gtest.h>
#include "../ompCamera.h"
#include "../ompIntersect.h"
#include "../ompRay.h"
#include "../../File.h"

/***************************************************************************************\

Test:
    Intersect_Plane

Description:
    Get intersection point of ray and plane.

\***************************************************************************************/
TEST( ompDrawTests, Draw_Test )
{
    constexpr int horizontalRays = 1000;
    constexpr int verticalRays = 1000;

    RT::Camera camera;
    camera.Direction = RT::vec4( 1, 0, 0 );
    camera.Origin = RT::vec4( 0, 0, 0 );
    camera.Up = RT::vec4( 0, 1, 0 );
    camera.AspectRatio = 1;
    camera.FocalLength = 1;
    camera.HorizontalFOV = RT::Radians( 75 );

    std::vector<RT::Ray> primaryRays( horizontalRays * verticalRays );
    RT::SpawnPrimaryRays( camera, horizontalRays, verticalRays, primaryRays.data() );

    RT::Triangle tri;
    tri.A = RT::vec4( 5, 0, 2 );
    tri.B = RT::vec4( 5, 0, -2 );
    tri.C = RT::vec4( 5, 2, 0 );

    RT::File out( "draw-test.png", "wb" );
    RT::PngWriter png( out );

    png.set_width( horizontalRays );
    png.set_height( verticalRays );
    png.set_bit_depth( 8 );
    png.set_color_type( PNG_COLOR_TYPE_RGB );

    struct char3 { png_byte r, g, b; };

    std::vector<char3> pDstImageData( horizontalRays * verticalRays );

    for( unsigned y = 0; y < verticalRays; ++y )
    {
        for( unsigned x = 0; x < horizontalRays; ++x )
        {
            RT::Ray ray = primaryRays[y * horizontalRays + x];
            RT::vec4 intersection = RT::Intersect( ray, tri );

            char3* dstPixel = &pDstImageData[x + y * horizontalRays];

            if( intersection.x != std::numeric_limits<RT::float_t>::infinity() )
            {
                dstPixel->r = 0;
                dstPixel->g = 0;
                dstPixel->b = 0;
            }
            else
            {
                dstPixel->r = 255;
                dstPixel->g = 255;
                dstPixel->b = 255;
            }
        }
    }

    png.write_data( reinterpret_cast<png_bytep>(pDstImageData.data()) );
}
