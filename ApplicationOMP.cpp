#include "ApplicationOMP.h"
#include "File.h"
#include <omp.h>

/***************************************************************************************\

Function:
    ApplicationOMP::ApplicationOMP

Description:
    Constructor

\***************************************************************************************/
RT::ApplicationOMP::ApplicationOMP( const RT::CommandLineArguments& cmdargs )
    : Application( cmdargs )
    , m_Scene()
{
    m_Scene = m_SceneLoader.LoadScene<OMP::SceneTraits>( m_CommandLineArguments.appInputFilename );
}

/***************************************************************************************\

Function:
    ApplicationOMP::Run

Description:
    Runs OpenMP implementation of the application.

\***************************************************************************************/
int RT::ApplicationOMP::Run()
{
    const int X = 1000, Y = 1000;

    const auto primaryRays = m_Scene.Cameras[0].SpawnPrimaryRays( X, Y );
    const size_t primaryRayCount = primaryRays.size();

    std::vector<RT::vec4> intersections( X*Y );

    for( size_t i = 0; i < primaryRayCount; ++i )
    {
        intersections[i] = vec4( std::numeric_limits<RT::float_t>::infinity() );
    }

#   pragma omp parallel
    for( size_t i = 0; i < primaryRayCount; ++i )
    {
        for( auto object : m_Scene.Objects )
        {
            for( auto triangle : object.Triangles )
            {
                RT::vec4 intersection = primaryRays[i].Intersect( triangle );

                if( intersection.x != std::numeric_limits<RT::float_t>::infinity() )
                {
                    if( intersection.x < intersections[i].x )
                    {
                        intersections[i].x = intersection.x;
                    }
                }
            }
        }
    }

    RT::File out( m_CommandLineArguments.appOutputFilename.c_str(), "wb" );
    RT::PngWriter png( out );

    png.set_width( X );
    png.set_height( Y );
    png.set_bit_depth( 8 );
    png.set_color_type( PNG_COLOR_TYPE_RGB );

    struct char3 { png_byte r, g, b; };

    std::vector<char3> pDstImageData( X*Y );

    for( unsigned y = 0; y < Y; ++y )
    {
        for( unsigned x = 0; x < X; ++x )
        {
            const unsigned offset = y * X + x;

            if( intersections[offset].x != std::numeric_limits<RT::float_t>::infinity() )
            {
                pDstImageData[offset].r = 0;
                pDstImageData[offset].g = 0;
                pDstImageData[offset].b = 0;
            }
            else
            {
                pDstImageData[offset].r = 255;
                pDstImageData[offset].g = 255;
                pDstImageData[offset].b = 255;
            }
        }
    }

    png.write_data( reinterpret_cast<png_bytep>(pDstImageData.data()) );

    return 0;
}
