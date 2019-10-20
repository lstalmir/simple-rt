#include "ApplicationOMP.h"
#include "File.h"
#include <iostream>
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

    if( m_CommandLineArguments.appAdjustAspect != -1.0f )
    {
        RT::float_t aspect = m_CommandLineArguments.appAdjustAspect;

        if( aspect == 0.0f )
        {
            aspect = static_cast<RT::float_t>(m_CommandLineArguments.appWidth) /
                static_cast<RT::float_t>(m_CommandLineArguments.appHeight);
        }

        for( auto& camera : m_Scene.Cameras )
        {
            camera.AspectRatio = aspect;
        }
    }
}

/***************************************************************************************\

Function:
    ApplicationOMP::Run

Description:
    Runs OpenMP implementation of the application.

\***************************************************************************************/
int RT::ApplicationOMP::Run()
{
    const int X = m_CommandLineArguments.appWidth;
    const int Y = m_CommandLineArguments.appHeight;

    const auto primaryRays = m_Scene.Cameras[0].SpawnPrimaryRays( X, Y );
    const size_t primaryRayCount = primaryRays.size();

    std::vector<RT::vec4> intersections( X * Y );

    struct char3 { png_byte r, g, b; };
    std::vector<char3> pDstImageData( X * Y );

    BenchmarkBegin();

#   pragma omp parallel
    for( size_t i = 0; i < primaryRayCount; ++i )
    {
        intersections[i] = vec4( std::numeric_limits<RT::float_t>::infinity() );

        for( auto object : m_Scene.Objects )
        {
            if( m_CommandLineArguments.appDisableBoundingBoxes || primaryRays[i].Intersect( object.BoundingBox ) )
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

        if( intersections[i].x == std::numeric_limits<RT::float_t>::infinity() )
        {
            pDstImageData[i].r = 0;
            pDstImageData[i].g = 0;
            pDstImageData[i].b = 0;
        }
        else
        {
            pDstImageData[i].r = 255;
            pDstImageData[i].g = 255;
            pDstImageData[i].b = 255;
        }
    }

    BenchmarkEnd();

    RT::File out( m_CommandLineArguments.appOutputFilename.c_str(), "wb" );
    RT::PngWriter png( out );

    png.set_width( X );
    png.set_height( Y );
    png.set_bit_depth( 8 );
    png.set_color_type( PNG_COLOR_TYPE_RGB );
    png.write_data( reinterpret_cast<png_bytep>(pDstImageData.data()) );

    ReportBenchmarkTime( std::cout );

    return 0;
}
