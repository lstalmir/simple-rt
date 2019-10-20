#include "ApplicationOMP.h"
#include "File.h"
#include <atomic>
#include <iostream>
#include <omp.h>
#include <tbb/concurrent_vector.h>

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
    srand( 1 );

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
    const int primaryRayCount = primaryRays.size();

    std::vector<RayTriangleIntersection> intersections( primaryRays.size() );

    struct char3 { png_byte r, g, b; };
    std::vector<char3> pDstImageData( primaryRays.size() );

    BenchmarkBegin();

#   pragma omp parallel for
    for( int i = 0; i < primaryRayCount; ++i )
    {
        intersections[i].pTriangle = nullptr;
        intersections[i].ColorDistance = vec4( std::numeric_limits<RT::float_t>::infinity() );

        for( auto& object : m_Scene.Objects )
        {
            if( m_CommandLineArguments.appDisableBoundingBoxes || primaryRays[i].Intersect( object.BoundingBox ) )
            {
                for( auto& triangle : object.Triangles )
                {
                    RT::vec4 intersection = primaryRays[i].Intersect( triangle );

                    if( intersection.w != std::numeric_limits<RT::float_t>::infinity() )
                    {
                        if( intersection.w < intersections[i].ColorDistance.w )
                        {
                            intersections[i].pTriangle = &triangle;

                            intersections[i].ColorDistance.w = intersection.w;

                            intersections[i].ColorDistance.x = object.Color.x;
                            intersections[i].ColorDistance.y = object.Color.y;
                            intersections[i].ColorDistance.z = object.Color.z;
                        }
                    }
                }
            }
        }

        if( intersections[i].ColorDistance.w == std::numeric_limits<RT::float_t>::infinity() )
        {
            pDstImageData[i].r = 0;
            pDstImageData[i].g = 0;
            pDstImageData[i].b = 0;
        }
        else
        {
            // Ray from intersection to light
            float intensity = 0.5;
            __m128 DIST, O, D, LO, INTENSITY, BIAS;
            DIST = _mm_set1_ps( intersections[i].ColorDistance.w );
            BIAS = _mm_set1_ps( 0.05f );
            DIST = _mm_sub_ps( DIST, BIAS );
            O = _mm_load_ps( &primaryRays[i].Origin.data );
            D = _mm_load_ps( &primaryRays[i].Direction.data );
            O = _mm_fmadd_ps( DIST, D, O );
            LO = _mm_load_ps( &m_Scene.Lights[0].Position.data );
            D = _mm_sub_ps( LO, O );
            INTENSITY = _mm_mul_ps( D, D );
            INTENSITY = _mm_rcp_ps( INTENSITY );
            D = Normalize3( D );

            RT::OMP::Ray secondaryRay;
            _mm_store_ps( &secondaryRay.Origin.data, O );
            _mm_store_ps( &secondaryRay.Direction.data, D );

            bool inTheShadows = false;

            for( auto& object : m_Scene.Objects )
            {
                if( m_CommandLineArguments.appDisableBoundingBoxes || secondaryRay.Intersect( object.BoundingBox ) )
                {
                    for( auto& triangle : object.Triangles )
                    {
                        RT::vec4 intersection = secondaryRay.Intersect( triangle );

                        if( intersection.w != std::numeric_limits<RT::float_t>::infinity() )
                        {
                            inTheShadows = true;
                            break;
                        }
                    }
                }

                if( inTheShadows )
                {
                    break;
                }
            }

            if( !inTheShadows )
            {
                intensity *= 2;
            }

            pDstImageData[i].r = std::min( 255.0f, intersections[i].ColorDistance.x * intensity );
            pDstImageData[i].g = std::min( 255.0f, intersections[i].ColorDistance.y * intensity );
            pDstImageData[i].b = std::min( 255.0f, intersections[i].ColorDistance.z * intensity );
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
