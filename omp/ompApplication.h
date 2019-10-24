#pragma once
#include "../Application.h"
#include "../File.h"
#include "ompScene.h"
#include <iostream>
#include <omp.h>

namespace RT::OMP
{
    template<RT::ApplicationIntrinMode IntrinMode>
    class Application : public RT::Application
    {
        using SceneTypes = RT::OMP::SceneTypes<IntrinMode == RT::ApplicationIntrinMode::eEnabled>;
        using SceneTraits = RT::MakeSceneTraits<SceneTypes, RT::OMP::SceneFunctions<SceneTypes>>;

        struct RayTriangleIntersection
        {
            const typename SceneTypes::ObjectType::TriangleType* pTriangle;
            RT::vec4 ColorDistance;
        };

    public:
        inline Application( const RT::CommandLineArguments& cmdargs )
            : RT::Application( cmdargs )
        {
            srand( 1 );

            m_Scene = m_SceneLoader.LoadScene<SceneTraits>( m_CommandLineArguments.appInputFilename );

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

        inline virtual int Run() override
        {
            const int X = m_CommandLineArguments.appWidth;
            const int Y = m_CommandLineArguments.appHeight;

            const auto primaryRays = m_Scene.Cameras[0].SpawnPrimaryRays( X, Y );
            const int primaryRayCount = primaryRays.size();

            std::vector<RayTriangleIntersection> intersections( primaryRays.size() );

            struct char3 { png_byte r, g, b; };
            std::vector<char3> pDstImageData( primaryRays.size() );

            BenchmarkBegin();

#           pragma omp parallel for
            for( int i = 0; i < primaryRayCount; ++i )
            {
                intersections[i].pTriangle = nullptr;
                intersections[i].ColorDistance = vec4( std::numeric_limits<RT::float_t>::infinity() );

                for( const auto& object : m_Scene.Objects )
                {
                    if( m_CommandLineArguments.appDisableBoundingBoxes || primaryRays[i].Intersect( object.BoundingBox ) )
                    {
                        for( const auto& triangle : object.Triangles )
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
                    float intensity = 1;

                    auto secondaryRays = m_Scene.Lights[0].SpawnSecondaryRays( primaryRays[i], intersections[i].ColorDistance.w );

                    for( int i = 0; i < secondaryRays.size(); ++i )
                    {
                        bool inTheShadows = false;

                        for( const auto& object : m_Scene.Objects )
                        {
                            if( m_CommandLineArguments.appDisableBoundingBoxes || secondaryRays[i].Intersect( object.BoundingBox ) )
                            {
                                for( const auto& triangle : object.Triangles )
                                {
                                    RT::vec4 intersection = secondaryRays[i].Intersect( triangle );

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
                            intensity += 1;
                        }
                        else
                        {
                            intensity += 0.2;
                        }
                    }

                    intensity /= static_cast<RT::float_t>(m_Scene.Lights[0].Subdivs);

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

    protected:
        RT::Scene<SceneTraits> m_Scene;
    };
}
