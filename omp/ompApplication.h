#pragma once
#include "../Optimizations.h"
#include "../Application.h"
#include "../File.h"
#include "../Tbb.h"
#include "ompScene.h"
#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include <omp.h>

namespace RT::OMP
{
    class Application : public RT::Application
    {
        using SceneTypes = RT::OMP::SceneTypes;
        using SceneTraits = RT::MakeSceneTraits<SceneTypes, RT::OMP::SceneFunctions<SceneTypes>>;

        struct Intersection
        {
            float m_Distance;
            Triangle m_Triangle;
            vec4 m_Color;
        };

        struct SecondaryRay
        {
            Ray m_Ray;
            int m_PrimaryRayIndex;
            int m_PreviousRayIndex;
            int m_Depth;
            Intersection m_Intersection;
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
            const int primaryRayCount = (int)primaryRays.size();

            m_Rays.resize( primaryRayCount );

            struct char3 { png_byte r, g, b; };
            std::vector<char3> pDstImageData( primaryRayCount );

            memset( pDstImageData.data(), 0, pDstImageData.size() * sizeof( char3 ) );

            // Constant in the runtime
            const int objectCount = (int)m_Scene.Objects.size();

            BenchmarkBegin();

            #pragma omp parallel
            {
                // Spawn primary ray tasks
                #pragma omp for nowait
                for( int ray = 0; ray < primaryRayCount; ++ray )
                {
                    auto& rayData = m_Rays[ray];
                    rayData.m_Ray = primaryRays[ray];
                    rayData.m_PrimaryRayIndex = ray;
                    rayData.m_PreviousRayIndex = -1;
                    rayData.m_Depth = 0;
                    rayData.m_Intersection.m_Distance = std::numeric_limits<RT::float_t>::infinity();

                    m_Tasks.push( std::bind( &Application::ObjectIntersection, this, ray ) );
                }

                // Execute tasks
                std::function<void()> task;

                while( m_Tasks.try_pop( task ) )
                {
                    task();
                }

                #pragma omp barrier

                // Merge results
                #pragma omp for
                for( int i = 0; i < primaryRayCount; ++i )
                {
                    const auto& ray = m_Rays[i];

                    pDstImageData[i].r = (png_byte)std::min( 255.0f, ray.m_Intersection.m_Color.x );
                    pDstImageData[i].g = (png_byte)std::min( 255.0f, ray.m_Intersection.m_Color.y );
                    pDstImageData[i].b = (png_byte)std::min( 255.0f, ray.m_Intersection.m_Color.z );
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

        tbb::concurrent_vector<SecondaryRay> m_Rays;
        tbb::concurrent_queue<std::function<void()>> m_Tasks;

        void ObjectIntersection( int rayIndex )
        {
            auto& primaryRay = m_Rays[rayIndex];

            for( const auto& object : m_Scene.Objects )
            {
                // Ray-Object intersection, test bounding box and spawn per-triangle intersections
                if( RT_DISABLE_BOUNDING_BOXES || primaryRay.m_Ray.Intersect( object.BoundingBox ) )
                {
                    for( const auto& triangle : object.Triangles )
                    {
                        // Ray-Triangle intersection
                        RT::vec4 intersection = primaryRay.m_Ray.Intersect( triangle );

                        if( intersection.w < primaryRay.m_Intersection.m_Distance )
                        {
                            // Update intersection
                            primaryRay.m_Intersection.m_Triangle = triangle;
                            primaryRay.m_Intersection.m_Distance = intersection.w;
                            primaryRay.m_Intersection.m_Color = object.Color;
                        }
                    }
                }
            }

            // Generate secondary rays
            if( primaryRay.m_Intersection.m_Distance < std::numeric_limits<RT::float_t>::infinity() )
            {
                m_Tasks.push( std::bind( &Application::LightIntersection, this, rayIndex, 0 ) );
            }
        }

        void LightIntersection( int rayIndex, int lightIndex )
        {
            auto& primaryRay = m_Rays[rayIndex];
            const auto& light = m_Scene.Lights[lightIndex];

            float lightIntensity = 1.f;

            const auto secondaryRays = light.SpawnSecondaryRays(
                primaryRay.m_Ray,
                primaryRay.m_Intersection.m_Distance );

            for( const auto& secondaryRay : secondaryRays )
            {
                bool inTheShadows = false;

                for( const auto& object : m_Scene.Objects )
                {
                    if( RT_DISABLE_BOUNDING_BOXES || secondaryRay.Intersect( object.BoundingBox ) )
                    {
                        // Bounding-box test passed, check each triangle to check if the ray actually intersects the object
                        // Since we are in light ray pass, only one intersection is enough
                        for( const auto& triangle : object.Triangles )
                        {
                            RT::vec4 intersectionPoint = secondaryRay.Intersect( triangle );

                            if( intersectionPoint.w < std::numeric_limits<RT::float_t>::infinity() )
                            {
                                // The light ray hits other object
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
                    lightIntensity += 1;
                }
                else
                {
                    lightIntensity += 0.2f;
                }
            }

            lightIntensity /= light.Subdivs;

            #pragma omp atomic
            primaryRay.m_Intersection.m_Color.x *= lightIntensity;

            #pragma omp atomic
            primaryRay.m_Intersection.m_Color.y *= lightIntensity;

            #pragma omp atomic
            primaryRay.m_Intersection.m_Color.z *= lightIntensity;
        }
    };
}
