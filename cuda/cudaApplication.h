#pragma once
#include "../Optimizations.h"
#include "../Application.h"
#include "../File.h"
#include "cudaScene.h"
#include <functional>
#include <iostream>
#include <queue>
#include <vector>

namespace RT
{
    namespace CUDA
    {
        class Application : public RT::Application
        {
            using SceneTypes = RT::CUDA::SceneTypes;
            using SceneTraits = RT::MakeSceneTraits<SceneTypes, RT::CUDA::SceneFunctions<SceneTypes>>;

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

                    for( int camera = 0; camera < m_Scene.Cameras.size(); ++camera )
                    {
                        m_Scene.Private.CameraDeviceMemory.Host( camera ).AspectRatio = aspect;
                    }
                }
            }

            inline virtual int Run() override
            {
                const int X = m_CommandLineArguments.appWidth;
                const int Y = m_CommandLineArguments.appHeight;

                const auto primaryRays = m_Scene.Cameras[0].SpawnPrimaryRays( X, Y );
                const int primaryRayCount = (int)primaryRays.Size();

                Array<SecondaryRayData> primaryRaysDeviceMemory( primaryRayCount );

                m_Rays.resize( primaryRayCount );

                #if RT_ENABLE_ANTIALIASING
                const int imageDataSize = primaryRayCount / 4;
                #else
                const int imageDataSize = primaryRayCount;
                #endif

                struct char3 { png_byte r, g, b; };
                std::vector<char3> pDstImageData( imageDataSize );

                memset( pDstImageData.data(), 0, pDstImageData.size() * sizeof( char3 ) );

                // Constant in the runtime
                const int objectCount = (int)m_Scene.Objects.size();

                BenchmarkBegin();

                // Spawn primary ray tasks
                for( int ray = 0; ray < primaryRayCount; ++ray )
                {
                    auto rayData = SecondaryRay( primaryRaysDeviceMemory, ray );
                    rayData.Memory.Host().Ray = primaryRays.Host( ray );
                    rayData.Memory.Host().PrimaryRayIndex = ray;
                    rayData.Memory.Host().PreviousRayIndex = -1;
                    rayData.Memory.Host().Depth = 0;
                    rayData.Memory.Host().Intersection.Distance = vec4( std::numeric_limits<RT::float_t>::infinity() );
                    rayData.Memory.Host().Type = SecondaryRayType::ePrimary;

                    m_Tasks.push( std::bind( &Application::ObjectIntersection, this, ray ) );
                }

                primaryRaysDeviceMemory.Update();

                // Execute tasks
                std::function<void()> task;

                while( !m_Tasks.empty() )
                {
                    m_Tasks.front()(); // invoke
                    m_Tasks.pop();
                }

                // Merge results
                for( int i = 0; i < primaryRayCount; ++i )
                {
                    const auto& ray = m_Rays[i];
                    //pDstImageData[i].r = (png_byte)std::min( 255.0f, ray.m_Intersection.m_Color.x );
                    //pDstImageData[i].g = (png_byte)std::min( 255.0f, ray.m_Intersection.m_Color.y );
                    //pDstImageData[i].b = (png_byte)std::min( 255.0f, ray.m_Intersection.m_Color.z );
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

            std::vector<SecondaryRay> m_Rays;
            std::queue<std::function<void()>> m_Tasks;

            void ObjectIntersection( Array<RayData> rays );

        };
    }
}
