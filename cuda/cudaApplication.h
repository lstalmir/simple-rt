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

                    m_Scene.Private.CameraDeviceMemory.Update();
                }
            }

            inline virtual int Run() override
            {
                const int X = m_CommandLineArguments.appWidth;
                const int Y = m_CommandLineArguments.appHeight;

                auto primaryRays_raw = m_Scene.Cameras[0].SpawnPrimaryRays( X, Y );
                const int primaryRayCount = (int)primaryRays_raw.Size();

                Array<SecondaryRayData> primaryRays( primaryRayCount );

                #if RT_ENABLE_ANTIALIASING
                const int imageDataSize = 3 * primaryRayCount / 4;
                #else
                const int imageDataSize = 3 * primaryRayCount;
                #endif

                std::vector<png_byte> pDstImageData( imageDataSize );
                Array<png_byte> imageData( imageDataSize );

                // Constant in the runtime
                const int objectCount = (int)m_Scene.Objects.size();

                primaryRays_raw.Sync();

                BenchmarkBegin();

                // Spawn primary ray tasks
                for( int ray = 0; ray < primaryRayCount; ++ray )
                {
                    auto rayData = SecondaryRay( primaryRays, ray );
                    rayData.Memory.Host().Ray = primaryRays_raw.Host( ray );
                    rayData.Memory.Host().PrimaryRayIndex = ray;
                    rayData.Memory.Host().PreviousRayIndex = -1;
                    rayData.Memory.Host().Depth = 0;
                    rayData.Memory.Host().Intersection.Distance = vec4( std::numeric_limits<RT::float_t>::infinity() );
                    rayData.Memory.Host().Type = SecondaryRayType::ePrimary;
                    rayData.Memory.Host().Mutex = 0;
                }

                primaryRays.Update();

                // Array of rays to process
                Array<SecondaryRayData> secondaryRays = primaryRays;

                while( secondaryRays.Size() > 0 )
                {
                    const int numIntersections = ComputeIntersections( secondaryRays );

                    // Update primary rays
                    ProcessLightIntersections( primaryRays, secondaryRays );

                    // Create new ray set from computed intersections
                    secondaryRays = SpawnSecondaryRays( secondaryRays, numIntersections );
                }

                FinalizePrimaryRays( primaryRays, imageData );

                imageData.Sync();
                imageData.HostCopyTo( pDstImageData.data() );

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

            int ComputeIntersections( Array<SecondaryRayData> rays );

            // These arrays must be preserved until the device is synced
            Array<RayData> m_ShadowRays;
            Array<int> m_NumShadowIntersections;
            void ProcessLightIntersections( Array<SecondaryRayData> primaryRays, Array<SecondaryRayData> rays );

            Array<SecondaryRayData> SpawnSecondaryRays( Array<SecondaryRayData> rays, int numIntersections );

            void FinalizePrimaryRays( Array<SecondaryRayData> primaryRays, Array<png_byte> imageData );
        };
    }
}
