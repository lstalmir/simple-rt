#pragma once
#include "../Application.h"
#include "../File.h"
#include "ompScene.h"
#include <atomic>
#include <iostream>
#include <omp.h>
#include <tbb/concurrent_queue.h>

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
            RT::vec4 Color;
            float Distance;
            std::mutex Mutex;
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

            std::vector<RayTriangleIntersection> intersections( primaryRayCount );

            struct char3 { png_byte r, g, b; };
            std::vector<char3> pDstImageData( primaryRayCount );

            BenchmarkBegin();

#           pragma omp parallel
            {
                // Spawn tasks
#               pragma omp master
                {
                    for( int ray = 0; ray < primaryRayCount; ++ray )
                    {
                        intersections[ray].pTriangle = nullptr;
                        intersections[ray].Distance = std::numeric_limits<RT::float_t>::infinity();

                        for( int obj = 0; obj < m_Scene.Objects.size(); ++obj )
                        {
                            m_pTasks.push( new IntersectionTask(
                                this, nullptr,
                                &primaryRays[ray],
                                &m_Scene.Objects[obj],
                                &intersections[ray] ) );
                        }
                    }
                }

#               pragma omp barrier

                SlaveProc();

#               pragma omp barrier

#               pragma omp for
                for( int i = 0; i < primaryRayCount; ++i )
                {
                    /*
                    intersections[i].pTriangle = nullptr;
                    intersections[i].Distance = std::numeric_limits<RT::float_t>::infinity();

                    for( const auto& object : m_Scene.Objects )
                    {
                        if( m_CommandLineArguments.appDisableBoundingBoxes || primaryRays[i].Intersect( object.BoundingBox ) )
                        {
                            for( const auto& triangle : object.Triangles )
                            {
                                RT::vec4 intersection = primaryRays[i].Intersect( triangle );

                                if( intersection.w != std::numeric_limits<RT::float_t>::infinity() )
                                {
                                    if( intersection.w < intersections[i].Distance )
                                    {
                                        intersections[i].pTriangle = &triangle;
                                        intersections[i].Distance = intersection.w;
                                        intersections[i].Color = object.Color;
                                    }
                                }
                            }
                        }
                    }
                    */

                    if( intersections[i].Distance == std::numeric_limits<RT::float_t>::infinity() )
                    {
                        pDstImageData[i].r = 0;
                        pDstImageData[i].g = 0;
                        pDstImageData[i].b = 0;
                    }
                    else
                    {
                        // Ray from intersection to light
                        float intensity = 1;

                        auto secondaryRays = m_Scene.Lights[0].SpawnSecondaryRays( primaryRays[i], intersections[i].Distance );

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

                        pDstImageData[i].r = std::min( 255.0f, intersections[i].Color.x * intensity );
                        pDstImageData[i].g = std::min( 255.0f, intersections[i].Color.y * intensity );
                        pDstImageData[i].b = std::min( 255.0f, intersections[i].Color.z * intensity );
                    }
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

        struct TaskBatch
        {
            struct Task* pTasks;
            std::atomic_size_t numTasks;
            std::atomic_size_t numReady;
        };

        struct Task
        {
            TaskBatch* pBatch;
            Application* pApplication;

            inline Task(
                Application* pApplication_ = nullptr, 
                TaskBatch* pBatch_ = nullptr )
                : pBatch( pBatch_ )
                , pApplication( pApplication_ )
            {
            }

            virtual void Execute() = 0;
        };

        struct IntersectionTask : Task
        {
            const typename SceneTypes::CameraType::RayType* pRay;
            const typename SceneTypes::ObjectType* pObject;

            RayTriangleIntersection* pIntersection;

            inline IntersectionTask(
                Application* pApplication_ = nullptr,
                TaskBatch* pBatch_ = nullptr,
                const typename SceneTypes::CameraType::RayType* pRay_ = nullptr,
                const typename SceneTypes::ObjectType* pObject_ = nullptr,
                RayTriangleIntersection* pIntersection_ = nullptr )
                : Task( pApplication_, pBatch_ )
                , pRay( pRay_ )
                , pObject( pObject_ )
                , pIntersection( pIntersection_ )
            {
            }

            inline virtual void Execute() override final
            {
                // Ray-Object intersection, test bounding box and spawn per-triangle intersections
                if( pApplication->m_CommandLineArguments.appDisableBoundingBoxes ||
                    pRay->Intersect( pObject->BoundingBox ) )
                {
                    const size_t triangleCount = pObject->Triangles.size();

                    for( uint32_t i = 0; i < triangleCount; ++i )
                    {
                        // Ray-Triangle intersection
                        RT::vec4 intersection = pRay->Intersect( pObject->Triangles[i] );

                        std::unique_lock lk( pIntersection->Mutex );
                        if( intersection.w < pIntersection->Distance )
                        {
                            // Update intersection distance and color
                            pIntersection->Distance = intersection.w;
                            pIntersection->Color = pObject->Color;
                        }
                    }
                }
            }
        };

        tbb::concurrent_queue<Task*> m_pTasks;

        void MasterProc()
        {

        }

        void SlaveProc()
        {
            Task* pTask = nullptr;

            // Execute until there are no more intersections to test
            while( m_pTasks.try_pop( pTask ) )
            {
                // Execute
                pTask->Execute();

                // Cleanup
                if( pTask->pBatch )
                {
                    // Update task batch
                    pTask->pBatch->numReady++;

                    if( pTask->pBatch->numReady == pTask->pBatch->numTasks )
                    {
                        // Destroy the task batch
                        delete[] pTask->pBatch->pTasks;
                        delete pTask->pBatch;
                    }
                }
                else
                {
                    // Task was submitted independently
                    delete pTask;
                }
            }
        }
    };
}
