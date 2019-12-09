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
#include <stack>
#include <omp.h>

namespace RT::OMP
{
    class Application : public RT::Application
    {
        using SceneTypes = RT::OMP::SceneTypes;
        using SceneTraits = RT::MakeSceneTraits<SceneTypes, RT::OMP::SceneFunctions<SceneTypes>>;

        enum class SecondaryRayType
        {
            ePrimary,
            eReflection,
            eRefraction
        };

        struct Intersection
        {
            vec4 m_Distance;
            Triangle m_Triangle;
            vec4 m_Color;
            // Synchronize access to this object
            std::mutex m_Mutex;

            inline Intersection() = default;
            inline Intersection( const Intersection& intersection )
                : m_Distance( intersection.m_Distance )
                , m_Triangle( intersection.m_Triangle )
                , m_Color( intersection.m_Color )
            {
            }
        };

        struct SecondaryRay
        {
            Ray m_Ray;
            int m_PrimaryRayIndex;
            int m_PreviousRayIndex;
            int m_Depth;
            Intersection m_Intersection;
            SecondaryRayType m_Type;
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
                    rayData.m_Intersection.m_Distance = vec4( std::numeric_limits<RT::float_t>::infinity() );
                    rayData.m_Type = SecondaryRayType::ePrimary;

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
                if( !RT_ENABLE_BOUNDING_BOXES || primaryRay.m_Ray.Intersect( object.BoundingBox ) )
                {
                    for( const auto& triangle : object.Triangles )
                    {
                        #if RT_ENABLE_BACKFACE_CULL
                        #if RT_ENABLE_INTRINSICS
                        __m128 D = _mm_load_ps( &primaryRay.m_Ray.Direction.data );
                        __m128 N = _mm_load_ps( &triangle.Normal.data );
                        D = Dot( D, N );
                        D = _mm_cmpge_ps( D, _mm_setzero_ps() );
                        if( _mm_cvtss_f32( D ) )
                        {
                            // Backface culling
                            continue;
                        }
                        #else
                        if( primaryRay.m_Ray.Direction.Dot( triangle.Normal ) >= 0 )
                        {
                            // Backface culling
                            continue;
                        }
                        #endif
                        #endif

                        // Ray-Triangle intersection
                        RT::vec4 intersection = primaryRay.m_Ray.Intersect( triangle );

                        if( intersection.x < primaryRay.m_Intersection.m_Distance.x )
                        {
                            // Update intersection
                            primaryRay.m_Intersection.m_Triangle = triangle;
                            primaryRay.m_Intersection.m_Distance = intersection;
                            primaryRay.m_Intersection.m_Color = object.Color;
                        }
                    }
                }
            }

            // Generate secondary rays
            if( primaryRay.m_Intersection.m_Distance.x < std::numeric_limits<RT::float_t>::infinity() )
            {
                // Trace shadows for this fragment
                m_Tasks.push( std::bind( &Application::LightIntersection, this, rayIndex, 0 ) );

                #if RT_MAX_RAY_DEPTH > 0
                if( primaryRay.m_Depth < RT_MAX_RAY_DEPTH )
                {
                    // Generate reflection and refraction rays
                    SecondaryRay reflectionRay;

                    // Calculate actual intersection point
                    vec4 intersectionPoint;
                    #if RT_ENABLE_INTRINSICS
                    __m128 O = _mm_load_ps( &primaryRay.m_Ray.Origin.data );
                    __m128 D = _mm_load_ps( &primaryRay.m_Ray.Direction.data );
                    __m128 F = _mm_set1_ps( primaryRay.m_Intersection.m_Distance.x );
                    O = _mm_fmadd_ps( D, F, O );
                    _mm_store_ps( &intersectionPoint.data, O );
                    #else
                    intersectionPoint = primaryRay.m_Ray.Origin + primaryRay.m_Ray.Direction * primaryRay.m_Intersection.m_Distance;
                    #endif

                    // Calculate normal in the intersection point
                    vec4 intersectionNormal;
                    #if RT_ENABLE_INTRINSICS
                    __m128 ONES = _mm_set1_ps( 1.f );
                    __m128 I = _mm_load_ps( &primaryRay.m_Intersection.m_Distance.data );
                    __m128 U = _mm_shuffle_ps( I, I, _MM_SHUFFLE( 1, 1, 1, 1 ) );
                    __m128 V = _mm_shuffle_ps( I, I, _MM_SHUFFLE( 2, 2, 2, 2 ) );
                    __m128 N0 = _mm_load_ps( &primaryRay.m_Intersection.m_Triangle.An.data );
                    __m128 N1 = _mm_load_ps( &primaryRay.m_Intersection.m_Triangle.Bn.data );
                    __m128 N2 = _mm_load_ps( &primaryRay.m_Intersection.m_Triangle.Cn.data );
                    N1 = _mm_mul_ps( N1, U );
                    N2 = _mm_mul_ps( N2, V );
                    N1 = _mm_add_ps( N1, N2 );
                    U = _mm_add_ps( U, V );
                    U = _mm_sub_ps( ONES, U );
                    __m128 N = _mm_fmadd_ps( N0, U, N1 );
                    N = Normalize3( N );
                    _mm_store_ps( &intersectionNormal.data, N );
                    #else
                    // TODO
                    #endif

                    reflectionRay.m_Ray = primaryRay.m_Ray.Reflect( intersectionNormal, intersectionPoint );
                    reflectionRay.m_Depth = primaryRay.m_Depth + 1;
                    reflectionRay.m_PrimaryRayIndex = primaryRay.m_PrimaryRayIndex;
                    reflectionRay.m_PreviousRayIndex = rayIndex;
                    reflectionRay.m_Intersection.m_Distance.x = std::numeric_limits<RT::float_t>::infinity();
                    reflectionRay.m_Type = SecondaryRayType::eReflection;

                    auto reflectionRayIterator = m_Rays.push_back( reflectionRay );
                    const int reflectionRayIndex = reflectionRayIterator - m_Rays.begin();

                    m_Tasks.push( std::bind( &Application::ObjectIntersection, this, reflectionRayIndex ) );
                }
                #endif
            }
        }

        void LightIntersection( int rayIndex, int lightIndex )
        {
            auto& primaryRay = m_Rays[rayIndex];
            const auto& light = m_Scene.Lights[lightIndex];

            #if RT_ENABLE_INTRINSICS
            const __m128 lightLevel = _mm_set1_ps( 1.f );
            const __m128 shadowLevel = _mm_set1_ps( 0.2f );
            __m128 lightIntensity = lightLevel;
            #else
            const float lightLevel = 1.f;
            const float shadowLevel = 0.2f;
            float lightIntensity = lightLevel;
            #endif

            const auto shadowRays = light.SpawnSecondaryRays(
                primaryRay.m_Ray,
                primaryRay.m_Intersection.m_Distance.x );

            for( const auto& shadowRay : shadowRays )
            {
                bool inTheShadows = false;

                for( const auto& object : m_Scene.Objects )
                {
                    if( !RT_ENABLE_BOUNDING_BOXES || shadowRay.Intersect( object.BoundingBox ) )
                    {
                        // Bounding-box test passed, check each triangle to check if the ray actually intersects the object
                        // Since we are in light ray pass, only one intersection is enough
                        for( const auto& triangle : object.Triangles )
                        {
                            #if RT_ENABLE_BACKFACE_CULL
                            #if RT_ENABLE_INTRINSICS
                            __m128 D = _mm_load_ps( &shadowRay.Direction.data );
                            __m128 N = _mm_load_ps( &triangle.Normal.data );
                            D = Dot( D, N );
                            D = _mm_cmpge_ps( D, _mm_setzero_ps() );
                            if( _mm_cvtss_f32( D ) )
                            {
                                // Backface culling
                                continue;
                            }
                            #else
                            if( shadowRay.Direction.Dot( triangle.Normal ) >= 0 )
                            {
                                // Backface culling
                                continue;
                            }
                            #endif
                            #endif

                            RT::vec4 intersectionPoint = shadowRay.Intersect( triangle );

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

                #if RT_ENABLE_INTRINSICS
                lightIntensity = _mm_add_ps( lightIntensity, !inTheShadows ? lightLevel : shadowLevel );
                #else
                lightIntensity += !inTheShadows ? lightLevel : shadowLevel;
                #endif
            }

            // Distance to the light
            #if RT_ENABLE_INTRINSICS
            {
                __m128 rayOrigin = _mm_load_ps( &primaryRay.m_Ray.Origin.data );
                __m128 lightPosition = _mm_load_ps( &light.Position.data );

                __m128 distance;
                distance = _mm_sub_ps( lightPosition, rayOrigin );
                distance = Length3( distance );
                distance = _mm_mul_ps( distance, _mm_set1_ps( 0.001f ) );
                distance = _mm_pow_ps( distance, _mm_set1_ps( 2.0f ) );

                __m128i iLightSubdivs = _mm_set1_epi32( light.Subdivs );
                __m128 lightSubdivs = _mm_cvtepi32_ps( iLightSubdivs );

                distance = _mm_mul_ps( distance, lightSubdivs );
                lightIntensity = _mm_div_ps( lightIntensity, distance );

                __m128 color;

                std::scoped_lock lock( primaryRay.m_Intersection.m_Mutex );
                color = _mm_load_ps( &primaryRay.m_Intersection.m_Color.data );
                color = _mm_mul_ps( color, lightIntensity );
                _mm_store_ps( &primaryRay.m_Intersection.m_Color.data, color );
            }
            #else
            {
                const float distance = (light.Position - primaryRay.m_Ray.Origin).Length3() / 100.0f;
                lightIntensity /= light.Subdivs * std::pow( distance, 4 );

                // Synchronize access to the intersection parameters
                std::scoped_lock lock( primaryRay.m_Intersection.m_Mutex );

                primaryRay.m_Intersection.m_Color.x *= lightIntensity;
                primaryRay.m_Intersection.m_Color.y *= lightIntensity;
                primaryRay.m_Intersection.m_Color.z *= lightIntensity;
            }
            #endif

            #if RT_MAX_RAY_DEPTH > 0
            if( primaryRay.m_Depth == RT_MAX_RAY_DEPTH )
            {
                // End of ray path
                // Iterate over all rays and update colors
                int previousRayIndex = primaryRay.m_PreviousRayIndex;

                while( previousRayIndex != -1 )
                {
                    auto& currentRay = m_Rays[rayIndex];
                    auto& previousRay = m_Rays[previousRayIndex];

                    switch( currentRay.m_Type )
                    {
                    case SecondaryRayType::eReflection:
                    {
                        // Compute reflection coefficient
                        const float kr = previousRay.m_Ray.Fresnel( previousRay.m_Intersection.m_Triangle.Normal, 3.5f );

                        // Update the color
                        std::scoped_lock lk( previousRay.m_Intersection.m_Mutex );
                        previousRay.m_Intersection.m_Color =
                            previousRay.m_Intersection.m_Color.Lerp( currentRay.m_Intersection.m_Color, kr );

                        break;
                    }
                    }

                    rayIndex = previousRayIndex;
                    previousRayIndex = previousRay.m_PreviousRayIndex;
                }
            }
            #endif
        }
    };
}
