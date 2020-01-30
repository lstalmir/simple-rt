#include "ApplicationBench.h"
#include <iostream>

#include "omp/ompRay.h"
#include "omp/ompRay2.h"

/***************************************************************************************\

Function:
    ApplicationBench::ApplicationBench

Description:
    Constructor

\***************************************************************************************/
RT::ApplicationBench::ApplicationBench( const RT::CommandLineArguments& cmdargs )
    : Application( cmdargs )
{
}

/***************************************************************************************\

Function:
    ApplicationBench::Run

Description:
    Run benchmarks.

\***************************************************************************************/
int RT::ApplicationBench::Run()
{
    std::cout << "RT_ENABLE_INTRINSICS = " << RT_ENABLE_INTRINSICS << "\n";
    std::cout << "RT_ENABLE_AVX = " << RT_ENABLE_AVX << "\n";
    std::cout << std::endl;

    Bench_OMP_Ray_Intersect_Plane();
    Bench_OMP_Ray2x2_Intersect_Plane();

    return 0;
}

#pragma optimize( "", off )

void RT::ApplicationBench::Bench_OMP_Ray_Intersect_Plane()
{
    static constexpr int NUM_INTERSECTIONS = 100000000;

    OMP::Ray ray[ 4 ];
    ray[ 0 ].Origin = RT::vec4( 0, 1, 0 );
    ray[ 0 ].Direction = RT::vec4( 1, -1, 0 );
    ray[ 1 ].Origin = RT::vec4( 0, 1, 0 );
    ray[ 1 ].Direction = RT::vec4( 1, 0, 0 );
    ray[ 2 ].Origin = RT::vec4( 0, 1, 0 );
    ray[ 2 ].Direction = RT::vec4( 0, 1, 0 );
    ray[ 3 ].Origin = RT::vec4( 0, 1, 0 );
    ray[ 3 ].Direction = RT::vec4( 0, 0, 1 );

    RT::OMP::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    BenchmarkBegin();
    for( int i = 0; i < NUM_INTERSECTIONS; ++i )
    {
        volatile RT::vec4 intersectionPoint0 = ray[ 0 ].Intersect( plane );
        volatile RT::vec4 intersectionPoint1 = ray[ 1 ].Intersect( plane );
        volatile RT::vec4 intersectionPoint2 = ray[ 2 ].Intersect( plane );
        volatile RT::vec4 intersectionPoint3 = ray[ 3 ].Intersect( plane );
    }
    BenchmarkEnd();

    ReportBenchmarkTime( std::cout << "Bench.OMP.Ray.Intersect.Plane: " );
}

void RT::ApplicationBench::Bench_OMP_Ray2x2_Intersect_Plane()
{
    static constexpr int NUM_INTERSECTIONS = 100000000;

    OMP::Ray2x2 ray;
    ray.Origin[ 0 ] = RT::vec4( 0, 1, 0 );
    ray.Direction[ 0 ] = RT::vec4( 1, -1, 0 );
    ray.Origin[ 1 ] = RT::vec4( 0, 1, 0 );
    ray.Direction[ 1 ] = RT::vec4( 1, 0, 0 );
    ray.Origin[ 2 ] = RT::vec4( 0, 1, 0 );
    ray.Direction[ 2 ] = RT::vec4( 0, 1, 0 );
    ray.Origin[ 3 ] = RT::vec4( 0, 1, 0 );
    ray.Direction[ 3 ] = RT::vec4( 0, 0, 1 );

    RT::OMP::Plane plane;
    plane.Origin = RT::vec4( 0 );
    plane.Normal = RT::vec4( 0, 1 );

    BenchmarkBegin();
    for( int i = 0; i < NUM_INTERSECTIONS; ++i )
    {
        volatile RT::vec4_2x2 intersectionPoints = ray.Intersect( plane );
    }
    BenchmarkEnd();

    ReportBenchmarkTime( std::cout << "Bench.OMP.Ray2x2.Intersect.Plane: " );
}
