#include "oclRay.h"

RT::ocl::RayKernels::RayKernels()
    : IntersectPlane( nullptr )
{
}

RT::ocl::RayKernels::RayKernels( cl::Context context )
    : RayKernels()
{
    m_clRayProgram = cl::Program( context, s_pKernels );
    m_clRayProgram.build();

    IntersectPlane = cl::Kernel( m_clRayProgram, "IntersectPlane" );
}

const char* const RT::ocl::RayKernels::s_pKernels =
R"(
    struct RAY
    {
        float3 Origin;
        float3 Direction;
    };

    struct PLANE
    {
        float3 Origin;
        float3 Normal;
    };

    __kernel void IntersectPlane(
        __global const RAY*   pRays,
        __global const PLANE* pPlanes,
                 int          count,
        __global char*        pIntersects )
    {
        int id = get_global_linear_id();
        if( id < count )
        {
            const RAY* pThisRay = pRays + id;
            const PLANE* pThisPlane = pPlane + id;
            float denom = dot( 
        }
    }
)";
